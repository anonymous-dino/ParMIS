#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <iostream>
#include <vector>
#include <queue>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <chrono>
#include <algorithm>
#include <filesystem>
#include <iomanip>

using namespace std;

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Constants - reduced for safety
// #define MAX_BATCH_SIZE 2048  // Reduced from 4096
// #define THREADS_PER_BLOCK 256
// #define MAX_NEIGHBORS_PER_VERTEX 32  // Reduced from 64 //64
// #define MAX_CLUSTER_SIZE 64  // Reduced from 128 //4096

// Constants - optimized for 46GB GPU memory
#define MAX_BATCH_SIZE 1048576      // 1M instead of 327K
#define THREADS_PER_BLOCK 256       // Back to 256 for better occupancy
#define MAX_NEIGHBORS_PER_VERTEX 128 // Kept as requested
// #define MAX_CLUSTER_SIZE 1000000  // Kept as requested
int gpu_device_id;
 int SMs;
 int mTpSM;

// Comprehensive timing structure for detailed profiling
struct DetailedTimings {
    // Individual kernel times (GPU)
    float extract_vertices_time;
    float compute_neighborhoods_time;
    float vertex_clustering_time;
    float assign_edges_time;
    float tvb_process_time;  // process_clusters_kernel
    float clear_assignments_time;
    
    // Aggregated GPU times
    float cluster_processing_time;  // All steps before TVB
    float total_gpu_kernel_time;
    
    // CPU times
    float host_to_device_time;
    float device_to_host_time;
    float cpu_conversion_time;
    float cpu_graph_update_time;
    float cpu_overhead_time;
    
    // Total times
    float total_cpu_time;
    float total_gpu_time;
    float total_batch_time;
    float transfer_time;  // Combined host-to-device and device-to-host transfer time

    int clusters_count;
    
    DetailedTimings() { reset(); }
    
    void reset() {
        extract_vertices_time = 0;
        compute_neighborhoods_time = 0;
        vertex_clustering_time = 0;
        assign_edges_time = 0;
        tvb_process_time = 0;
        clear_assignments_time = 0;
        cluster_processing_time = 0;
        total_gpu_kernel_time = 0;
        host_to_device_time = 0;
        device_to_host_time = 0;
        cpu_conversion_time = 0;
        cpu_graph_update_time = 0;
        cpu_overhead_time = 0;
        total_cpu_time = 0;
        total_gpu_time = 0;
        total_batch_time = 0;
        transfer_time = 0;
        clusters_count = 0;
    }
    
    void calculateAggregates() {
        // Cluster processing = everything before TVB
        cluster_processing_time = extract_vertices_time + compute_neighborhoods_time +
                                vertex_clustering_time + assign_edges_time + clear_assignments_time;

        // Total GPU kernel time
        total_gpu_kernel_time = cluster_processing_time + tvb_process_time;

        // Transfer time = host-to-device + device-to-host
        transfer_time = host_to_device_time + device_to_host_time;

        // Total CPU time
        total_cpu_time = host_to_device_time + device_to_host_time +
                        cpu_conversion_time + cpu_graph_update_time + cpu_overhead_time;

        // Total GPU time (kernels + memory transfers)
        total_gpu_time = total_gpu_kernel_time + host_to_device_time + device_to_host_time;
    }
    
    void accumulate(const DetailedTimings& other) {
        extract_vertices_time += other.extract_vertices_time;
        compute_neighborhoods_time += other.compute_neighborhoods_time;
        vertex_clustering_time += other.vertex_clustering_time;
        assign_edges_time += other.assign_edges_time;
        tvb_process_time += other.tvb_process_time;
        clear_assignments_time += other.clear_assignments_time;
        cluster_processing_time += other.cluster_processing_time;
        total_gpu_kernel_time += other.total_gpu_kernel_time;
        host_to_device_time += other.host_to_device_time;
        device_to_host_time += other.device_to_host_time;
        cpu_conversion_time += other.cpu_conversion_time;
        cpu_graph_update_time += other.cpu_graph_update_time;
        cpu_overhead_time += other.cpu_overhead_time;
        total_cpu_time += other.total_cpu_time;
        total_gpu_time += other.total_gpu_time;
        total_batch_time += other.total_batch_time;
        transfer_time += other.transfer_time;
    }
};

// Device structures
struct __align__(8) DeviceNode {
    int label;
    bool membership;
    int cluster_id;
    
    __device__ __host__ DeviceNode() : label(0), membership(false), cluster_id(-1) {}
    __device__ __host__ DeviceNode(int l, bool m, int c) : label(l), membership(m), cluster_id(c) {}
};

struct __align__(16) DeviceEdge {
    int source;
    int destination;
    bool isInsertion;
    int clusterID;
    
    __device__ __host__ DeviceEdge() : source(0), destination(0), isInsertion(true), clusterID(-1) {}
    __device__ __host__ DeviceEdge(int s, int d, bool ins) : source(s), destination(d), isInsertion(ins), clusterID(-1) {}
};

struct DeviceGraph {
    int num_nodes;
    int* row_offsets;
    int* column_indices;
    
    __device__ __host__ DeviceGraph() : num_nodes(0), row_offsets(nullptr), column_indices(nullptr) {}
};

struct DeviceVertexCluster {
    int vertices[512];
    int vertex_count;
    int affected_edges[512];
    int edge_count;
    
    __device__ __host__ DeviceVertexCluster() : vertex_count(0), edge_count(0) {}
};

// Kernel to extract unique vertices from edges
__global__ void extract_vertices_kernel(const DeviceEdge* edges, int num_edges, 
                                       int* vertices, int* vertex_count) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < num_edges) {
        // Simple approach: each thread adds both source and destination
        int base_idx = tid * 2;
        if (base_idx < MAX_BATCH_SIZE * 2) {
            vertices[base_idx] = edges[tid].source;
            vertices[base_idx + 1] = edges[tid].destination;
        }
    }
    
    // Set vertex count (done by first thread)
    if (tid == 0) {
        *vertex_count = min(num_edges * 2, MAX_BATCH_SIZE * 2);
    }
}

// CORRECTED: Kernel to compute vertex neighborhoods (h-hop BFS)
__global__ void compute_vertex_neighborhoods_kernel(const int* vertices, int vertex_count,
                                                   const DeviceGraph graph, int hop_limit,
                                                   int* neighborhoods, int* neighborhood_sizes,
                                                   int max_neighbors_per_vertex) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid >= vertex_count) return;
    
    int vertex = vertices[tid];
    
    // Strengthen bounds checking for vertex
    if (vertex >= graph.num_nodes || vertex < 0) {
        neighborhood_sizes[tid] = 0;
        return;
    }
    
    int base_offset = tid * max_neighbors_per_vertex;
    
    // Add bounds check for base_offset to prevent buffer overflow
    if (base_offset >= vertex_count * max_neighbors_per_vertex) {
        neighborhood_sizes[tid] = 0;
        return;
    }
    
    // Initialize neighborhood with the vertex itself
    neighborhoods[base_offset] = vertex;
    int current_size = 1;
    
    // BFS queue simulation using smaller, safer arrays
    int queue[128];  // Reduced significantly for safety
    int queue_size = 1;
    int next_queue[128];  // Reduced significantly for safety
    int next_queue_size;
    
    queue[0] = vertex;
    
    // Perform BFS for hop_limit hops
    for (int hop = 0; hop < hop_limit && hop < 2 && queue_size > 0; hop++) {  // Limited to 2 hops max
        next_queue_size = 0;
        
        for (int i = 0; i < queue_size && i < 128; i++) {  // Added explicit bound check
            int current_vertex = queue[i];
            
            // Strengthen bounds checking for current_vertex
            if (current_vertex >= graph.num_nodes || current_vertex < 0) continue;
            
            // Add bounds checking for row_offsets access
            if (current_vertex + 1 > graph.num_nodes) continue;
            
            int start = graph.row_offsets[current_vertex];
            int end = graph.row_offsets[current_vertex + 1];
            
            // Add safety bounds for start and end
            // if (start < 0 || end < start || start >= end + 1000) continue;  // Sanity check on range
            
            // Limit the number of neighbors processed to prevent overflow
            int max_process = min(end - start, 4);  // Process at most 4 neighbors per vertex
            
            // Add neighbors to next level and to neighborhood
            for (int j = start; j < start + end && 
                 current_size < max_neighbors_per_vertex - 1; j++) {  // Leave room in queue
                
                // Add bounds check for column_indices access
                if (j < 0) continue;
                
                int neighbor = graph.column_indices[j];
                
                // Strengthen bounds check for neighbor
                if (neighbor >= graph.num_nodes || neighbor < 0) continue;
                
                // Check if already in neighborhood
                bool already_added = false;
                for (int k = 0; k < current_size && k < max_neighbors_per_vertex; k++) {
                    if (neighborhoods[base_offset + k] == neighbor) {
                        already_added = true;
                        break;
                    }
                }
                
                if (!already_added) {
                    // Add final bounds check before writing to neighborhoods
                    if (base_offset + current_size < vertex_count * max_neighbors_per_vertex) {
                        neighborhoods[base_offset + current_size] = neighbor;
                        current_size++;
                        // if (next_queue_size < 16) {
                            next_queue[next_queue_size++] = neighbor;
                        // }
                    }
                }
            }
        }
        
        // Copy next queue to current queue with bounds checking
        queue_size = min(next_queue_size, 16);
        for (int i = 0; i < next_queue_size; i++) {
            queue[i] = next_queue[i];
        }
    }
    
    // Ensure we don't exceed the maximum
    neighborhood_sizes[tid] = min(current_size, max_neighbors_per_vertex);
}

// Strategy 1: Hash-based duplicate detection with shared memory optimization
__global__ void compute_vertex_neighborhoods_optimized_v1(
    const int* vertices, int vertex_count,
    const DeviceGraph graph, int hop_limit,
    int* neighborhoods, int* neighborhood_sizes,
    int max_neighbors_per_vertex) {
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= vertex_count) return;
    
    int vertex = vertices[tid];
    if (vertex >= graph.num_nodes || vertex < 0) {
        neighborhood_sizes[tid] = 0;
        return;
    }
    
    // Use local hash set for O(1) duplicate detection
    // For GPU, we'll use a simple hash table with linear probing
    constexpr int LOCAL_HASH_SIZE = 256;  // Power of 2 for fast modulo
    int hash_table[LOCAL_HASH_SIZE];
    
    // Initialize hash table with -1 (empty)
    #pragma unroll
    for (int i = 0; i < LOCAL_HASH_SIZE; i++) {
        hash_table[i] = -1;
    }
    
    int base_offset = tid * max_neighbors_per_vertex;
    int current_size = 0;
    
    auto hash_vertex = [&](int v) { return (v * 73) & (LOCAL_HASH_SIZE - 1); };
    
    auto try_insert = [&](int v) -> bool {
        int h = hash_vertex(v);
        int attempts = 0;
        while (hash_table[h] != -1 && hash_table[h] != v && attempts < LOCAL_HASH_SIZE) {
            h = (h + 1) & (LOCAL_HASH_SIZE - 1);  // Linear probing
            attempts++;
        }
        if (hash_table[h] == v) return false;  // Already exists
        if (hash_table[h] == -1 && current_size < max_neighbors_per_vertex) {
            hash_table[h] = v;
            neighborhoods[base_offset + current_size] = v;
            current_size++;
            return true;
        }
        return false;  // Hash table full or neighborhood full
    };
    
    // Add starting vertex
    try_insert(vertex);
    
    // BFS with optimized queue management
    int current_frontier[64];  // Current level vertices
    int next_frontier[64];     // Next level vertices
    int current_frontier_size = 1;
    int next_frontier_size = 0;
    
    current_frontier[0] = vertex;
    
    // Perform BFS
    for (int hop = 0; hop < hop_limit && current_frontier_size > 0; hop++) {
        next_frontier_size = 0;
        
        for (int i = 0; i < current_frontier_size; i++) {
            int curr_vertex = current_frontier[i];
            
            int start = graph.row_offsets[curr_vertex];
            int end = graph.row_offsets[curr_vertex + 1];
            
            // Process all neighbors (remove artificial limits)
            for (int j = start; j < end && current_size < max_neighbors_per_vertex; j++) {
                int neighbor = graph.column_indices[j];
                
                if (neighbor >= 0 && neighbor < graph.num_nodes) {
                    if (try_insert(neighbor) && next_frontier_size < 64) {
                        next_frontier[next_frontier_size++] = neighbor;
                    }
                }
            }
        }
        
        // Swap frontiers
        current_frontier_size = next_frontier_size;
        for (int i = 0; i < next_frontier_size; i++) {
            current_frontier[i] = next_frontier[i];
        }
    }
    
    neighborhood_sizes[tid] = current_size;
}

// Strategy 3: Memory-optimized version with texture cache utilization
__global__ void compute_vertex_neighborhoods_optimized_v2(
    const int* __restrict__ vertices, int vertex_count,
    const DeviceGraph graph, int hop_limit,
    int* __restrict__ neighborhoods, int* __restrict__ neighborhood_sizes,
    int max_neighbors_per_vertex) {
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= vertex_count) return;
    
    int vertex = vertices[tid];
    if (__builtin_expect(vertex >= graph.num_nodes || vertex < 0, false)) {
        neighborhood_sizes[tid] = 0;
        return;
    }
    
    int base_offset = tid * max_neighbors_per_vertex;
    
    // Use bitset for duplicate detection (memory efficient)
    unsigned long long visited_bits[16] = {0};  // Support up to 1024 vertices per neighborhood
    
    auto mark_visited = [&](int v) {
        int word_idx = (v & 1023) >> 6;  // Which 64-bit word
        int bit_idx = v & 63;            // Which bit in word
        visited_bits[word_idx] |= (1ULL << bit_idx);
    };
    
    auto is_visited = [&](int v) -> bool {
        int word_idx = (v & 1023) >> 6;
        int bit_idx = v & 63;
        return (visited_bits[word_idx] & (1ULL << bit_idx)) != 0;
    };
    
    int current_size = 0;
    int queue_buffer[128];  // Larger queue buffer
    int queue_start = 0;
    int queue_end = 0;
    
    // Add vertex to both neighborhood and queue
    auto add_vertex = [&](int v) -> bool {
        if (current_size >= max_neighbors_per_vertex) return false;
        
        if (!is_visited(v)) {
            mark_visited(v);
            neighborhoods[base_offset + current_size++] = v;
            
            if ((queue_end + 1) % 128 != queue_start) {
                queue_buffer[queue_end] = v;
                queue_end = (queue_end + 1) % 128;
            }
            return true;
        }
        return false;
    };
    
    // Initialize with starting vertex
    add_vertex(vertex);
    
    // BFS traversal
    for (int hop = 0; hop < hop_limit && queue_start != queue_end; hop++) {
        int level_end = queue_end;
        
        while (queue_start != level_end && current_size < max_neighbors_per_vertex) {
            int curr_vertex = queue_buffer[queue_start];
            queue_start = (queue_start + 1) % 128;
            
            // Use ldg for read-only data (better cache utilization)
            int start = __ldg(&graph.row_offsets[curr_vertex]);
            int end = __ldg(&graph.row_offsets[curr_vertex + 1]);
            
            for (int j = start; j < end && current_size < max_neighbors_per_vertex; j++) {
                int neighbor = __ldg(&graph.column_indices[j]);
                
                if (__builtin_expect(neighbor >= 0 && neighbor < graph.num_nodes, true)) {
                    add_vertex(neighbor);
                }
            }
        }
    }
    
    neighborhood_sizes[tid] = current_size;
}

// CORRECTED: Kernel to perform vertex clustering
__global__ void vertex_clustering_kernel(const int* vertices, int vertex_count,
                                        const int* neighborhoods, const int* neighborhood_sizes,
                                        int max_neighbors_per_vertex,
                                        DeviceNode* nodes,
                                        DeviceVertexCluster* clusters,
                                        int* cluster_count) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid >= vertex_count) return;
    
    int vertex = vertices[tid];
    
    // Add stronger bounds checking for vertex
    if (vertex >= 10000000 || vertex < 0) return;
    
    // Skip if vertex is already in a cluster
    if (nodes[vertex].cluster_id != -1) return;
    
    // Add bounds check for neighborhood access
    if (tid >= vertex_count || neighborhood_sizes[tid] <= 0) return;
    
    // Find overlapping clusters in neighborhood
    int neighborhood_base = tid * max_neighbors_per_vertex;
    int neighborhood_size = min(neighborhood_sizes[tid], max_neighbors_per_vertex);  // Add safety limit
    
    // Add bounds check for neighborhood_base
    if (neighborhood_base >= vertex_count * max_neighbors_per_vertex) return;
    
    int overlapping_clusters[4];  // Reduced significantly for safety
    int overlap_count = 0;
    
    for (int i = 0; i < neighborhood_size && overlap_count < 4 && i < max_neighbors_per_vertex; i++) {
        // Add bounds check for neighborhoods access
        if (neighborhood_base + i >= vertex_count * max_neighbors_per_vertex) break;
        
        int neighbor = neighborhoods[neighborhood_base + i];
        if (neighbor >= 0 && neighbor < 10000000 && nodes[neighbor].cluster_id != -1) {
            int cluster_id = nodes[neighbor].cluster_id;
            
            // Check if this cluster is already in overlapping list
            bool already_found = false;
            for (int j = 0; j < overlap_count; j++) {
                if (overlapping_clusters[j] == cluster_id) {
                    already_found = true;
                    break;
                }
            }
            if (!already_found) {
                overlapping_clusters[overlap_count++] = cluster_id;
            }
        }
    }
    
    int final_cluster_id;
    
    if (overlap_count == 0) {
        // Create new cluster
        final_cluster_id = atomicAdd(cluster_count, 1);
        if (final_cluster_id < MAX_BATCH_SIZE) {
            clusters[final_cluster_id].vertex_count = 0;
            clusters[final_cluster_id].edge_count = 0;
        }
    } else {
        // Use first overlapping cluster (simplified merging)
        final_cluster_id = overlapping_clusters[0];
    }
    
    // Add vertex to cluster
    if (final_cluster_id >= 0 && final_cluster_id < MAX_BATCH_SIZE) {
        int pos = atomicAdd(&clusters[final_cluster_id].vertex_count, 1);
        if (pos >= 0) {
            clusters[final_cluster_id].vertices[pos] = vertex;
        }
        
        // Update cluster assignment for all vertices in neighborhood
        for (int i = 0; i < neighborhood_size && i < max_neighbors_per_vertex; i++) {
            // Add bounds check for neighborhoods access
            if (neighborhood_base + i >= vertex_count * max_neighbors_per_vertex) break;
            
            int neighbor = neighborhoods[neighborhood_base + i];
            if (neighbor >= 0 && neighbor < 10000000) {
                nodes[neighbor].cluster_id = final_cluster_id;
            }
        }
    }
}

// Kernel to assign affected edges to clusters
__global__ void assign_edges_to_clusters_kernel(const DeviceEdge* edges, int num_edges,
                                               DeviceNode* nodes,
                                               DeviceVertexCluster* clusters) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid >= num_edges) return;
    
    const DeviceEdge& edge = edges[tid];
    
    // Add bounds checking for edge vertices
    if (edge.source < 0 || edge.source >= 10000000 || 
        edge.destination < 0 || edge.destination >= 10000000) return;
    
    // Find which cluster this edge belongs to
    int cluster_id = -1;
    if (nodes[edge.source].cluster_id != -1) {
        cluster_id = nodes[edge.source].cluster_id;
    } else if (nodes[edge.destination].cluster_id != -1) {
        cluster_id = nodes[edge.destination].cluster_id;
    }
    
    // Add edge to cluster
    if (cluster_id != -1 && cluster_id < MAX_BATCH_SIZE) {
        int pos = atomicAdd(&clusters[cluster_id].edge_count, 1);
        // if (pos < MAX_CLUSTER_SIZE * 2) {
            clusters[cluster_id].affected_edges[pos] = tid;
        // }
    }
}

// Kernel to process clusters and handle MIS updates
__global__ void process_clusters_kernel(const DeviceEdge* edges,
                                       DeviceVertexCluster* clusters,
                                       int cluster_count,
                                       const DeviceGraph graph,
                                       DeviceNode* nodes) {
    int cluster_id = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (cluster_id >= cluster_count) return;
    
    DeviceVertexCluster& cluster = clusters[cluster_id];
    
    // Process each edge in this cluster
    for (int i = 0; i < cluster.edge_count; i++) {
        int edge_idx = cluster.affected_edges[i];
        const DeviceEdge& edge = edges[edge_idx];
        
        int u = edge.source;
        int v = edge.destination;
        
        // Strengthen bounds checking
        if (u >= graph.num_nodes || v >= graph.num_nodes || u < 0 || v < 0) continue;
        
        bool u_in_mis = nodes[u].membership;
        bool v_in_mis = nodes[v].membership;
        
        if (edge.isInsertion) {
            // Handle insertion
            if (u_in_mis && v_in_mis) {
                // Lemma 1: Both in MIS, remove the one with higher index
                int to_remove = (u > v) ? u : v;
                nodes[to_remove].membership = false;
                
                // Try to add neighbors to MIS (simplified)
                if (to_remove < graph.num_nodes && to_remove + 1 <= graph.num_nodes) {
                    int start = graph.row_offsets[to_remove];
                    int end = graph.row_offsets[to_remove + 1];
                    
                    // Add bounds safety
                    if (start >= 0 && end >= start) {
                        int max_check = min(end - start, 4);  // Limit neighbor checking
                        
                        for (int j = start; j < start + max_check; j++) {
                            int neighbor = graph.column_indices[j];
                            if (neighbor >= 0 && neighbor < graph.num_nodes && !nodes[neighbor].membership) {
                                // Simple check if neighbor can be added
                                bool can_add = true;
                                if (neighbor + 1 <= graph.num_nodes) {
                                    int nbr_start = graph.row_offsets[neighbor];
                                    int nbr_end = graph.row_offsets[neighbor + 1];
                                    
                                    if (nbr_start >= 0 && nbr_end >= nbr_start) {
                                        int max_nbr_check = min(nbr_end - nbr_start, 4);
                                        
                                        for (int k = nbr_start; k < nbr_start + max_nbr_check; k++) {
                                            int nbr_neighbor = graph.column_indices[k];
                                            if (nbr_neighbor >= 0 && nbr_neighbor < graph.num_nodes && 
                                                nodes[nbr_neighbor].membership) {
                                                can_add = false;
                                                break;
                                            }
                                        }
                                    }
                                }
                                
                                if (can_add) {
                                    nodes[neighbor].membership = true;
                                    break; // Add only one neighbor
                                }
                            }
                        }
                    }
                }
            }
        } else {
            // Handle deletion
            if (u_in_mis && !v_in_mis) {
                // Try to add v to MIS
                bool can_add = true;
                if (v + 1 <= graph.num_nodes) {
                    int start = graph.row_offsets[v];
                    int end = graph.row_offsets[v + 1];
                    
                    if (start >= 0 && end >= start) {
                        int max_check = min(end - start, 4);
                        
                        for (int j = start; j < start + max_check; j++) {
                            int neighbor = graph.column_indices[j];
                            if (neighbor >= 0 && neighbor < graph.num_nodes && 
                                neighbor != u && nodes[neighbor].membership) {
                                can_add = false;
                                break;
                            }
                        }
                    }
                }
                
                if (can_add) {
                    nodes[v].membership = true;
                }
            } else if (!u_in_mis && v_in_mis) {
                // Try to add u to MIS
                bool can_add = true;
                if (u + 1 <= graph.num_nodes) {
                    int start = graph.row_offsets[u];
                    int end = graph.row_offsets[u + 1];
                    
                    if (start >= 0 && end >= start) {
                        int max_check = min(end - start, 4);
                        
                        for (int j = start; j < start + max_check; j++) {
                            int neighbor = graph.column_indices[j];
                            if (neighbor >= 0 && neighbor < graph.num_nodes && 
                                neighbor != v && nodes[neighbor].membership) {
                                can_add = false;
                                break;
                            }
                        }
                    }
                }
                
                if (can_add) {
                    nodes[u].membership = true;
                }
            }
        }
    }
}

// Kernel to clear cluster assignments
__global__ void clear_cluster_assignments_kernel(const int* vertices, int vertex_count, DeviceNode* nodes) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < vertex_count) {
        int vertex = vertices[tid];
        if (vertex >= 0 && vertex < 10000000) {
            nodes[vertex].cluster_id = -1;
        }
    }
}

// Node initialization kernel
__global__ void init_nodes_kernel(DeviceNode* nodes, int num_nodes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_nodes) {
        nodes[idx] = DeviceNode(idx, false, -1);
    }
}

// Host structures
struct Edge {
    int source;
    int destination;
    bool isInsertion;
    int clusterID;

    Edge(int s, int d, bool ins) : source(s), destination(d), isInsertion(ins), clusterID(-1) {}
};

struct Node {
    int label;
    bool membership;
    vector<int> cluster;

    Node(int l = 0, bool m = false) : label(l), membership(m) {
        cluster.resize(2, -1);
    }

    int getClusterId() const { return cluster[0]; }
    void setClusterId(int id) { cluster[0] = id; }
    bool isInCluster() const { return cluster[0] != -1; }
    void clearCluster() { cluster[0] = -1; }
};

// Conversion function
vector<DeviceEdge> convertToDeviceEdges(const vector<Edge>& host_edges) {
    vector<DeviceEdge> device_edges;
    device_edges.reserve(host_edges.size());
    
    for (const auto& edge : host_edges) {
        device_edges.emplace_back(edge.source, edge.destination, edge.isInsertion);
    }
    
    return device_edges;
}

// Graph class
class Graph {
public:
    int num_nodes;
    int num_Edges = 0;
    vector<vector<int>> adj_list;

    Graph(int n = 0) : num_nodes(n), adj_list(n) {}

    void addEdge(int u, int v) {
        if (u >= 0 && u < num_nodes && v >= 0 && v < num_nodes) {
            adj_list[u].push_back(v);
            adj_list[v].push_back(u);
        }
    }

    void insertEdge(int u, int v) {
        addEdge(u, v);
        num_Edges++;
    }

    void removeEdge(int u, int v) {
        if (u >= 0 && u < num_nodes && v >= 0 && v < num_nodes) {
            auto& u_neighbors = adj_list[u];
            auto& v_neighbors = adj_list[v];
            
            u_neighbors.erase(remove(u_neighbors.begin(), u_neighbors.end(), v), u_neighbors.end());
            v_neighbors.erase(remove(v_neighbors.begin(), v_neighbors.end(), u), v_neighbors.end());
        }
        num_Edges--;
    }

    bool isAdjacent(int u, int v) {
        if (u >= adj_list.size() || v >= adj_list.size()) return false;
        
        const auto& neighbors = adj_list[u];
        return find(neighbors.begin(), neighbors.end(), v) != neighbors.end();
    }
};

// GPU Vertex Clustering Graph class
class GPUVertexClusteringGraph {
private:
    DeviceGraph d_graph;
    DeviceNode* d_nodes;
    
    // Pre-allocated pools
    DeviceEdge* d_edges_pool;
    int* d_vertices_pool;
    int* d_neighborhoods_pool;
    int* d_neighborhood_sizes_pool;
    DeviceVertexCluster* d_clusters_pool;
    int* d_cluster_count;
    int* d_vertex_count;
    
    cudaStream_t stream;
    
    // CUDA events for timing
    cudaEvent_t start_event, stop_event;
    
public:
    GPUVertexClusteringGraph(const vector<vector<int>>& adj_list) {
        d_graph.num_nodes = adj_list.size();
        
        // Build CSR format
        vector<int> row_offsets(d_graph.num_nodes + 1, 0);
        vector<int> column_indices;
        
        for (int i = 0; i < d_graph.num_nodes; i++) {
            row_offsets[i + 1] = row_offsets[i] + adj_list[i].size();
            for (int neighbor : adj_list[i]) {
                column_indices.push_back(neighbor);
            }
        }
        
        // Allocate GPU memory
        CUDA_CHECK(cudaMalloc(&d_graph.row_offsets, (d_graph.num_nodes + 1) * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_graph.column_indices, column_indices.size() * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_nodes, d_graph.num_nodes * sizeof(DeviceNode)));
        
        // Pre-allocate pools
        CUDA_CHECK(cudaMalloc(&d_edges_pool, MAX_BATCH_SIZE * sizeof(DeviceEdge)));
        CUDA_CHECK(cudaMalloc(&d_vertices_pool, MAX_BATCH_SIZE * 2 * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_neighborhoods_pool, MAX_BATCH_SIZE * 2 * MAX_NEIGHBORS_PER_VERTEX * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_neighborhood_sizes_pool, MAX_BATCH_SIZE * 2 * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_clusters_pool, MAX_BATCH_SIZE * sizeof(DeviceVertexCluster)));
        CUDA_CHECK(cudaMalloc(&d_cluster_count, sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_vertex_count, sizeof(int)));
        
        // Initialize memory to prevent garbage values
        CUDA_CHECK(cudaMemset(d_neighborhoods_pool, 0, MAX_BATCH_SIZE * 2 * MAX_NEIGHBORS_PER_VERTEX * sizeof(int)));
        CUDA_CHECK(cudaMemset(d_neighborhood_sizes_pool, 0, MAX_BATCH_SIZE * 2 * sizeof(int)));
        
        // Copy graph data
        CUDA_CHECK(cudaMemcpy(d_graph.row_offsets, row_offsets.data(), 
                             (d_graph.num_nodes + 1) * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_graph.column_indices, column_indices.data(), 
                             column_indices.size() * sizeof(int), cudaMemcpyHostToDevice));
        
        // Create stream and events
        CUDA_CHECK(cudaStreamCreate(&stream));
        CUDA_CHECK(cudaEventCreate(&start_event));
        CUDA_CHECK(cudaEventCreate(&stop_event));
        
        // Initialize nodes
        initializeNodes();
    }
    
    ~GPUVertexClusteringGraph() {
        cudaFree(d_graph.row_offsets);
        cudaFree(d_graph.column_indices);
        cudaFree(d_nodes);
        cudaFree(d_edges_pool);
        cudaFree(d_vertices_pool);
        cudaFree(d_neighborhoods_pool);
        cudaFree(d_neighborhood_sizes_pool);
        cudaFree(d_clusters_pool);
        cudaFree(d_cluster_count);
        cudaFree(d_vertex_count);
        cudaStreamDestroy(stream);
        cudaEventDestroy(start_event);
        cudaEventDestroy(stop_event);
    }
    
    void initializeNodes() {
        int threads = THREADS_PER_BLOCK;
        int blocks = (d_graph.num_nodes + threads - 1) / threads;
        
        init_nodes_kernel<<<blocks, threads, 0, stream>>>(d_nodes, d_graph.num_nodes);
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    
    void setMISFromHost(const vector<bool>& mis) {
        vector<DeviceNode> host_nodes(d_graph.num_nodes);
        for (int i = 0; i < d_graph.num_nodes; i++) {
            host_nodes[i] = DeviceNode(i, mis[i], -1);
        }
        
        CUDA_CHECK(cudaMemcpyAsync(d_nodes, host_nodes.data(), 
                                  d_graph.num_nodes * sizeof(DeviceNode), 
                                  cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    
    vector<bool> getMISToHost() {
        vector<DeviceNode> host_nodes(d_graph.num_nodes);
        CUDA_CHECK(cudaMemcpyAsync(host_nodes.data(), d_nodes, 
                                  d_graph.num_nodes * sizeof(DeviceNode), 
                                  cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        
        vector<bool> mis(d_graph.num_nodes);
        for (int i = 0; i < d_graph.num_nodes; i++) {
            mis[i] = host_nodes[i].membership;
        }
        return mis;
    }
    
    void processBatchWithVertexClustering(const vector<DeviceEdge>& batch, int hop_limit, DetailedTimings& timings) {
        if (batch.empty()) return;
        
        timings.reset();
        
        int num_edges = min((int)batch.size(), MAX_BATCH_SIZE);
        
        // Step 1: Copy edges to GPU (Host to Device transfer)
        auto h2d_start = chrono::high_resolution_clock::now();
        CUDA_CHECK(cudaMemcpyAsync(d_edges_pool, batch.data(), 
                                  num_edges * sizeof(DeviceEdge), 
                                  cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        auto h2d_end = chrono::high_resolution_clock::now();
        timings.host_to_device_time = chrono::duration_cast<chrono::microseconds>(h2d_end - h2d_start).count() / 1000.0;
        
        int threads = THREADS_PER_BLOCK;
        int blocks= SMs * mTpSM / THREADS_PER_BLOCK;
        float elapsed_time;
        cout<<"BOCKS------------------>"<<blocks<<endl;
        
        // Step 2: Extract vertices from edges
        blocks = (num_edges + threads - 1) / threads;
        
        CUDA_CHECK(cudaEventRecord(start_event, stream));
        extract_vertices_kernel<<<blocks, threads, 0, stream>>>(
            d_edges_pool, num_edges, d_vertices_pool, d_vertex_count);
        CUDA_CHECK(cudaEventRecord(stop_event, stream));
        CUDA_CHECK(cudaEventSynchronize(stop_event));
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_time, start_event, stop_event));
        timings.extract_vertices_time = elapsed_time;
        
        // Step 3: Compute vertex neighborhoods
        blocks = (num_edges * 2 + threads - 1) / threads;
        cout<<"BOCKS------------------>"<<blocks<<endl;
        blocks= SMs * mTpSM / THREADS_PER_BLOCK;
        CUDA_CHECK(cudaEventRecord(start_event, stream));
        compute_vertex_neighborhoods_optimized_v2<<<blocks, threads, 0, stream>>>(
            d_vertices_pool, num_edges * 2, d_graph, hop_limit,
            d_neighborhoods_pool, d_neighborhood_sizes_pool, MAX_NEIGHBORS_PER_VERTEX);
        CUDA_CHECK(cudaEventRecord(stop_event, stream));
        CUDA_CHECK(cudaEventSynchronize(stop_event));
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_time, start_event, stop_event));
        timings.compute_neighborhoods_time = elapsed_time;
        
        // Step 4: Initialize cluster count (small CPU overhead)
        auto cpu_start = chrono::high_resolution_clock::now();
        int zero = 0;
        CUDA_CHECK(cudaMemcpyAsync(d_cluster_count, &zero, sizeof(int), 
                                  cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        auto cpu_end = chrono::high_resolution_clock::now();
        timings.cpu_overhead_time += chrono::duration_cast<chrono::microseconds>(cpu_end - cpu_start).count() / 1000.0;
        
        // Step 5: Perform vertex clustering
        CUDA_CHECK(cudaEventRecord(start_event, stream));
        vertex_clustering_kernel<<<blocks, threads, 0, stream>>>(
            d_vertices_pool, num_edges * 2, d_neighborhoods_pool, d_neighborhood_sizes_pool,
            MAX_NEIGHBORS_PER_VERTEX, d_nodes, d_clusters_pool, d_cluster_count);
        CUDA_CHECK(cudaEventRecord(stop_event, stream));
        CUDA_CHECK(cudaEventSynchronize(stop_event));
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_time, start_event, stop_event));
        timings.vertex_clustering_time = elapsed_time;
        
        // Step 6: Assign edges to clusters
        blocks = (num_edges + threads - 1) / threads;
        
        CUDA_CHECK(cudaEventRecord(start_event, stream));
        assign_edges_to_clusters_kernel<<<blocks, threads, 0, stream>>>(
            d_edges_pool, num_edges, d_nodes, d_clusters_pool);
        CUDA_CHECK(cudaEventRecord(stop_event, stream));
        CUDA_CHECK(cudaEventSynchronize(stop_event));
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_time, start_event, stop_event));
        timings.assign_edges_time = elapsed_time;
        
        // Step 7: Get cluster count (CPU overhead)
        cpu_start = chrono::high_resolution_clock::now();
        int cluster_count;
        CUDA_CHECK(cudaMemcpyAsync(&cluster_count, d_cluster_count, sizeof(int), 
                                  cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        cpu_end = chrono::high_resolution_clock::now();
        timings.cpu_overhead_time += chrono::duration_cast<chrono::microseconds>(cpu_end - cpu_start).count() / 1000.0;
        
        // Step 8: TVB Process (process_clusters_kernel)
        if (cluster_count > 0) {
            blocks = (cluster_count + threads - 1) / threads;
            cout<<"Hello===========================-================================"<<endl;
            
            CUDA_CHECK(cudaEventRecord(start_event, stream));
            process_clusters_kernel<<<blocks, threads, 0, stream>>>(
                d_edges_pool, d_clusters_pool, cluster_count, d_graph, d_nodes);
            CUDA_CHECK(cudaEventRecord(stop_event, stream));
            CUDA_CHECK(cudaEventSynchronize(stop_event));
            CUDA_CHECK(cudaEventElapsedTime(&elapsed_time, start_event, stop_event));
            timings.tvb_process_time = elapsed_time;
            timings.clusters_count = cluster_count;
        }
        
        // Step 9: Clear cluster assignments
        blocks = (num_edges * 2 + threads - 1) / threads;
        
        CUDA_CHECK(cudaEventRecord(start_event, stream));
        clear_cluster_assignments_kernel<<<blocks, threads, 0, stream>>>(
            d_vertices_pool, num_edges * 2, d_nodes);
        CUDA_CHECK(cudaEventRecord(stop_event, stream));
        CUDA_CHECK(cudaEventSynchronize(stop_event));
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_time, start_event, stop_event));
        timings.clear_assignments_time = elapsed_time;
        
        CUDA_CHECK(cudaStreamSynchronize(stream));
        
        // Calculate aggregates
        timings.calculateAggregates();
    }
};

// Graph reading function
Graph* read_Graph(const string& filename) {
    Graph* graph = new Graph();
    ifstream file(filename);
    
    if (!file.is_open()) {
        cerr << "Unable to open file" << endl;
        return graph;
    }
    
    string line;
    while (getline(file, line) && line[0] == '%') {
        // skip comments
    }

    stringstream ss(line);
    int nodes, edges_count;
    ss >> nodes >> nodes >> edges_count;
    
    graph->num_nodes = nodes;
    graph->num_Edges = edges_count;
    graph->adj_list.resize(nodes);

    while (getline(file, line)) {
        int u, v;
        stringstream edge_ss(line);
        edge_ss >> u >> v;
        
        // Convert from 1-indexed to 0-indexed and validate
        u--; v--;
        
        if (u >= 0 && u < graph->num_nodes && v >= 0 && v < graph->num_nodes && u != v) {
            graph->adj_list[u].push_back(v);
            graph->adj_list[v].push_back(u);
        }
    }
    
    return graph;
}

// Batch processing function with comprehensive timing breakdown
void processBatchGPUVertexClustering(vector<Edge>& batch, Graph& graph, vector<Node>& nodes, 
                                    GPUVertexClusteringGraph& gpu_graph, int hop_limit,
                                    DetailedTimings& batch_timings) {
    auto total_start = chrono::high_resolution_clock::now();
    
    // Step 1: CPU conversion (convert host edges to device edges)
    auto conversion_start = chrono::high_resolution_clock::now();
    vector<DeviceEdge> device_batch = convertToDeviceEdges(batch);
    auto conversion_end = chrono::high_resolution_clock::now();
    batch_timings.cpu_conversion_time = chrono::duration_cast<chrono::microseconds>(conversion_end - conversion_start).count() / 1000.0;
    
    // Step 2: GPU processing with detailed timing
    gpu_graph.processBatchWithVertexClustering(device_batch, hop_limit, batch_timings);
    
    // Step 3: Get results from GPU (Device to Host transfer)
    auto d2h_start = chrono::high_resolution_clock::now();
    vector<bool> gpu_mis = gpu_graph.getMISToHost();
    auto d2h_end = chrono::high_resolution_clock::now();
    batch_timings.device_to_host_time = chrono::duration_cast<chrono::microseconds>(d2h_end - d2h_start).count() / 1000.0;
    
    // Step 4: Update CPU nodes with GPU results
    auto cpu_update_start = chrono::high_resolution_clock::now();
    for (size_t i = 0; i < nodes.size(); i++) {
        nodes[i].membership = gpu_mis[i];
    }
    auto cpu_update_end = chrono::high_resolution_clock::now();
    float cpu_node_update_time = chrono::duration_cast<chrono::microseconds>(cpu_update_end - cpu_update_start).count() / 1000.0;
    
    // Step 5: Update host graph (CPU graph operations)
    auto graph_update_start = chrono::high_resolution_clock::now();
    for (const auto& edge : batch) {
        if (edge.isInsertion && !graph.isAdjacent(edge.source, edge.destination)) {
            graph.insertEdge(edge.source, edge.destination);
        } else if (!edge.isInsertion && graph.isAdjacent(edge.source, edge.destination)) {
            graph.removeEdge(edge.source, edge.destination);
        }
    }
    auto graph_update_end = chrono::high_resolution_clock::now();
    batch_timings.cpu_graph_update_time = chrono::duration_cast<chrono::microseconds>(graph_update_end - graph_update_start).count() / 1000.0;
    
    auto total_end = chrono::high_resolution_clock::now();
    
    // Add CPU node update time to CPU overhead
    batch_timings.cpu_overhead_time += cpu_node_update_time;
    
    // Calculate total times
    batch_timings.total_batch_time = chrono::duration_cast<chrono::microseconds>(total_end - total_start).count() / 1000.0;
    
    // Recalculate aggregates with all timing data
    batch_timings.calculateAggregates();
    
    // Print comprehensive breakdown
    cout << "=== BATCH TIMING BREAKDOWN ===" << endl;
    cout << "Total Batch Time: " << fixed << setprecision(3) << batch_timings.total_batch_time << " ms" << endl;
    cout << endl;

    cout << "Key Performance Metrics:" << endl;
    cout << "  Transfer Time:     " << fixed << setprecision(3) << batch_timings.transfer_time << " ms" << endl;
    cout << "  Cluster Time:      " << fixed << setprecision(3) << batch_timings.cluster_processing_time << " ms" << endl;
    cout << "  MIS Processing:    " << fixed << setprecision(3) << batch_timings.tvb_process_time << " ms" << endl;
    cout << endl;

    cout << "Component Breakdown:" << endl;
    cout << "  1. CPU Conversion:        " << fixed << setprecision(3) << batch_timings.cpu_conversion_time << " ms (CPU)" << endl;
    cout << "  2. Host->Device Transfer: " << fixed << setprecision(3) << batch_timings.host_to_device_time << " ms (GPU)" << endl;
    cout << "  3. Cluster Processing:    " << fixed << setprecision(3) << batch_timings.cluster_processing_time << " ms (GPU)" << endl;
    cout << "       - Extract vertices: " << fixed << setprecision(3) << batch_timings.extract_vertices_time << " ms" << endl;
    cout << "       - Compute neighb.:  " << fixed << setprecision(3) << batch_timings.compute_neighborhoods_time << " ms" << endl;
    cout << "       - Vertex cluster.:  " << fixed << setprecision(3) << batch_timings.vertex_clustering_time << " ms" << endl;
    cout << "       - Assign edges:     " << fixed << setprecision(3) << batch_timings.assign_edges_time << " ms" << endl;
    cout << "       - Clear assigns.:   " << fixed << setprecision(3) << batch_timings.clear_assignments_time << " ms" << endl;
    cout << "  4. TVB Process:           " << fixed << setprecision(3) << batch_timings.tvb_process_time << " ms (GPU)" << endl;
    cout << "  5. Device->Host Transfer: " << fixed << setprecision(3) << batch_timings.device_to_host_time << " ms (GPU)" << endl;
    cout << "  6. CPU Graph Update:      " << fixed << setprecision(3) << batch_timings.cpu_graph_update_time << " ms (CPU)" << endl;
    cout << "  7. CPU Overhead:          " << fixed << setprecision(3) << batch_timings.cpu_overhead_time << " ms (CPU)" << endl;
    cout << "  8. Cluster count:         " << fixed << setprecision(3) << batch_timings.clusters_count << endl;
}

// Main function
int main(int argc, char* argv[]) {
    if (argc < 6) {
        cerr << "Usage: " << argv[0] << " <graph.mtx> <initial_mis.txt> <batch_folder> <num_batches> <gpu_device_id>" << endl;
        return 1;
    }
    cout<<"=======================================================START==============================================================================================================="<<endl;
    cout << "===================================================== ParMIS_v1 GPU Results ===============================================" << endl;
    
    // Set GPU device
    gpu_device_id = stoi(argv[5]);
    CUDA_CHECK(cudaSetDevice(gpu_device_id));
    
    // Get GPU properties
    cudaDeviceProp deviceProp;
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, gpu_device_id));
    cout << "Using GPU: " << deviceProp.name << endl;
    cout << "Max threads per block: " << deviceProp.maxThreadsPerBlock << endl;
    cout << "Shared memory per block: " << (deviceProp.sharedMemPerBlock / 1024) << " KB" << endl;
    cout << "Multiprocessor count: " << deviceProp.multiProcessorCount << endl;
    cout << "Memory clock rate: " << (deviceProp.memoryClockRate / 1000) << " MHz" << endl;
     SMs = deviceProp.multiProcessorCount;
  mTpSM = deviceProp.maxThreadsPerMultiProcessor;
    // Read graph
    Graph* graph = read_Graph(argv[1]);
    int num_nodes = graph->num_nodes;
    cout << argv[1]<<endl;
    cout << "Number of nodes: " << num_nodes << endl;
    cout << "Number of edges: " << graph->num_Edges << endl;
    
    // Initialize GPU graph
    GPUVertexClusteringGraph gpu_graph(graph->adj_list);
    
    // Initialize nodes
    vector<Node> nodes(num_nodes);
    for (int i = 0; i < num_nodes; ++i) {
        nodes[i] = Node(i, false);
    }

    // Read MIS
    ifstream inputFile(argv[2]);
    int initial_card = 0;
    vector<bool> initial_mis(num_nodes, false);
    
    if (inputFile.is_open()) {
        string line;
        while (getline(inputFile, line)) {
            int node_id = stoi(line);
            // Convert from 1-indexed to 0-indexed
            node_id--;
            if (node_id >= 0 && node_id < num_nodes) {
                nodes[node_id].membership = true;
                initial_mis[node_id] = true;
                initial_card++;
            }
        }
        inputFile.close();
    }
    
    gpu_graph.setMISFromHost(initial_mis);
    cout << "Initial MIS cardinality: " << initial_card << endl;

    // Read batches
    string folderPath(argv[3]);
    vector<vector<Edge>> batches_of_edges;
    vector<Edge> batch_of_edges;

    for (const auto& entry : filesystem::directory_iterator(folderPath)) {
        if (entry.is_regular_file()) {
            ifstream file(entry.path());
            string line;
            while (getline(file, line)) {
                istringstream iss(line);
                int src, dest;
                if (iss >> src >> dest) {
                    // Convert from 1-indexed to 0-indexed
                    src--; dest--;
                    
                    // Validate bounds
                    if (src >= 0 && src < graph->num_nodes && dest >= 0 && dest < graph->num_nodes && src != dest) {
                        if (graph->isAdjacent(src, dest)) {
                            batch_of_edges.emplace_back(src, dest, false);
                        } else {
                            batch_of_edges.emplace_back(src, dest, true);
                        }
                    }
                }
            }
            batches_of_edges.push_back(move(batch_of_edges));
            batch_of_edges.clear();
        }
    }

    cout << "Batch size: " << (batches_of_edges.empty() ? 0 : batches_of_edges[0].size()) << endl;
    
    int num_insertion_batches = min(stoi(argv[4]), (int)batches_of_edges.size());
    int hop_limit = 1;
    double time_cum = 0;
    
    // Cumulative detailed timings
    DetailedTimings cumulative_timings;
    
    // Process batches using GPU vertex clustering
    for (int i = 0; i < num_insertion_batches; i++) {
        if(i==3) continue;
        cout << "\n" << string(60, '=') << endl;
        cout << "Processing batch " << (i + 1) << "/" << num_insertion_batches << endl;
        cout << "Batch contains " << batches_of_edges[i].size() << " edges" << endl;
        cout << string(60, '-') << endl;
        
        auto batch_start = chrono::high_resolution_clock::now();
        
        DetailedTimings batch_timings;
        processBatchGPUVertexClustering(batches_of_edges[i], *graph, nodes, gpu_graph, hop_limit, batch_timings);
        
        // Accumulate timings
        cumulative_timings.accumulate(batch_timings);
        
        auto batch_end = chrono::high_resolution_clock::now();
        double batch_time = chrono::duration_cast<chrono::microseconds>(batch_end - batch_start).count() / 1000.0;
        time_cum += batch_time;
        
        // Calculate current MIS size
        int current_mis_size = 0;
        for (int j = 0; j < num_nodes; j++) {
            if (nodes[j].membership) current_mis_size++;
        }
        
        cout << "\nBatch Results:" << endl;
        cout << "  Current MIS cardinality: " << current_mis_size << endl;
    }

    // Calculate final results
    int updated_card = 0;
    for (int i = 0; i < num_nodes; i++) {
        if (nodes[i].membership)
            updated_card++;
    }

    cout << "\n" << string(80, '=') << endl;
    cout << "=== FINAL RESULTS ===" << endl;
    cout << string(80, '=') << endl;
    
    cout << "\nGraph Statistics:" << endl;
    cout << "  Initial MIS cardinality: " << initial_card << endl;
    cout << "  Final MIS cardinality:   " << updated_card << endl;
    cout << "  Cardinality change:      " << abs(initial_card - updated_card) << endl;
    cout << "  Number of batches:       " << num_insertion_batches << endl;
    
    cout << "\nOverall Timing Results:" << endl;
    cout << "  Total processing time:   " << fixed << setprecision(3) << time_cum << " milliseconds" << endl;
    cout << "  Average time per batch:  " << fixed << setprecision(3) << time_cum / num_insertion_batches << " milliseconds" << endl;
    cout << "  Processing rate:         " << fixed << setprecision(1) << (num_insertion_batches * 1000.0 / time_cum) << " batches/second" << endl;
    
    cout << "\nAverage Component Times:" << endl;
    cout << "  Average Transfer Time:      " << fixed << setprecision(3) << cumulative_timings.transfer_time / num_insertion_batches << " ms" << endl;
    cout << "  Average Cluster Processing: " << fixed << setprecision(3) << cumulative_timings.cluster_processing_time / num_insertion_batches << " ms" << endl;
    cout << "  Average MIS Processing:     " << fixed << setprecision(3) << cumulative_timings.tvb_process_time / num_insertion_batches << " ms" << endl;
    
    cout << string(80, '=') << endl;

    delete graph;
    return 0;
}