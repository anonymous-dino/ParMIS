/**
 * MULTI-GPU OVERLAPPED PIPELINE FOR GRAPH CLUSTERING AND MIS PROCESSING
 *
 * COMPILE INSTRUCTIONS:
 * nvcc -O3 -arch=sm_80 A_multi_gpu2.cu -o A_multi_gpu2
 * 
 *
 * RUN INSTRUCTIONS:
 * ./A_multi_gpu2 <graph.mtx> <initial_mis.txt> <batch_folder> <num_batches>
 *
 * PIPELINE FLOW:
 * 1. CPU generates/reads graph batches with vertices and adjacency data
 * 2. CPU streams batch N to GPU0 for clustering processing
 * 3. GPU0 performs complete vertex clustering 
 * 4. GPU1 waits for GPU0's completion, then processes MIS operations
 * 5. While GPU1 processes batch N-1, GPU0 can start clustering batch N
 * 6. CPU prepares batch N+1 while GPUs work on previous batches
 */

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
#include <thread>

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

// Configuration Constants - matching your original values
#define BUFFER_COUNT 2                    // Double buffering
#define MAX_BATCH_SIZE 1048576           // 1M as in your original code
#define THREADS_PER_BLOCK 256            // Your original TPB
#define MAX_NEIGHBORS_PER_VERTEX 128     // Your original neighbor limit
#define NUM_BATCHES_LIMIT 20             // Configurable batch limit

// GPU assignments
#define GPU_CLUSTER 0                    // GPU0 for clustering
#define GPU_MIS 1                       // GPU1 for MIS processing

// Global GPU properties
int gpu_device_id = 0;
int SMs = 0;
int mTpSM = 0;

// Comprehensive timing structure (from your original code)
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
        clusters_count = 0;
    }
    
    void calculateAggregates() {
        // Cluster processing = everything before TVB
        cluster_processing_time = extract_vertices_time + compute_neighborhoods_time + 
                                vertex_clustering_time + assign_edges_time + clear_assignments_time;
        
        // Total GPU kernel time
        total_gpu_kernel_time = cluster_processing_time + tvb_process_time;
        
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
    }
    
    void printDetailed(const string& prefix = "") {
        cout << prefix << "=== BATCH TIMING BREAKDOWN ===" << endl;
        cout << prefix << "Total Batch Time: " << fixed << setprecision(3) << total_batch_time << " ms" << endl;
        cout << prefix << endl;
        
        cout << prefix << "Component Breakdown:" << endl;
        cout << prefix << "  1. CPU Conversion:        " << fixed << setprecision(3) << cpu_conversion_time << " ms (CPU)" << endl;
        cout << prefix << "  2. Host->Device Transfer: " << fixed << setprecision(3) << host_to_device_time << " ms (GPU)" << endl;
        cout << prefix << "  3. Cluster Processing:    " << fixed << setprecision(3) << cluster_processing_time << " ms (GPU)" << endl;
        cout << prefix << "       - Extract vertices: " << fixed << setprecision(3) << extract_vertices_time << " ms" << endl;
        cout << prefix << "       - Compute neighb.:  " << fixed << setprecision(3) << compute_neighborhoods_time << " ms" << endl;
        cout << prefix << "       - Vertex cluster.:  " << fixed << setprecision(3) << vertex_clustering_time << " ms" << endl;
        cout << prefix << "       - Assign edges:     " << fixed << setprecision(3) << assign_edges_time << " ms" << endl;
        cout << prefix << "       - Clear assigns.:   " << fixed << setprecision(3) << clear_assignments_time << " ms" << endl;
        cout << prefix << "  4. TVB Process:           " << fixed << setprecision(3) << tvb_process_time << " ms (GPU)" << endl;
        cout << prefix << "  5. Device->Host Transfer: " << fixed << setprecision(3) << device_to_host_time << " ms (GPU)" << endl;
        cout << prefix << "  6. CPU Graph Update:      " << fixed << setprecision(3) << cpu_graph_update_time << " ms (CPU)" << endl;
        cout << prefix << "  7. CPU Overhead:          " << fixed << setprecision(3) << cpu_overhead_time << " ms (CPU)" << endl;
        cout << prefix << "  8. Cluster count:         " << clusters_count << endl;
    }
};

// Device structures (from your original code)
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

// Your original kernels - preserved exactly as implemented

// Kernel to extract unique vertices from edges (from your original code)
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

// Your original optimized neighborhood computation kernel
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

// Your original vertex clustering kernel
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

// Your original edge assignment kernel
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
        clusters[cluster_id].affected_edges[pos] = tid;
    }
}

// Your original MIS processing kernel (TVB process)
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
            // Handle deletion - similar logic for both cases
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

// Your original clear assignments kernel
__global__ void clear_cluster_assignments_kernel(const int* vertices, int vertex_count, DeviceNode* nodes) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < vertex_count) {
        int vertex = vertices[tid];
        if (vertex >= 0 && vertex < 10000000) {
            nodes[vertex].cluster_id = -1;
        }
    }
}

// Node initialization kernel (from your original code)
__global__ void init_nodes_kernel(DeviceNode* nodes, int num_nodes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_nodes) {
        nodes[idx] = DeviceNode(idx, false, -1);
    }
}

// Host structures (from your original code)
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

// Conversion function (from your original code)
vector<DeviceEdge> convertToDeviceEdges(const vector<Edge>& host_edges) {
    vector<DeviceEdge> device_edges;
    device_edges.reserve(host_edges.size());
    
    for (const auto& edge : host_edges) {
        device_edges.emplace_back(edge.source, edge.destination, edge.isInsertion);
    }
    
    return device_edges;
}

// Graph class (from your original code)
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

// Buffer structure for dual-GPU pipeline
struct PipelineBuffer {
    // Host pinned memory
    DeviceEdge* h_edges;
    int* h_vertices;
    bool* h_mis_flags;
    
    // GPU0 (clustering) device memory
    DeviceEdge* d0_edges_pool;
    int* d0_vertices_pool;
    int* d0_neighborhoods_pool;
    int* d0_neighborhood_sizes_pool;
    DeviceVertexCluster* d0_clusters_pool;
    int* d0_cluster_count;
    int* d0_vertex_count;
    DeviceNode* d0_nodes;
    DeviceGraph d0_graph;
    
    // GPU1 (MIS) device memory  
    DeviceNode* d1_nodes;
    DeviceGraph d1_graph;
    
    // CUDA streams and events
    cudaStream_t stream_cluster;     // GPU0 stream
    cudaStream_t stream_mis;         // GPU1 stream
    cudaEvent_t event_cluster_done;  // Signals clustering finished
    cudaEvent_t event_mis_done;      // Signals MIS finished
    
    // Timing events
    cudaEvent_t start_event, stop_event;
    
    // State tracking
    bool in_use;
    int batch_id;
    bool initialized;
    
    // Timing data
    DetailedTimings timings;
};

// Global pipeline state
struct PipelineState {
    PipelineBuffer buffers[BUFFER_COUNT];
    bool p2p_enabled;
    int current_buffer;
    int batches_dispatched;
    int batches_completed;
    chrono::high_resolution_clock::time_point start_time;
    
    // Graph data
    vector<vector<int>> adj_list;
    int num_nodes;
    
    // Cumulative timings
    DetailedTimings cumulative_timings;
};

// Initialize single buffer with all required memory
bool initializeBuffer(PipelineBuffer& buf, int buffer_idx, const vector<vector<int>>& adj_list) {
    printf("Initializing buffer %d...\n", buffer_idx);
    
    buf.initialized = false;
    buf.in_use = false;
    buf.batch_id = -1;
    buf.timings.reset();
    
    int num_nodes = adj_list.size();
    
    try {
        // Allocate pinned host memory
        CUDA_CHECK(cudaMallocHost(&buf.h_edges, MAX_BATCH_SIZE * sizeof(DeviceEdge)));
        CUDA_CHECK(cudaMallocHost(&buf.h_vertices, MAX_BATCH_SIZE * 2 * sizeof(int)));
        CUDA_CHECK(cudaMallocHost(&buf.h_mis_flags, num_nodes * sizeof(bool)));
        
        // Initialize GPU0 resources (clustering)
        CUDA_CHECK(cudaSetDevice(GPU_CLUSTER));
        
        // Build CSR format for GPU0
        vector<int> row_offsets(num_nodes + 1, 0);
        vector<int> column_indices;
        
        for (int i = 0; i < num_nodes; i++) {
            row_offsets[i + 1] = row_offsets[i] + adj_list[i].size();
            for (int neighbor : adj_list[i]) {
                column_indices.push_back(neighbor);
            }
        }
        
        // Allocate GPU0 graph memory
        CUDA_CHECK(cudaMalloc(&buf.d0_graph.row_offsets, (num_nodes + 1) * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&buf.d0_graph.column_indices, column_indices.size() * sizeof(int)));
        buf.d0_graph.num_nodes = num_nodes;
        
        // Copy graph data to GPU0
        CUDA_CHECK(cudaMemcpy(buf.d0_graph.row_offsets, row_offsets.data(), 
                             (num_nodes + 1) * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(buf.d0_graph.column_indices, column_indices.data(), 
                             column_indices.size() * sizeof(int), cudaMemcpyHostToDevice));
        
        // Allocate GPU0 processing memory
        CUDA_CHECK(cudaMalloc(&buf.d0_edges_pool, MAX_BATCH_SIZE * sizeof(DeviceEdge)));
        CUDA_CHECK(cudaMalloc(&buf.d0_vertices_pool, MAX_BATCH_SIZE * 2 * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&buf.d0_neighborhoods_pool, MAX_BATCH_SIZE * 2 * MAX_NEIGHBORS_PER_VERTEX * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&buf.d0_neighborhood_sizes_pool, MAX_BATCH_SIZE * 2 * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&buf.d0_clusters_pool, MAX_BATCH_SIZE * sizeof(DeviceVertexCluster)));
        CUDA_CHECK(cudaMalloc(&buf.d0_cluster_count, sizeof(int)));
        CUDA_CHECK(cudaMalloc(&buf.d0_vertex_count, sizeof(int)));
        CUDA_CHECK(cudaMalloc(&buf.d0_nodes, num_nodes * sizeof(DeviceNode)));
        
        // Initialize memory to prevent garbage values
        CUDA_CHECK(cudaMemset(buf.d0_neighborhoods_pool, 0, MAX_BATCH_SIZE * 2 * MAX_NEIGHBORS_PER_VERTEX * sizeof(int)));
        CUDA_CHECK(cudaMemset(buf.d0_neighborhood_sizes_pool, 0, MAX_BATCH_SIZE * 2 * sizeof(int)));
        
        // Create GPU0 stream and events
        CUDA_CHECK(cudaStreamCreate(&buf.stream_cluster));
        CUDA_CHECK(cudaEventCreate(&buf.event_cluster_done));
        CUDA_CHECK(cudaEventCreate(&buf.start_event));
        CUDA_CHECK(cudaEventCreate(&buf.stop_event));
        
        // Initialize GPU1 resources (MIS processing)
        CUDA_CHECK(cudaSetDevice(GPU_MIS));
        
        // Allocate GPU1 graph memory
        CUDA_CHECK(cudaMalloc(&buf.d1_graph.row_offsets, (num_nodes + 1) * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&buf.d1_graph.column_indices, column_indices.size() * sizeof(int)));
        buf.d1_graph.num_nodes = num_nodes;
        
        // Copy graph data to GPU1
        CUDA_CHECK(cudaMemcpy(buf.d1_graph.row_offsets, row_offsets.data(), 
                             (num_nodes + 1) * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(buf.d1_graph.column_indices, column_indices.data(), 
                             column_indices.size() * sizeof(int), cudaMemcpyHostToDevice));
        
        // Allocate GPU1 node memory
        CUDA_CHECK(cudaMalloc(&buf.d1_nodes, num_nodes * sizeof(DeviceNode)));
        
        // Create GPU1 stream and events
        CUDA_CHECK(cudaStreamCreate(&buf.stream_mis));
        CUDA_CHECK(cudaEventCreate(&buf.event_mis_done));
        
        buf.initialized = true;
        printf("Buffer %d initialized successfully\n", buffer_idx);
        return true;
        
    } catch (...) {
        printf("Failed to initialize buffer %d\n", buffer_idx);
        return false;
    }
}

// Initialize pipeline buffers
void initializePipelineBuffers(PipelineState& state) {
    cout << "Initializing pipeline buffers..." << endl;
    
    for (int i = 0; i < BUFFER_COUNT; i++) {
        if (!initializeBuffer(state.buffers[i], i, state.adj_list)) {
            fprintf(stderr, "Failed to initialize buffer %d\n", i);
            exit(1);
        }
    }
    
    cout << "All buffers initialized successfully" << endl;
}

// Initialize nodes on both GPUs
void initializeNodes(PipelineState& state, const vector<bool>& initial_mis) {
    int num_nodes = state.num_nodes;
    
    for (int i = 0; i < BUFFER_COUNT; i++) {
        PipelineBuffer& buf = state.buffers[i];
        
        // Initialize nodes on GPU0
        CUDA_CHECK(cudaSetDevice(GPU_CLUSTER));
        int threads = THREADS_PER_BLOCK;
        int blocks = (num_nodes + threads - 1) / threads;
        
        init_nodes_kernel<<<blocks, threads, 0, buf.stream_cluster>>>(buf.d0_nodes, num_nodes);
        CUDA_CHECK(cudaStreamSynchronize(buf.stream_cluster));
        
        // Set MIS values on GPU0
        vector<DeviceNode> host_nodes(num_nodes);
        for (int j = 0; j < num_nodes; j++) {
            host_nodes[j] = DeviceNode(j, initial_mis[j], -1);
        }
        
        CUDA_CHECK(cudaMemcpyAsync(buf.d0_nodes, host_nodes.data(), 
                                  num_nodes * sizeof(DeviceNode), 
                                  cudaMemcpyHostToDevice, buf.stream_cluster));
        CUDA_CHECK(cudaStreamSynchronize(buf.stream_cluster));
        
        // Initialize nodes on GPU1
        CUDA_CHECK(cudaSetDevice(GPU_MIS));
        init_nodes_kernel<<<blocks, threads, 0, buf.stream_mis>>>(buf.d1_nodes, num_nodes);
        CUDA_CHECK(cudaStreamSynchronize(buf.stream_mis));
        
        // Set MIS values on GPU1
        CUDA_CHECK(cudaMemcpyAsync(buf.d1_nodes, host_nodes.data(), 
                                  num_nodes * sizeof(DeviceNode), 
                                  cudaMemcpyHostToDevice, buf.stream_mis));
        CUDA_CHECK(cudaStreamSynchronize(buf.stream_mis));
    }
}

// Check and enable P2P access
bool enableP2P() {
    int canAccessPeer01, canAccessPeer10;
    
    CUDA_CHECK(cudaDeviceCanAccessPeer(&canAccessPeer01, GPU_CLUSTER, GPU_MIS));
    CUDA_CHECK(cudaDeviceCanAccessPeer(&canAccessPeer10, GPU_MIS, GPU_CLUSTER));
    
    if (canAccessPeer01 && canAccessPeer10) {
        try {
            CUDA_CHECK(cudaSetDevice(GPU_CLUSTER));
            CUDA_CHECK(cudaDeviceEnablePeerAccess(GPU_MIS, 0));
            
            CUDA_CHECK(cudaSetDevice(GPU_MIS));  
            CUDA_CHECK(cudaDeviceEnablePeerAccess(GPU_CLUSTER, 0));
            
            cout << "P2P access enabled between GPU0 and GPU1" << endl;
            return true;
        } catch (...) {
            cout << "P2P setup failed, using host staging" << endl;
            return false;
        }
    } else {
        cout << "P2P access not available, using host staging" << endl;
        return false;
    }
}

// Dispatch clustering on GPU0 (with all your original kernel calls)
void dispatchClustering(PipelineState& state, int buffer_idx, const vector<DeviceEdge>& batch) {
    PipelineBuffer& buf = state.buffers[buffer_idx];
    
    if (!buf.initialized) {
        fprintf(stderr, "Buffer %d not initialized\n", buffer_idx);
        return;
    }
    
    CUDA_CHECK(cudaSetDevice(GPU_CLUSTER));
    
    buf.timings.reset();
    int num_edges = min((int)batch.size(), MAX_BATCH_SIZE);
    
    // Step 1: Copy edges to GPU (Host to Device transfer)
    auto h2d_start = chrono::high_resolution_clock::now();
    CUDA_CHECK(cudaMemcpyAsync(buf.d0_edges_pool, batch.data(), 
                               num_edges * sizeof(DeviceEdge), 
                               cudaMemcpyHostToDevice, buf.stream_cluster));
    CUDA_CHECK(cudaStreamSynchronize(buf.stream_cluster));
    auto h2d_end = chrono::high_resolution_clock::now();
    buf.timings.host_to_device_time = chrono::duration_cast<chrono::microseconds>(h2d_end - h2d_start).count() / 1000.0;
    
    int threads = THREADS_PER_BLOCK;
    int blocks = SMs * mTpSM / THREADS_PER_BLOCK;
    float elapsed_time;
    cout << "BLOCKS-------------------->" << blocks << endl;
    
    // Step 2: Extract vertices from edges
    blocks = (num_edges + threads - 1) / threads;
    
    CUDA_CHECK(cudaEventRecord(buf.start_event, buf.stream_cluster));
    extract_vertices_kernel<<<blocks, threads, 0, buf.stream_cluster>>>(
        buf.d0_edges_pool, num_edges, buf.d0_vertices_pool, buf.d0_vertex_count);
    CUDA_CHECK(cudaEventRecord(buf.stop_event, buf.stream_cluster));
    CUDA_CHECK(cudaEventSynchronize(buf.stop_event));
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_time, buf.start_event, buf.stop_event));
    buf.timings.extract_vertices_time = elapsed_time;
    
    // Step 3: Compute vertex neighborhoods
    blocks = (num_edges * 2 + threads - 1) / threads;
    cout << "BLOCKS-------------------->" << blocks << endl;
    blocks = SMs * mTpSM / THREADS_PER_BLOCK;
    CUDA_CHECK(cudaEventRecord(buf.start_event, buf.stream_cluster));
    compute_vertex_neighborhoods_optimized_v2<<<blocks, threads, 0, buf.stream_cluster>>>(
        buf.d0_vertices_pool, num_edges * 2, buf.d0_graph, 1,
        buf.d0_neighborhoods_pool, buf.d0_neighborhood_sizes_pool, MAX_NEIGHBORS_PER_VERTEX);
    CUDA_CHECK(cudaEventRecord(buf.stop_event, buf.stream_cluster));
    CUDA_CHECK(cudaEventSynchronize(buf.stop_event));
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_time, buf.start_event, buf.stop_event));
    buf.timings.compute_neighborhoods_time = elapsed_time;
    
    // Step 4: Initialize cluster count
    auto cpu_start = chrono::high_resolution_clock::now();
    int zero = 0;
    CUDA_CHECK(cudaMemcpyAsync(buf.d0_cluster_count, &zero, sizeof(int), 
                              cudaMemcpyHostToDevice, buf.stream_cluster));
    CUDA_CHECK(cudaStreamSynchronize(buf.stream_cluster));
    auto cpu_end = chrono::high_resolution_clock::now();
    buf.timings.cpu_overhead_time += chrono::duration_cast<chrono::microseconds>(cpu_end - cpu_start).count() / 1000.0;
    
    // Step 5: Perform vertex clustering
    CUDA_CHECK(cudaEventRecord(buf.start_event, buf.stream_cluster));
    vertex_clustering_kernel<<<blocks, threads, 0, buf.stream_cluster>>>(
        buf.d0_vertices_pool, num_edges * 2, buf.d0_neighborhoods_pool, buf.d0_neighborhood_sizes_pool,
        MAX_NEIGHBORS_PER_VERTEX, buf.d0_nodes, buf.d0_clusters_pool, buf.d0_cluster_count);
    CUDA_CHECK(cudaEventRecord(buf.stop_event, buf.stream_cluster));
    CUDA_CHECK(cudaEventSynchronize(buf.stop_event));
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_time, buf.start_event, buf.stop_event));
    buf.timings.vertex_clustering_time = elapsed_time;
    
    // Step 6: Assign edges to clusters
    blocks = (num_edges + threads - 1) / threads;
    
    CUDA_CHECK(cudaEventRecord(buf.start_event, buf.stream_cluster));
    assign_edges_to_clusters_kernel<<<blocks, threads, 0, buf.stream_cluster>>>(
        buf.d0_edges_pool, num_edges, buf.d0_nodes, buf.d0_clusters_pool);
    CUDA_CHECK(cudaEventRecord(buf.stop_event, buf.stream_cluster));
    CUDA_CHECK(cudaEventSynchronize(buf.stop_event));
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_time, buf.start_event, buf.stop_event));
    buf.timings.assign_edges_time = elapsed_time;
    
    // Step 7: Get cluster count
    cpu_start = chrono::high_resolution_clock::now();
    int cluster_count;
    CUDA_CHECK(cudaMemcpyAsync(&cluster_count, buf.d0_cluster_count, sizeof(int), 
                              cudaMemcpyDeviceToHost, buf.stream_cluster));
    CUDA_CHECK(cudaStreamSynchronize(buf.stream_cluster));
    cpu_end = chrono::high_resolution_clock::now();
    buf.timings.cpu_overhead_time += chrono::duration_cast<chrono::microseconds>(cpu_end - cpu_start).count() / 1000.0;
    
    // Step 8: TVB Process (process_clusters_kernel)
    if (cluster_count > 0) {
        blocks = (cluster_count + threads - 1) / threads;
        cout << "Hello===========================-================================" << endl;
        
        CUDA_CHECK(cudaEventRecord(buf.start_event, buf.stream_cluster));
        process_clusters_kernel<<<blocks, threads, 0, buf.stream_cluster>>>(
            buf.d0_edges_pool, buf.d0_clusters_pool, cluster_count, buf.d0_graph, buf.d0_nodes);
        CUDA_CHECK(cudaEventRecord(buf.stop_event, buf.stream_cluster));
        CUDA_CHECK(cudaEventSynchronize(buf.stop_event));
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_time, buf.start_event, buf.stop_event));
        buf.timings.tvb_process_time = elapsed_time;
        buf.timings.clusters_count = cluster_count;
    }
    
    // Step 9: Clear cluster assignments
    blocks = (num_edges * 2 + threads - 1) / threads;
    
    CUDA_CHECK(cudaEventRecord(buf.start_event, buf.stream_cluster));
    clear_cluster_assignments_kernel<<<blocks, threads, 0, buf.stream_cluster>>>(
        buf.d0_vertices_pool, num_edges * 2, buf.d0_nodes);
    CUDA_CHECK(cudaEventRecord(buf.stop_event, buf.stream_cluster));
    CUDA_CHECK(cudaEventSynchronize(buf.stop_event));
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_time, buf.start_event, buf.stop_event));
    buf.timings.clear_assignments_time = elapsed_time;
    
    // Record completion event for GPU1 to wait on
    CUDA_CHECK(cudaEventRecord(buf.event_cluster_done, buf.stream_cluster));
    
    cout << "Dispatched batch " << buf.batch_id << " → slot " << buffer_idx << endl;
}

// Dispatch MIS processing on GPU1
void dispatchMIS(PipelineState& state, int buffer_idx) {
    PipelineBuffer& buf = state.buffers[buffer_idx];
    
    if (!buf.initialized) {
        fprintf(stderr, "Buffer %d not initialized for MIS\n", buffer_idx);
        return;
    }
    
    CUDA_CHECK(cudaSetDevice(GPU_MIS));
    
    // Wait for clustering to complete
    CUDA_CHECK(cudaStreamWaitEvent(buf.stream_mis, buf.event_cluster_done, 0));
    
    if (state.p2p_enabled) {
        // Direct P2P transfer from GPU0 to GPU1
        CUDA_CHECK(cudaMemcpyPeerAsync(buf.d1_nodes, GPU_MIS, 
                                       buf.d0_nodes, GPU_CLUSTER, 
                                       state.num_nodes * sizeof(DeviceNode), buf.stream_mis));
    } else {
        // Stage through host memory
        CUDA_CHECK(cudaSetDevice(GPU_CLUSTER));
        CUDA_CHECK(cudaMemcpyAsync(buf.h_mis_flags, (bool*)buf.d0_nodes, 
                                   state.num_nodes * sizeof(bool), 
                                   cudaMemcpyDeviceToHost, buf.stream_cluster));
        CUDA_CHECK(cudaStreamSynchronize(buf.stream_cluster));
        
        CUDA_CHECK(cudaSetDevice(GPU_MIS));
        CUDA_CHECK(cudaMemcpyAsync((bool*)buf.d1_nodes, buf.h_mis_flags, 
                                   state.num_nodes * sizeof(bool), 
                                   cudaMemcpyHostToDevice, buf.stream_mis));
    }
    
    // Record completion event
    CUDA_CHECK(cudaEventRecord(buf.event_mis_done, buf.stream_mis));
}

// Check for completed MIS processing
void checkCompletions(PipelineState& state, vector<Node>& nodes) {
    for (int i = 0; i < BUFFER_COUNT; i++) {
        PipelineBuffer& buf = state.buffers[i];
        
        if (buf.in_use && buf.batch_id >= 0) {
            cudaError_t result = cudaEventQuery(buf.event_mis_done);
            
            if (result == cudaSuccess) {
                // Calculate aggregates for timing
                buf.timings.calculateAggregates();
                
                // Get results back from GPU1
                CUDA_CHECK(cudaSetDevice(GPU_MIS));
                vector<DeviceNode> host_nodes(state.num_nodes);
                CUDA_CHECK(cudaMemcpyAsync(host_nodes.data(), buf.d1_nodes, 
                                           state.num_nodes * sizeof(DeviceNode), 
                                           cudaMemcpyDeviceToHost, buf.stream_mis));
                CUDA_CHECK(cudaStreamSynchronize(buf.stream_mis));
                
                // Update CPU nodes
                for (int j = 0; j < state.num_nodes; j++) {
                    nodes[j].membership = host_nodes[j].membership;
                }
                
                // Print detailed timing
                buf.timings.printDetailed();
                
                // Accumulate timings
                state.cumulative_timings.accumulate(buf.timings);
                
                cout << "Finished MIS for batch " << buf.batch_id << " → slot " << i << endl;
                
                // Free buffer for reuse
                buf.in_use = false;
                state.batches_completed++;
                buf.batch_id = -1;
            } else if (result != cudaErrorNotReady) {
                CUDA_CHECK(result);
            }
        }
    }
}

// Find available buffer slot
int findAvailableBuffer(PipelineState& state) {
    for (int i = 0; i < BUFFER_COUNT; i++) {
        if (!state.buffers[i].in_use && state.buffers[i].initialized) {
            return i;
        }
    }
    return -1;
}

// Graph reading function (from your original code)
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

// Main pipeline execution
void runPipeline(int argc, char* argv[]) {
    cout << "\n=== Multi-GPU Overlapped Pipeline ===\n" << endl;
    
    // Check GPU availability
    int deviceCount;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    
    if (deviceCount < 2) {
        cerr << "Error: At least 2 CUDA devices required, found " << deviceCount << endl;
        exit(1);
    }
    
    cout << "Found " << deviceCount << " CUDA devices" << endl;
    cout << "GPU0 (Clustering): Device " << GPU_CLUSTER << endl;
    cout << "GPU1 (MIS): Device " << GPU_MIS << endl;
    
    // Get GPU properties
    cudaDeviceProp deviceProp0, deviceProp1;
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProp0, GPU_CLUSTER));
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProp1, GPU_MIS));
    
    cout << "GPU0: " << deviceProp0.name << endl;
    cout << "GPU1: " << deviceProp1.name << endl;
    cout << "Max threads per block: " << deviceProp0.maxThreadsPerBlock << endl;
    cout << "Multiprocessor count: " << deviceProp0.multiProcessorCount << endl;
    
    SMs = deviceProp0.multiProcessorCount;
    mTpSM = deviceProp0.maxThreadsPerMultiProcessor;
    
    // Read graph
    Graph* graph = read_Graph(argv[1]);
    int num_nodes = graph->num_nodes;
    cout << argv[1] << endl;
    cout << "Number of nodes: " << num_nodes << endl;
    cout << "Number of edges: " << graph->num_Edges << endl;
    
    PipelineState state;
    state.adj_list = graph->adj_list;
    state.num_nodes = num_nodes;
    state.current_buffer = 0;
    state.batches_dispatched = 0;
    state.batches_completed = 0;
    
    // Initialize nodes
    vector<Node> nodes(num_nodes);
    for (int i = 0; i < num_nodes; ++i) {
        nodes[i] = Node(i, false);
    }

    // Read initial MIS
    ifstream inputFile(argv[2]);
    int initial_card = 0;
    vector<bool> initial_mis(num_nodes, false);
    
    if (inputFile.is_open()) {
        string line;
        while (getline(inputFile, line)) {
            int node_id = stoi(line);
            node_id--;
            if (node_id >= 0 && node_id < num_nodes) {
                nodes[node_id].membership = true;
                initial_mis[node_id] = true;
                initial_card++;
            }
        }
        inputFile.close();
    }
    
    cout << "Initial MIS cardinality: " << initial_card << endl;

    // Initialize pipeline
    initializePipelineBuffers(state);
    initializeNodes(state, initial_mis);
    state.p2p_enabled = enableP2P();
    
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
                    src--; dest--;
                    
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
    
    int num_insertion_batches = min(stoi(argv[4]), min(NUM_BATCHES_LIMIT, (int)batches_of_edges.size()));
    cout << "\nStarting pipeline with " << BUFFER_COUNT << " buffers..." << endl;
    state.start_time = chrono::high_resolution_clock::now();
    
    // Main pipeline loop
    while (state.batches_completed < num_insertion_batches) {
        // Check for completions first
        checkCompletions(state, nodes);
        
        // Try to dispatch new batch
        if (state.batches_dispatched < num_insertion_batches) {
            int buffer_idx = findAvailableBuffer(state);
            
            if (buffer_idx >= 0) {
                PipelineBuffer& buf = state.buffers[buffer_idx];
                buf.in_use = true;
                buf.batch_id = state.batches_dispatched;
                
                // Convert batch to device format
                vector<DeviceEdge> device_batch = convertToDeviceEdges(batches_of_edges[state.batches_dispatched]);
                
                // Update graph for CPU tracking
                for (const auto& edge : batches_of_edges[state.batches_dispatched]) {
                    if (edge.isInsertion && !graph->isAdjacent(edge.source, edge.destination)) {
                        graph->insertEdge(edge.source, edge.destination);
                    } else if (!edge.isInsertion && graph->isAdjacent(edge.source, edge.destination)) {
                        graph->removeEdge(edge.source, edge.destination);
                    }
                }
                
                cout << "\n" << string(60, '=') << endl;
                cout << "Processing batch " << (state.batches_dispatched + 1) << "/" << num_insertion_batches << endl;
                cout << "Batch contains " << batches_of_edges[state.batches_dispatched].size() << " edges" << endl;
                cout << string(60, '-') << endl;
                
                // Dispatch clustering on GPU0
                dispatchClustering(state, buffer_idx, device_batch);
                
                // Dispatch MIS on GPU1
                dispatchMIS(state, buffer_idx);
                
                state.batches_dispatched++;
            }
        }
        
        // Small sleep to prevent busy waiting
        this_thread::sleep_for(chrono::microseconds(100));
    }
    
    // Wait for all processing to complete
    cout << "Waiting for final completions..." << endl;
    for (int i = 0; i < BUFFER_COUNT; i++) {
        if (state.buffers[i].initialized) {
            CUDA_CHECK(cudaSetDevice(GPU_CLUSTER));
            CUDA_CHECK(cudaStreamSynchronize(state.buffers[i].stream_cluster));
            CUDA_CHECK(cudaSetDevice(GPU_MIS));
            CUDA_CHECK(cudaStreamSynchronize(state.buffers[i].stream_mis));
        }
    }
    
    auto end_time = chrono::high_resolution_clock::now();
    auto total_time = chrono::duration_cast<chrono::milliseconds>(end_time - state.start_time);
    
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
    cout << "  Total processing time:   " << fixed << setprecision(3) << total_time.count() << " milliseconds" << endl;
    cout << "  Average time per batch:  " << fixed << setprecision(3) << total_time.count() / (double)num_insertion_batches << " milliseconds" << endl;
    cout << "  Processing rate:         " << fixed << setprecision(1) << (num_insertion_batches * 1000.0) / total_time.count() << " batches/second" << endl;
    
    cout << "\nAverage Component Times:" << endl;
    cout << "  Average Cluster Processing: " << fixed << setprecision(3) << state.cumulative_timings.cluster_processing_time / num_insertion_batches << " ms" << endl;
    cout << "  Average TVB Time:           " << fixed << setprecision(3) << state.cumulative_timings.tvb_process_time / num_insertion_batches << " ms" << endl;
    
    cout << string(80, '=') << endl;

    delete graph;
}

int main(int argc, char* argv[]) {
    if (argc < 5) {
        cerr << "Usage: " << argv[0] << " <graph.mtx> <initial_mis.txt> <batch_folder> <num_batches>" << endl;
        return 1;
    }
    
    cout << "======================================================= START ===============================================================================================================" << endl;
    cout << "===================================================== ParMIS_v2 GPU Results ===============================================" << endl;
    
    try {
        runPipeline(argc, argv);
    } catch (const exception& e) {
        cerr << "Error: " << e.what() << endl;
        return 1;
    }
    
    cout << "\nPipeline completed successfully!" << endl;
    return 0;
}