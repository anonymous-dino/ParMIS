#include <omp.h>
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
#include <math.h>
#include <set>
#include <stdbool.h>
#include <stdio.h>
using namespace std;
vector<double> time_per_thread;
int n_threads = 64;
int total_Clusters = 0;
int total_Clusters_DU = 0;
// Struct representing a Node
struct Node
{
    int label;
    bool membership; // true if in MIS, false otherwise

    Node(int l = 0, bool m = false) : label(l), membership(m) {}
};

// // Class representing a Graph in CSR format
class Graph
{
public:
    int num_nodes;
    int num_Edges = 0;
    vector<vector<int>> adj_list; // Adjacency list
    vector<Node *> nodes;

    Graph(int n = 0) : num_nodes(n), adj_list(n), nodes(n)
    {
    }

    // Function to add an edge to the graph (undirected graph)
    void addEdge(int u, int v)
    {
        if (u >= 0 && u < num_nodes && v >= 0 && v < num_nodes)
        {
            adj_list[u].push_back(v);
            adj_list[v].push_back(u); // Since the graph is undirected
        }
    }

    // Function to remove an edge (u, v)
    void removeEdge(int u, int v)
    {
        if (u >= 0 && u < num_nodes && v >= 0 && v < num_nodes)
        {
            adj_list[u].erase(remove(adj_list[u].begin(), adj_list[u].end(), v), adj_list[u].end());
            adj_list[v].erase(remove(adj_list[v].begin(), adj_list[v].end(), u), adj_list[v].end());
        }
        num_Edges--;
    }

    // Function to get neighbors of a node
    vector<int> getNeighbors(int node) const
    {
        if (node < 0 || node >= num_nodes)
            return {};
        return adj_list[node];
    }

    // Function to insert an edge (u, v) - similar to addEdge
    void insertEdge(int u, int v)
    {
        addEdge(u, v);
        num_Edges++;
    }

    bool isAdjacent(int u, int v)
    {
        for (int i = 0; i < adj_list[u].size(); i++)
        {
            if (adj_list[u][i] == v)
            {
                return true;
            }
        }
        return false;
    }
};
// Struct representing an Edge
struct Edge
{
    int source;
    int destination;
    bool isInsertion; // true for insertion, false for deletion
    int clusterID;    // Assigned during clustering

    Edge(int s, int d, bool ins) : source(s), destination(d), isInsertion(ins), clusterID(-1) {}
};

// Class representing an Edge Cluster
class EdgeCluster
{
public:
    vector<Edge> edges;
    EdgeCluster() {edges.reserve(100);}
    // vector<int> affected_nodes;
};

// Class representing a Vertex Cluster (for vertex-based clustering)
class VertexCluster
{
public:
    vector<int> vertices;
    vector<Edge> affected_edges; // Edges that involve vertices in this cluster
    VertexCluster() {
        vertices.reserve(50);
        affected_edges.reserve(100);
    }
};

// Optimized BFS for affected area calculation
vector<int> getAffectedArea(const Edge &edge, const Graph &graph, int hop_limit)
{
    // Pre-allocate for better performance (estimate based on average degree)
    vector<int> affected_area;
    affected_area.reserve(20 * hop_limit); // Heuristic: assume average degree ~10

    // Use vector<bool> for efficient storage and quick lookups
    vector<bool> visited(graph.num_nodes, false);

    // Queue for BFS - use deque for better performance
    deque<pair<int, int>> q; // Node, hop count

    // Add source node
    q.push_back({edge.source, 0});
    visited[edge.source] = true;
    affected_area.push_back(edge.source);

    // Add destination node (avoid duplication for self-loops)
    if (edge.destination != edge.source)
    {
        q.push_back({edge.destination, 0});
        visited[edge.destination] = true;
        affected_area.push_back(edge.destination);
    }

    // BFS up to hop_limit
    while (!q.empty())
    {
        auto [node, hops] = q.front();
        q.pop_front();

        if (hops >= hop_limit)
            continue;

        const vector<int> &neighbors = graph.getNeighbors(node);

        // Process all neighbors
        for (int neighbor : neighbors)
        {
            if (!visited[neighbor])
            {
                visited[neighbor] = true;
                affected_area.push_back(neighbor);
                q.push_back({neighbor, hops + 1});
            }
        }
    }

    return affected_area;
}

// Vertex-based clustering utility functions
vector<int> extractVerticesFromEdges(const vector<Edge>& edges) {
    unordered_set<int> vertex_set;
    
    for (const Edge& edge : edges) {
        vertex_set.insert(edge.source);
        vertex_set.insert(edge.destination);
    }
    
    return vector<int>(vertex_set.begin(), vertex_set.end());
}

vector<Edge> getAffectedEdgesForVertexCluster(const vector<int>& cluster_vertices, 
                                             const vector<Edge>& all_edges) {
    unordered_set<int> vertex_set(cluster_vertices.begin(), cluster_vertices.end());
    vector<Edge> affected_edges;
    
    for (const Edge& edge : all_edges) {
        if (vertex_set.count(edge.source) || vertex_set.count(edge.destination)) {
            affected_edges.push_back(edge);
        }
    }
    
    return affected_edges;
}

vector<int> getVertexNeighborhood(int vertex, const Graph& graph, int hop_limit) {
    vector<int> neighborhood;
    neighborhood.reserve(20 * hop_limit);
    
    vector<bool> visited(graph.num_nodes, false);
    deque<pair<int, int>> q; // vertex, hop count
    
    q.push_back({vertex, 0});
    visited[vertex] = true;
    neighborhood.push_back(vertex);
    
    while (!q.empty()) {
        auto [current_vertex, hops] = q.front();
        q.pop_front();
        
        if (hops >= hop_limit) continue;
        
        const vector<int>& neighbors = graph.getNeighbors(current_vertex);
        for (int neighbor : neighbors) {
            if (!visited[neighbor]) {
                visited[neighbor] = true;
                neighborhood.push_back(neighbor);
                q.push_back({neighbor, hops + 1});
            }
        }
    }
    
    return neighborhood;
}

vector<EdgeCluster> globalClusters;
unordered_map<int, int> globalNodeToCluster;

// Function to cluster edges based on their affected areas
vector<EdgeCluster> clusters;

vector<EdgeCluster> clusterEdgesFromScratch(const vector<Edge> &newEdges, const Graph &graph, int hop_limit)
{
    vector<int> nodeToCluster;
    nodeToCluster.resize(graph.num_nodes+2, -1);

    // Create local clusters for this function call
    vector<EdgeCluster> local_clusters;
    local_clusters.reserve(newEdges.size());

    std::vector<std::vector<int>> N_h(newEdges.size());
    int nn_threads = n_threads;
    if(newEdges.size()/n_threads<5)
        nn_threads=1;
    #pragma omp parallel for num_threads(nn_threads) schedule(guided) shared(N_h)
    for (size_t i = 0; i < newEdges.size(); ++i) {
        N_h[i] = getAffectedArea(newEdges[i], graph, hop_limit);
    }
    int i=0;
    // Step 1: Initial clustering
    for (const Edge &edge : newEdges)
    {
        vector<int> affected_area = N_h[i];i++;
        // vector<bool> repeated_nodes(affected_area.size(), false);
        unordered_set<int> overlapping_clusters;
        // int i = 0;

        for (int node : affected_area)
        {
            // if (nodeToCluster.find(node) != nodeToCluster.end())
            // {
            //     overlapping_clusters.insert(nodeToCluster[node]);
            // }
            int clusterID = nodeToCluster[node];
            if (clusterID != -1)
            {
                overlapping_clusters.insert(clusterID);
            }
        }

        int clusterIndex;
        if (overlapping_clusters.empty())
        {
            // Create new cluster
            clusterIndex = local_clusters.size();
            local_clusters.emplace_back();
        }
        else
        {
            // Merge overlapping clusters
            clusterIndex = *overlapping_clusters.begin();
            int max_size_of_cluster = local_clusters[clusterIndex].edges.size();
            for(auto it = next(overlapping_clusters.begin()); it != overlapping_clusters.end(); ++it)
            {
                int otherIndex = *it;
                if (otherIndex != clusterIndex)
                {
                    if (local_clusters[otherIndex].edges.size() > max_size_of_cluster)
                    {
                        clusterIndex = otherIndex;
                        max_size_of_cluster = local_clusters[otherIndex].edges.size();
                    }
                }
            }
            for (auto it = next(overlapping_clusters.begin()); it != overlapping_clusters.end(); ++it)
            {
                int otherIndex = *it;
                // Use move semantics to reduce copying
                if (otherIndex != clusterIndex)
                {
                    // Merge clusters
                    local_clusters[clusterIndex].edges.insert(
                        local_clusters[clusterIndex].edges.end(),
                        make_move_iterator(local_clusters[otherIndex].edges.begin()),
                        make_move_iterator(local_clusters[otherIndex].edges.end()));
                    local_clusters[otherIndex] = EdgeCluster();
                }
            }
        }

        // Add edge to cluster
        local_clusters[clusterIndex].edges.push_back(edge);
        for (int node : affected_area)
        {
            nodeToCluster[node] = clusterIndex;
        }
    }

    return local_clusters;
}

// Vertex-based clustering implementation using simple node-to-cluster mapping
vector<VertexCluster> clusterVerticesOptimized(const vector<Edge>& newEdges, 
                                              const Graph& graph, 
                                              int hop_limit) {
    // Step 1: Extract unique vertices from edges
    vector<int> vertices = extractVerticesFromEdges(newEdges);
    
    // Step 2: Pre-compute neighborhoods for vertices in parallel
    unordered_map<int, vector<int>> vertex_neighborhoods;
    int nn_threads = (vertices.size() / n_threads < 5) ? 1 : n_threads;
    
    #pragma omp parallel for num_threads(nn_threads) schedule(dynamic)
    for (size_t i = 0; i < vertices.size(); ++i) {
        int vertex = vertices[i];
        vector<int> neighborhood = getVertexNeighborhood(vertex, graph, hop_limit);
        #pragma omp critical
        {
            vertex_neighborhoods[vertex] = move(neighborhood);
        }
    }
    
    // Step 3: Cluster vertices using simple mapping approach
    vector<VertexCluster> clusters;
    vector<int> nodeToCluster(graph.num_nodes, -1); // Simple array for node-to-cluster mapping
    int nextClusterId = 0;
    
    for (int vertex : vertices) {
        if (nodeToCluster[vertex] != -1) {
            continue; // Already processed
        }
        
        vector<int> neighborhood = vertex_neighborhoods[vertex];
        
        // Find overlapping clusters by checking neighbors
        unordered_set<int> overlapping_clusters;
        for (int neighbor : neighborhood) {
            if (neighbor < graph.num_nodes && nodeToCluster[neighbor] != -1) {
                overlapping_clusters.insert(nodeToCluster[neighbor]);
            }
        }
        
        int finalClusterId;
        
        if (overlapping_clusters.empty()) {
            // Create new cluster
            finalClusterId = nextClusterId++;
            clusters.resize(nextClusterId);
        } else if (overlapping_clusters.size() == 1) {
            // Single cluster overlap
            finalClusterId = *overlapping_clusters.begin();
        } else {
            // Multiple clusters need to be merged - find largest cluster
            finalClusterId = *overlapping_clusters.begin();
            size_t maxSize = clusters[finalClusterId].vertices.size();
            
            for (int clusterId : overlapping_clusters) {
                if (clusterId < clusters.size() && clusters[clusterId].vertices.size() > maxSize) {
                    maxSize = clusters[clusterId].vertices.size();
                    finalClusterId = clusterId;
                }
            }
            
            // Merge all other clusters into the target
            for (int clusterId : overlapping_clusters) {
                if (clusterId != finalClusterId && clusterId < clusters.size()) {
                    // Move vertices from source to target cluster
                    clusters[finalClusterId].vertices.insert(
                        clusters[finalClusterId].vertices.end(),
                        make_move_iterator(clusters[clusterId].vertices.begin()),
                        make_move_iterator(clusters[clusterId].vertices.end())
                    );
                    
                    // Update node cluster assignments
                    for (int v : clusters[clusterId].vertices) {
                        if (v < nodeToCluster.size()) {
                            nodeToCluster[v] = finalClusterId;
                        }
                    }

                    
                    // Clear the source cluster
                    clusters[clusterId] = VertexCluster();
                }
            }
        }
        
        // Add vertex to cluster
        clusters[finalClusterId].vertices.push_back(vertex);
        
        // Update cluster assignment for all vertices in neighborhood
        for (int neighbor : neighborhood) {
            if (neighbor < nodeToCluster.size()) {
                nodeToCluster[neighbor] = finalClusterId;
            }
        }
    }
    
    // Step 4: Assign affected edges to each cluster
    for (auto& cluster : clusters) {
        if (!cluster.vertices.empty()) {
            cluster.affected_edges = getAffectedEdgesForVertexCluster(cluster.vertices, newEdges);
        }
    }
    
    // Remove empty clusters
    clusters.erase(
        remove_if(clusters.begin(), clusters.end(),
                  [](const VertexCluster& c) { return c.vertices.empty(); }),
        clusters.end());
    
    return clusters;
}

// Function to handle insertion cases based on Lemmas 1-3
void handleInsertion(const Edge &edge, Graph &graph, vector<Node> &nodes)
{
    int u = edge.source;
    int v = edge.destination;

    bool u_in_mis = nodes[u].membership;
    bool v_in_mis = nodes[v].membership;
    // cout<<"Insertion-> "<< u <<" "<<v<<endl;
    if (u_in_mis && v_in_mis)
    {
        // Lemma 1: Both in MIS, need to remove one and potentially update neighbors
        // For simplicity, remove v from MIS
        v = min(u, v);
        nodes[v].membership = false;

        // Check neighbors of v to potentially add to MIS
        for (int neighbor : graph.getNeighbors(v))
        {
            if (!nodes[neighbor].membership)
            {
                // cout<<"in IF "<<endl;
                bool can_add = true;
                for (int nbr_of_nbr : graph.getNeighbors(neighbor))
                {
                    if (nodes[nbr_of_nbr].membership)
                    {
                        can_add = false;
                        break;
                    }
                }
                if (can_add)
                {
                    // cout<<"Can_add"<<neighbor<<endl;
                    nodes[neighbor].membership = true;
                }
            }
        }
    }
    else if (u_in_mis || v_in_mis)
    {
        // Lemma 2: At least one is in MIS, do nothing
        // No action needed
    }
    else
    {
        // Lemma 3: Neither is in MIS, do nothing
        // No action needed
    }

    // Finally, insert the edge into the graph
    // graph.insertEdge(u, v);
}

// Function to handle deletion cases based on Lemmas 4-6
void handleDeletion(const Edge &edge, Graph &graph, vector<Node> &nodes)
{
    int u = edge.source;
    int v = edge.destination;

    bool u_in_mis = nodes[u].membership;
    bool v_in_mis = nodes[v].membership;
    // cout << "Deletion-> "<< "u:" << u << " v:" << v << endl;

    if (u_in_mis || v_in_mis)
    {
        // Lemma 4: One of them is in MIS, try to add the other
        if (u_in_mis && !v_in_mis)
        {
            // Try to add v to MIS
            bool can_add = true;
            for (int neighbor : graph.getNeighbors(v))
            {
                if (nodes[neighbor].membership && neighbor != u)
                {
                    can_add = false;
                    break;
                }
            }
            if (can_add)
            {
                nodes[v].membership = true;
            }
        }
        else if (!u_in_mis && v_in_mis)
        {
            // Try to add u to MIS
            bool can_add = true;
            for (int neighbor : graph.getNeighbors(u))
            {
                if (nodes[neighbor].membership && neighbor != v)
                {
                    // cout << neighbor << endl;
                    can_add = false;
                    break;
                }
            }
            if (can_add)
            {
                nodes[u].membership = true;
            }
        }
    }
    else
    {
        // Lemma 5: Neither is in MIS, do nothing
        // No action needed
    }

    // Lemma 6: Both cannot be in MIS after deletion, which is already maintained
    // Finally, remove the edge from the graph
    // graph.removeEdge(u, v);
}

vector<double> timeFor_Clustering;
vector<double> timeFor_DUClustering;

// Function to process a batch of edges using edge clustering
void processBatchEdgeClustering(vector<Edge> &batch, Graph &graph, vector<Node> &nodes, int hop_limit, bool isFirstBatch = false)
{
    // Step 1: Cluster edges using edge-based approach
    double startTime = 0, endTime = 0;
    
    auto start_time = chrono::high_resolution_clock::now();
    vector<EdgeCluster> edge_clusters = clusterEdgesFromScratch(batch, graph, hop_limit);
    
    auto end_time = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(end_time - start_time).count() / 1000000.0;
    
    timeFor_DUClustering.push_back(duration);
    total_Clusters_DU += edge_clusters.size();
    
    // Step 2: Parallel processing of edge clusters
    int block, tid, start, end;
    block = ceil(static_cast<float>(edge_clusters.size()) / n_threads);
    
    #pragma omp parallel num_threads(n_threads) private(tid, start, end) shared(graph)
    {
        tid = omp_get_thread_num();
        startTime = omp_get_wtime();
        start = tid * block;
        end = std::min((tid + 1) * block, static_cast<int>(edge_clusters.size()));
        
        for (size_t i = start; i < end; ++i) {
            EdgeCluster& cluster = edge_clusters[i];
            
            // Process all edges in this cluster
            for (size_t j = 0; j < cluster.edges.size(); ++j) {
                Edge& edge = cluster.edges[j];
                
                #pragma omp critical
                {
                    if (!graph.isAdjacent(edge.source, edge.destination)) {
                        handleInsertion(edge, graph, nodes);
                        graph.insertEdge(edge.source, edge.destination);
                    } else {
                        handleDeletion(edge, graph, nodes);
                        graph.removeEdge(edge.source, edge.destination);
                    }
                }
            }
        }
        
        endTime = omp_get_wtime();
        time_per_thread[tid] = endTime - startTime;
    }
}

// Function to process a batch of edges using vertex clustering
void processBatch(vector<Edge> &batch, Graph &graph, vector<Node> &nodes, int hop_limit, bool isFirstBatch = false)
{
    // Step 1: Cluster vertices using optimized approach
    double startTime = 0, endTime = 0;
    
    auto start_time = chrono::high_resolution_clock::now();
    vector<VertexCluster> clusters = clusterVerticesOptimized(batch, graph, hop_limit);
    
    auto end_time = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(end_time - start_time).count() / 1000000.0;
    
    timeFor_Clustering.push_back(duration);
    total_Clusters += clusters.size();
    
    // Step 2: Parallel processing of vertex clusters
    int block, tid, start, end;
    block = ceil(static_cast<float>(clusters.size()) / n_threads);
    
    #pragma omp parallel num_threads(n_threads) private(tid, start, end) shared(graph)
    {
        tid = omp_get_thread_num();
        startTime = omp_get_wtime();
        start = tid * block;
        end = std::min((tid + 1) * block, static_cast<int>(clusters.size()));
        
        for (size_t i = start; i < end; ++i) {
            VertexCluster& cluster = clusters[i];
            
            // Process all edges that affect vertices in this cluster
            for (size_t j = 0; j < cluster.affected_edges.size(); ++j) {
                Edge& edge = cluster.affected_edges[j];
                
                #pragma omp critical
                {
                    if (!graph.isAdjacent(edge.source, edge.destination)) {
                        handleInsertion(edge, graph, nodes);
                        graph.insertEdge(edge.source, edge.destination);
                    } else {
                        handleDeletion(edge, graph, nodes);
                        graph.removeEdge(edge.source, edge.destination);
                    }
                }
            }
        }
        
        endTime = omp_get_wtime();
        time_per_thread[tid] = endTime - startTime;
    }
}

// Read Graphs from edgeList to adjacancy list
Graph *read_Graph(string filename)
{

    Graph *graph = new Graph();
    ifstream file(filename);
    if (file.is_open())
    {
        string line;
        while (getline(file, line) && line[0] == '%')
        {
            // ignore all lines starting with %
        }

        stringstream ss(line);
        int nodes, edgesCount;
        ss >> nodes >> nodes >> edgesCount;
        graph->num_nodes = nodes;
        graph->num_Edges = edgesCount;
        graph->adj_list.resize(nodes);

        while (getline(file, line))
        {
            int u, v;
            stringstream ss(line);
            ss >> u >> v;
            graph->adj_list[u].push_back(v);
            graph->adj_list[v].push_back(u);
        }
    }
    else
    {
        cout << "Unable to open file" << endl;
    }
    return graph;
}

int main(int argc, char *argv[])
{
    if (argc < 4)
    {
        std::cerr << "Usage: " << argv[0] << " <MTX> <MIS> <Batches> <Num_batch> <N_Threads>" << std::endl;
        return 1;
    }
    cout << "===================================================== ParMIS_v1 CPU Results ===============================================" << endl;
    cout << "----------------------------------------Start-----------------------------------------" << endl;

    n_threads = stoi(argv[5]);
    time_per_thread.resize(n_threads);

    // Read Graph -----------------------------------------------------
    std::string filename1 = argv[1];
    Graph graph = *read_Graph(filename1);
    int num_nodes = graph.num_nodes;
    cout << "Number of nodes: " << graph.num_nodes << endl;
    cout << "Number of edges: " << graph.num_Edges << endl;
    // // Initialize nodes
    vector<Node> nodes(num_nodes);
    for (int i = 0; i < num_nodes; ++i)
    {
        nodes[i] = Node(i, false); // Initially, no node is in the MIS
    }

    // Read MIS---------------------------------------------------------
    std::ifstream inputFile(argv[2]);
    int initial_card = 0;
    if (inputFile.is_open())
    {
        std::string line;
        while (std::getline(inputFile, line))
        {
            // maximalIndependentSet.push_back(std::stoi(line)+1);
            nodes[std::stoi(line) + 1].membership = true;
            initial_card++;
        }
        inputFile.close();
    }
    else
    {
        std::cerr << "Unable to open the file for reading" << std::endl;
    }
    cout << "Initial_card-> " << initial_card << endl;

    // Read Batches----------------------------------------------------------
    std::string folderPath(argv[3]);
    std::vector<std::pair<int, int>> edgeList;
    std::vector<std::vector<std::pair<int, int>>> edgeLists;
    vector<vector<Edge>> batches_of_edges;
    vector<Edge> batch_of_edges;

    for (const auto &entry : std::filesystem::directory_iterator(folderPath))
    {
        if (entry.is_regular_file())
        {
            std::ifstream file(entry.path());
            std::string line;
            while (std::getline(file, line))
            {
                std::istringstream iss(line);
                int src, dest;
                if (iss >> src)
                {
                    if (!(iss >> dest))
                    {
                        dest = -1;
                    }
                    edgeList.push_back(std::make_pair(src, dest));
                    if (graph.isAdjacent(src, dest))
                    {
                        batch_of_edges.push_back(Edge(src, dest, false));
                    }
                    else
                        batch_of_edges.push_back(Edge(src, dest, true));
                }
            }
            edgeLists.push_back(edgeList);
            edgeList.clear();
            batches_of_edges.push_back(batch_of_edges);
            batch_of_edges.clear();
        }
    }

    cout << "Batch Size:-> " << edgeLists[0].size() << " Ratio:-> " << folderPath[folderPath.size() - 1] << endl;

    int num_insertion_batches = stoi(argv[4]);
    // Define hop limit for clustering
    int hop_limit = 2;
    clusters.reserve(1000);
    time_per_thread.resize(n_threads, 0.0);

    // Create copies for edge clustering
    Graph graph_edge = graph;
    vector<Node> nodes_edge = nodes;
    
    // Run vertex clustering
    double time_cum_vertex = 0;
    cout << "Running Vertex Clustering..." << endl;
    for (int i = 0; i < num_insertion_batches; i++)
    {
        processBatch(batches_of_edges[i], graph, nodes, hop_limit, i == 0);
        double maxi = 0;
        for (int k = 0; k < time_per_thread.size(); k++)
        {
            if (maxi < time_per_thread[k])
            {
                maxi = time_per_thread[k];
            }
        }
        time_cum_vertex += maxi;
    }

    int updated_card_vertex = 0;
    for (int i = 0; i < num_nodes; i++)
    {
        if (nodes[i].membership)
            updated_card_vertex++;
    }

    // Reset time tracking for edge clustering
    time_per_thread.clear();
    time_per_thread.resize(n_threads, 0.0);

    // // Run edge clustering
    // double time_cum_edge = 0;
    // cout << "Running Edge Clustering..." << endl;
    // for (int i = 0; i < num_insertion_batches; i++)
    // {
    //     processBatchEdgeClustering(batches_of_edges[i], graph_edge, nodes_edge, hop_limit, i == 0);
    //     double maxi = 0;
    //     for (int k = 0; k < time_per_thread.size(); k++)
    //     {
    //         if (maxi < time_per_thread[k])
    //         {
    //             maxi = time_per_thread[k];
    //         }
    //     }
    //     time_cum_edge += maxi;
    // }

    // int updated_card_edge = 0;
    // for (int i = 0; i < num_nodes; i++)
    // {
    //     if (nodes_edge[i].membership)
    //         updated_card_edge++;
    // }

    double time_vertex_clustering = 0;
    for (int i = 0; i < timeFor_Clustering.size(); i++)
    {
        time_vertex_clustering += timeFor_Clustering[i];
    }
    // double time_edge_clustering = 0;
    // for (int i = 0; i < timeFor_DUClustering.size(); i++)
    // {
    //     time_edge_clustering += timeFor_DUClustering[i];
    // }
    
    cout << endl;
    cout << "================================================================================" << endl;
    cout << "                              CLUSTERING COMPARISON" << endl;
    cout << "================================================================================" << endl;
    cout << endl;
    
    cout << "-------------------------------Vertex Clustering------------------------------" << endl;
    cout << "Cardinality changed by-> " << abs(initial_card - updated_card_vertex) << endl;
    cout << "Total Incremental Edges Insertion time: " << fixed << setprecision(5) << time_cum_vertex * 1000 << " milliseconds" << endl;
    cout << "Average Incremental Edges Insertion time: " << fixed << setprecision(5) << time_cum_vertex / num_insertion_batches * 1000 << " milliseconds" << endl;
    cout << "Average Vertex Clustering Time per batch " << fixed << setprecision(5) << time_vertex_clustering / num_insertion_batches * 1000 << " milliseconds" << endl;
    cout << "Average Vertex Clusters per batch:- " << total_Clusters / num_insertion_batches << endl;

    // cout << endl;
    // cout << "-------------------------------Edge Clustering--------------------------------" << endl;
    // cout << "Cardinality changed by-> " << abs(initial_card - updated_card_edge) << endl;
    // cout << "Total Incremental Edges Insertion time: " << fixed << setprecision(5) << time_cum_edge * 1000 << " milliseconds" << endl;
    // cout << "Average Incremental Edges Insertion time: " << fixed << setprecision(5) << time_cum_edge / num_insertion_batches * 1000 << " milliseconds" << endl;
    // cout << "Average Edge Clustering Time per batch " << fixed << setprecision(5) << time_edge_clustering / num_insertion_batches * 1000 << " milliseconds" << endl;
    // cout << "Average Edge Clusters per batch:- " << total_Clusters_DU / num_insertion_batches << endl;

    // cout << endl;
    // cout << "-------------------------------Performance Comparison-------------------------" << endl;
    // cout << "Speedup (Edge vs Vertex): " << fixed << setprecision(3) << time_cum_vertex / time_cum_edge << "x" << endl;
    // cout << "Clustering Time Ratio (Edge vs Vertex): " << fixed << setprecision(3) << time_edge_clustering / time_vertex_clustering << "x" << endl;
    // cout << "Clusters Ratio (Edge vs Vertex): " << fixed << setprecision(3) << (double)(total_Clusters_DU) / total_Clusters << "x" << endl;
    // cout << "Final MIS cardinality - Vertex: " << updated_card_vertex << ", Edge: " << updated_card_edge << endl;
    // cout << "================================================================================" << endl;

    // Write results to file
    string output_filename = "clustering_comparison_results.txt";
    ofstream fout(output_filename);
    if (fout.is_open()) {
        fout << "================================================================================" << endl;
        fout << "                              CLUSTERING COMPARISON" << endl;
        fout << "================================================================================" << endl;
        fout << "Initial MIS cardinality: " << initial_card << endl;
        fout << "Number of batches: " << num_insertion_batches << endl;
        fout << "Threads: " << n_threads << endl;
        fout << "Hop limit: " << hop_limit << endl;
        fout << endl;
        
        fout << "-------------------------------Vertex Clustering------------------------------" << endl;
        fout << "Cardinality changed by-> " << abs(initial_card - updated_card_vertex) << endl;
        fout << "Total Incremental Edges Insertion time: " << fixed << setprecision(5) << time_cum_vertex * 1000 << " milliseconds" << endl;
        fout << "Average Incremental Edges Insertion time: " << fixed << setprecision(5) << time_cum_vertex / num_insertion_batches * 1000 << " milliseconds" << endl;
        fout << "Average Vertex Clustering Time per batch " << fixed << setprecision(5) << time_vertex_clustering / num_insertion_batches * 1000 << " milliseconds" << endl;
        fout << "Average Vertex Clusters per batch:- " << total_Clusters / num_insertion_batches << endl;

        // fout << endl;
        // fout << "-------------------------------Edge Clustering--------------------------------" << endl;
        // fout << "Cardinality changed by-> " << abs(initial_card - updated_card_edge) << endl;
        // fout << "Total Incremental Edges Insertion time: " << fixed << setprecision(5) << time_cum_edge * 1000 << " milliseconds" << endl;
        // fout << "Average Incremental Edges Insertion time: " << fixed << setprecision(5) << time_cum_edge / num_insertion_batches * 1000 << " milliseconds" << endl;
        // fout << "Average Edge Clustering Time per batch " << fixed << setprecision(5) << time_edge_clustering / num_insertion_batches * 1000 << " milliseconds" << endl;
        // fout << "Average Edge Clusters per batch:- " << total_Clusters_DU / num_insertion_batches << endl;

        // fout << endl;
        // fout << "-------------------------------Performance Comparison-------------------------" << endl;
        // fout << "Speedup (Edge vs Vertex): " << fixed << setprecision(3) << time_cum_vertex / time_cum_edge << "x" << endl;
        // fout << "Clustering Time Ratio (Edge vs Vertex): " << fixed << setprecision(3) << time_edge_clustering / time_vertex_clustering << "x" << endl;
        // fout << "Clusters Ratio (Edge vs Vertex): " << fixed << setprecision(3) << (double)(total_Clusters_DU) / total_Clusters << "x" << endl;
        // fout << "Final MIS cardinality - Vertex: " << updated_card_vertex << ", Edge: " << updated_card_edge << endl;
        // fout << "================================================================================" << endl;
        
        fout.close();
        cout << "Results saved to: " << output_filename << endl;
    }

    cout << endl;
    time_per_thread.clear();

    return 0;
}
// time ./t1 ../Datasets/batches_/scratch_hyrbid/livejournal/livejournal_3.mtx ./hybrid_mis/livejournal/livejournal_3.txt ../Datasets/batches_/batche/livejournal/3 2 64
// ./ECL/t1 Final_code/csr/Orkut.txt Final_code/mis/Orkut.txt Final_code/batches/Orkut/3/ 3 64
// ./ad ./Datasets_using/OtherGraphs/uk-2002.mtx ./small_Datasets/MISs/uk-2002.txt ./small_Datasets/Batches/uk-2002/3 3 64
//./ad ./Datasets_using/OtherGraphs/kmer_A2a.mtx ./small_Datasets/MISs/kmer_A2a.txt ./small_Datasets/Batches/kmer_A2a/3 3 64