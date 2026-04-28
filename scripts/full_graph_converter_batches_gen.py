import sys
import random
num_edges=0
batch_size=0
# num_nodes=0
def read_edge_list(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
        
        # Detect metadata lines
        metadata_end_index = 0
        for i, line in enumerate(lines):
            if line.startswith('%'):
                metadata_end_index = i + 1
            else:
                break
        
        # Read metadata
        metadata = lines[:metadata_end_index]
        
        # Read num_rows, num_cols, num_edges
        num_rows, num_cols, num_edges = map(int, lines[metadata_end_index].split())
        
        # Read edge list
        edge_list = [tuple(map(int, line.split())) for line in lines[metadata_end_index + 1:]]
        
    return metadata, num_rows, num_edges, edge_list


def split_graph(edge_list, split_ratio):
    global num_edges
    random.shuffle(edge_list)
    split_index = int(len(edge_list) * split_ratio)
    graph_90 = edge_list[:split_index]
    graph_10 = edge_list[split_index:]
    num_edges=num_edges-len(graph_10)
    return graph_90, graph_10

def create_batches(graph_10, batch_size):
    batches = [graph_10[i:i+batch_size] for i in range(0, len(graph_10), batch_size)]
    return batches

def join_with_90(graph_90, batch):
    return graph_90 + batch

def add_backedges(graph):
    backedges = [(edge[1], edge[0]) for edge in graph]
    return graph + backedges

def save_graph(filename, metadata, graph,num_nodes):
    global num_edges
    with open(filename, 'w') as f:
        f.writelines(metadata)
        # f.write('\n')
        f.write(f"{num_nodes} {num_nodes} {2*(num_edges)}\n")
        for edge in graph:
            f.write(f"{edge[0]} {edge[1]}\n")
def save_batches(filename, metadata, batch):
    with open(filename, 'w') as f:
        # f.writelines(metadata)
        # f.write('\n')
        # for batch in batches:
            for edge in batch:
                f.write(f"{edge[0]} {edge[1]}\n")
def main():
    global num_edges, batch_size
    metadata,num_nodes,num_edges, edge_list = read_edge_list(input_file)
    print(len(edge_list))
    
    if to_save_converted_graph:
        # Save the converted graph with backedges
        graph_95, graph_5 = split_graph(edge_list, 0.95)
        print("Graph Spilt Completed")
        print("Saving converted graph with backedges...")
        scratch_graph_with_backedges = add_backedges(graph_95)
        # Save the graph
        print(f"Saving graph to {output_file}...")
        save_graph(output_file, metadata, scratch_graph_with_backedges,num_nodes)
        print("Graph saved successfully.")
    else:
        graph_90, graph_10 = split_graph(edge_list, 0.90)
        print("Graph Spilt Completed")
        batch_size=int(num_edges*batch_size)
        batches = create_batches(graph_10, batch_size)
        print("Number of batches created:", len(batches))
        print(f"Batch size: {batch_size}")
        print("Skipping saving converted graph with backedges.")
        for i, batch in enumerate(batches):
                # batch_graph
                # batch_graph_with_backedges = add_backedges(batch_graph)
                save_batches(f".{output_batches.split('.')[1]}{i+1}.mtx", metadata, batch)
    
   # Save batches
    print("done")
if __name__ == "__main__":
    to_save_converted_graph=False
    if(len(sys.argv) == 4):
        input_file = sys.argv[1]
        output_batches=sys.argv[2]
        batch_size = float(sys.argv[3])
    elif (len(sys.argv) == 3):
        input_file = sys.argv[1]
        output_file = sys.argv[2]
        to_save_converted_graph = True
    else:
        print("Usage: python full_graph_converter_batches_gen.py <input_file> <output_file> [batch_size] OR")
        print("Usage: python full_graph_converter_batches_gen.py <input_file> <Output_file>")
        sys.exit(1)
    
    main()
