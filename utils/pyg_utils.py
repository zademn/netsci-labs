def print_dataset(dataset):
    print(f"Dataset: {dataset}:")
    print("======================")
    print(f"Number of graphs: {len(dataset)}")
    print(f"Number of features: {dataset.num_features}")
    print(f"Number of classes: {dataset.num_classes}")


def print_data(data):
    print(data)
    print("=" * 62)

    # Gather some statistics about the graph.
    print(f"Number of nodes: {data.num_nodes}")
    print(f"Number of edges: {data.num_edges}")
    print(f"Average node degree: {data.num_edges / data.num_nodes:.2f}")
    if hasattr(data, "train_mask"):
        print(f"Number of training nodes: {data.train_mask.sum()}")
        print(f"Training node label rate: {int(data.train_mask.sum()) / data.num_nodes:.2f}")
    print(f"Contains isolated nodes: {data.has_isolated_nodes()}")
    print(f"Contains self-loops: {data.has_self_loops()}")
    print(f"Is undirected: {data.is_undirected()}")
