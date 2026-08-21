import networkx as nx
import numpy as np
from tqdm import tqdm
import pickle
from pathlib import Path
from src.preprocessing.config import FEATURE_OUTPUT_PATH
from src.preprocessing.config import (
    FEATURE_OUTPUT_PATH,
    GRAPH_OUTPUT_PATH
)

def load_node_features(file_path):
    return np.load(file_path)

def create_temporal_graph(node_features):
    graph = nx.Graph()

    num_nodes = len(node_features)

    for i in range(num_nodes):
        graph.add_node(
            i,
            features=node_features[i]
        )

    for i in range(num_nodes - 1):
        graph.add_edge(i, i + 1)

    return graph

def save_graph(graph, file_path):
    genre = file_path.parent.name
    song_name = file_path.stem

    output_dir = GRAPH_OUTPUT_PATH / genre
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"{song_name}.pkl"

    with open(output_file, "wb") as f:
        pickle.dump(graph, f)

def process_single_graph(feature_file):

    node_features = load_node_features(feature_file)

    graph = create_temporal_graph(node_features)

    save_graph(graph, feature_file)

    return graph

def process_dataset_graphs():

    genres = [
        "blues",
        "classical",
        "country",
        "disco",
        "hiphop",
        "jazz",
        "metal",
        "pop",
        "reggae",
        "rock"
    ]

    total = 0

    for genre in genres:

        genre_path = FEATURE_OUTPUT_PATH / genre

        files = sorted(genre_path.glob("*.npy"))

        for file in tqdm(files, desc=genre):

            process_single_graph(file)

            total += 1

    print(f"\nCreated {total} graphs.")

if __name__ == "__main__":
    process_dataset_graphs()