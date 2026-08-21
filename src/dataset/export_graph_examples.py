import os
import glob
import pickle
import torch
from torch_geometric.data import Data


# ============================================================
# Configuration
# ============================================================

SOURCE_DIR = "data/processed/graphs"
OUTPUT_DIR = "data/processed/graphs/examples"

GENRES = [
    "blues",
    "classical",
    "country",
    "disco",
    "hiphop",
    "jazz",
    "metal",
    "pop",
    "reggae",
    "rock",
]

GRAPHS_PER_GENRE = 2


# ============================================================
# Create output directory
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# Convert one NetworkX graph to PyTorch Geometric Data
# ============================================================

def convert_graph(nx_graph, genre, source_file):

    # --------------------------------------------------------
    # Node features
    # --------------------------------------------------------

    node_features = []

    # Sort nodes to guarantee consistent ordering
    nodes = sorted(nx_graph.nodes())

    for node in nodes:
        features = nx_graph.nodes[node].get("features")

        if features is None:
            raise ValueError(
                f"Node {node} in {source_file} has no 'features' attribute."
            )

        # Convert numpy array -> torch tensor
        features = torch.tensor(features, dtype=torch.float32)

        node_features.append(features)

    x = torch.stack(node_features)


    # --------------------------------------------------------
    # Edges
    # --------------------------------------------------------

    edge_list = []

    for source, target in nx_graph.edges():
        edge_list.append([source, target])
        edge_list.append([target, source])

    if edge_list:
        edge_index = torch.tensor(
            edge_list,
            dtype=torch.long
        ).t().contiguous()
    else:
        edge_index = torch.empty(
            (2, 0),
            dtype=torch.long
        )


    # --------------------------------------------------------
    # PyTorch Geometric graph
    # --------------------------------------------------------

    data = Data(
        x=x,
        edge_index=edge_index
    )

    # Metadata
    data.genre = genre
    data.source_file = source_file
    data.num_nodes = x.size(0)
    data.num_edges = nx_graph.number_of_edges()

    return data


# ============================================================
# Export examples
# ============================================================

total_exported = 0

print("=" * 60)
print("EXPORTING GRAPH EXAMPLES")
print("=" * 60)

for genre in GENRES:

    genre_dir = os.path.join(SOURCE_DIR, genre)

    pattern = os.path.join(
        genre_dir,
        "*.pkl"
    )

    files = sorted(glob.glob(pattern))

    if len(files) < GRAPHS_PER_GENRE:
        print(
            f"WARNING: {genre} only has {len(files)} graphs."
        )
        selected_files = files
    else:
        selected_files = files[:GRAPHS_PER_GENRE]


    print(f"\n{genre}: {len(selected_files)} graphs")


    for index, source_file in enumerate(selected_files):

        # ----------------------------------------------------
        # Load NetworkX graph
        # ----------------------------------------------------

        with open(source_file, "rb") as f:
            graph = pickle.load(f)


        # ----------------------------------------------------
        # Convert
        # ----------------------------------------------------

        data = convert_graph(
            graph,
            genre,
            source_file
        )


        # ----------------------------------------------------
        # Output filename
        # ----------------------------------------------------

        output_file = os.path.join(
            OUTPUT_DIR,
            f"{genre}_{index:05d}.pt"
        )


        # ----------------------------------------------------
        # Save
        # ----------------------------------------------------

        torch.save(
            data,
            output_file
        )


        print(
            f"  ✓ {os.path.basename(source_file)} "
            f"-> {os.path.basename(output_file)} "
            f"(nodes={data.num_nodes}, "
            f"edges={data.num_edges}, "
            f"features={data.x.shape})"
        )

        total_exported += 1


# ============================================================
# Final summary
# ============================================================

print("\n" + "=" * 60)
print("EXPORT COMPLETE")
print("=" * 60)

print(f"Total graphs exported: {total_exported}")
print(f"Output directory: {OUTPUT_DIR}")

if total_exported >= 20:
    print("\n✓ Requirement #2 has the required 20+ graph samples.")
else:
    print(
        f"\nWARNING: Only {total_exported} graphs were exported."
    )