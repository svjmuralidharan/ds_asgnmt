import pandas as pd
import os
from collections import defaultdict

method_A = "Method_2"
method_B = "Method_4"
file_A = "data/method_2.parquet"
file_B = "data/method_4.parquet"

out_dir = "output"

df_A = pd.read_parquet(file_A)
df_B = pd.read_parquet(file_B)

def get_community_map(df):
    comm_map = defaultdict(set)
    for _, row in df.iterrows():
        comm_map[row["community"]].add(row["node"])
    return comm_map

comm_A = get_community_map(df_A)
comm_B = get_community_map(df_B)

# --- Build node to communities mapping ---
node_to_A = defaultdict(set)
node_to_B = defaultdict(set)
for comm, nodes in comm_A.items():
    for node in nodes:
        node_to_A[node].add(comm)
for comm, nodes in comm_B.items():
    for node in nodes:
        node_to_B[node].add(comm)

# --- Only compare communities that share at least one node ---
candidate_pairs = set()
for node in set(node_to_A) & set(node_to_B):
    for ca in node_to_A[node]:
        for cb in node_to_B[node]:
            candidate_pairs.add((ca, cb))

# --- Compute Jaccard only for candidate pairs ---
similarities = []
for ca, cb in candidate_pairs:
    nodes_a = comm_A[ca]
    nodes_b = comm_B[cb]
    intersection = len(nodes_a & nodes_b)
    union = len(nodes_a | nodes_b)
    score = intersection / union if union > 0 else 0
    similarities.append((ca, cb, score))

# --- Greedy one-to-one matching ---
similarities.sort(key=lambda x: -x[2])
best_matches = {}
used_B = set()
for a, b, score in similarities:
    if a not in best_matches and b not in used_B:
        best_matches[a] = (b, score)
        used_B.add(b)

# --- Save the results ---
import pandas as pd
alignment_df = pd.DataFrame([
    {f"Community_{method_A}": a, f"Community_{method_B}": b, "Jaccard_Similarity": score}
    for a, (b, score) in best_matches.items()
]).sort_values(by="Jaccard_Similarity", ascending=False)

alignment_df.to_csv(os.path.join(out_dir,f"aligned_communities_{method_A}_{method_B}.csv"), index=False)
#jaccard_df.to_csv(os.path.join(out_dir, "community_detection_jaccard_similarity_matrix.csv"), index=False)
print(f"Aligned and sorted communities saved to 'aligned_communities_{method_A}_{method_B}.csv'.")
print(alignment_df.head(15))
