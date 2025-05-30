import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from sklearn.metrics import normalized_mutual_info_score
import os

from utils.constants import DATA_DIR,OUT_DIR,FIG_DIR

class CommunityAnalysis:
    def __init__(self, data_dir=DATA_DIR, output_dir=OUT_DIR, fig_dir=FIG_DIR):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.fig_dir = fig_dir
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.fig_dir, exist_ok=True)
        self.method_files = [f"{self.data_dir}/method_{i}.parquet" for i in range(1, 6)]
        self.method_names = [f"Method_{i}" for i in range(1, 6)]
        self.community_assignments = self.load_data()

    def load_data(self):
        return {name: pd.read_parquet(file) for name, file in zip(self.method_names, self.method_files)}

    def summarize_communities(self):
        community_sizes = {}
        summary_stats = []
        for name, df in self.community_assignments.items():
            sizes = df.groupby("community")["node"].count().sort_values(ascending=False)
            community_sizes[name] = sizes
            summary_stats.append({
                "Method": name,
                "#Communities": len(sizes),
                "Largest": sizes.max(),
                "Smallest": sizes.min(),
                "Median Size": int(sizes.median()),
                "#Singletons": (sizes == 1).sum(),
                "Top_5_Community_Sizes": ", ".join(map(str, sizes.head(5).values))
            })
        summary_df = pd.DataFrame(summary_stats)
        summary_df.to_csv(f"{self.output_dir}/community_summary.csv", index=False)
        self.plot_community_size_distributions(community_sizes)
        return summary_df

    def plot_community_size_distributions(self, community_sizes):
        plt.figure(figsize=(18, 12))
        for i, name in enumerate(self.method_names):
            plt.subplot(2, 3, i+1)
            plt.hist(community_sizes[name].values, bins=30, alpha=0.7)
            plt.title(f"{name}: Community Size Distribution")
            plt.xlabel("Community Size")
            plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(f"{self.fig_dir}/community_size_distributions.png")
        plt.close()

    def compute_nmi_matrix(self):
        all_nodes = sorted(set.union(*(set(df["node"]) for df in self.community_assignments.values())))
        node_df = pd.DataFrame(index=all_nodes)
        for name, df in self.community_assignments.items():
            node_map = dict(zip(df["node"], df["community"]))
            node_df[name] = [node_map.get(node, -1) for node in all_nodes]

        N = len(self.method_names)
        nmi_mat = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                labels1 = node_df[self.method_names[i]].astype(str).values
                labels2 = node_df[self.method_names[j]].astype(str).values
                all_labels = np.unique(np.concatenate([labels1, labels2]))
                label_map = {label: idx for idx, label in enumerate(all_labels)}
                labels1_codes = np.array([label_map[lbl] for lbl in labels1])
                labels2_codes = np.array([label_map[lbl] for lbl in labels2])
                nmi_mat[i, j] = normalized_mutual_info_score(labels1_codes, labels2_codes)
        nmi_df = pd.DataFrame(nmi_mat, index=self.method_names, columns=self.method_names)
        nmi_df.to_csv(f"{self.output_dir}/nmi_similarity_matrix.csv")
        self.plot_nmi_matrix(nmi_df)
        return nmi_df

    def plot_nmi_matrix(self, nmi_df):
        plt.figure(figsize=(8, 6))
        sns.heatmap(nmi_df, annot=True, cmap="YlGnBu")
        plt.title("NMI Similarity Matrix")
        plt.tight_layout()
        plt.savefig(f"{self.fig_dir}/nmi_similarity_matrix.png")
        plt.close()

    def align_communities_jaccard(self, method_A, method_B):
        def get_community_map(df):
            comm_map = defaultdict(set)
            for _, row in df.iterrows():
                comm_map[row["community"]].add(row["node"])
            return comm_map

        df_A = self.community_assignments[method_A]
        df_B = self.community_assignments[method_B]
        comm_A = get_community_map(df_A)
        comm_B = get_community_map(df_B)

        node_to_A = defaultdict(set)
        node_to_B = defaultdict(set)
        for comm, nodes in comm_A.items():
            for node in nodes:
                node_to_A[node].add(comm)
        for comm, nodes in comm_B.items():
            for node in nodes:
                node_to_B[node].add(comm)
        candidate_pairs = set()
        for node in set(node_to_A) & set(node_to_B):
            for ca in node_to_A[node]:
                for cb in node_to_B[node]:
                    candidate_pairs.add((ca, cb))

        similarities = []
        for ca, cb in candidate_pairs:
            nodes_a = comm_A[ca]
            nodes_b = comm_B[cb]
            intersection = len(nodes_a & nodes_b)
            union = len(nodes_a | nodes_b)
            score = intersection / union if union > 0 else 0
            similarities.append((ca, cb, score))

        similarities.sort(key=lambda x: -x[2])
        best_matches = {}
        used_B = set()
        for a, b, score in similarities:
            if a not in best_matches and b not in used_B:
                best_matches[a] = (b, score)
                used_B.add(b)

        alignment_df = pd.DataFrame([
            {f"Community_{method_A}": a, f"Community_{method_B}": b, "Jaccard_Similarity": score}
            for a, (b, score) in best_matches.items()
        ]).sort_values(by="Jaccard_Similarity", ascending=False)
        alignment_df.to_csv(f"{self.output_dir}/aligned_communities_{method_A}_{method_B}.csv", index=False)
        self.plot_alignment_heatmap(alignment_df, method_A, method_B)
        return alignment_df

    def plot_alignment_heatmap(self, alignment_df, method_A, method_B, N_top=10):
        top_matches = alignment_df.head(N_top)
        heatmap_data = pd.pivot_table(
            top_matches,
            values="Jaccard_Similarity",
            index=f"Community_{method_A}",
            columns=f"Community_{method_B}",
            fill_value=0
        )
        plt.figure(figsize=(10, 8))
        sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="YlGnBu")
        plt.title(f"Top {N_top} Aligned Communities: {method_A} vs {method_B}")
        plt.tight_layout()
        plt.savefig(f"{self.fig_dir}/aligned_communities_heatmap_{method_A}_{method_B}.png")
        plt.close()

    def run_analysis(self):
        print("Starting exploratory data analysis...")
        self.summarize_communities()
        print("Computing similarity between methods...")
        nmi_df = self.compute_nmi_matrix()

        # Find most similar methods
        nmi_mat = nmi_df.values
        np.fill_diagonal(nmi_mat, -1)
        best_i, best_j = np.unravel_index(np.argmax(nmi_mat), nmi_mat.shape)
        method_A, method_B = self.method_names[best_i], self.method_names[best_j]
        print(f"Most similar methods: {method_A} and {method_B} (NMI={nmi_df.iloc[best_i, best_j]:.3f})")

        print(f"Aligning communities between {method_A} and {method_B}...")
        self.align_communities_jaccard(method_A, method_B)

        print("\nAnalysis complete.")
        print(f"- EDA: see '{self.output_dir}/community_summary.csv' and '{self.fig_dir}/community_size_distributions.png'")
        print(f"- NMI matrix: see '{self.output_dir}/nmi_similarity_matrix.csv' and '{self.fig_dir}/nmi_similarity_matrix.png'")
        print(f"- Aligned communities: see '{self.output_dir}/aligned_communities_{method_A}_{method_B}.csv' and heatmap in '{self.fig_dir}/aligned_communities_heatmap_{method_A}_{method_B}.png'")

if __name__ == "__main__":
    CommunityAnalysis().run_analysis()
    from publish_output_to_md import publish_to_md
    publish_to_md()  # Uses defaults: output, figures, output.md