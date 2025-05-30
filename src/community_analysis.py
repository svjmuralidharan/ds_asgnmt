import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from sklearn.metrics import normalized_mutual_info_score
import os
import logging
import glob
import re

from utils.constants import (
    DATA_DIR, OUT_DIR, FIG_DIR, METHOD_FILE_PATTERN,
    COMMUNITY_SUMMARY_CSV, NMI_MATRIX_CSV,
    COMMUNITY_SIZE_PNG, NMI_MATRIX_PNG
)

# --- Set up logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

class CommunityAnalysis:
    """
    Analyzes and compares community detection outputs from multiple methods.
    Performs EDA, computes similarity matrices, and aligns community assignments.
    """

    def __init__(self, data_dir=DATA_DIR, output_dir=OUT_DIR, fig_dir=FIG_DIR):
        """
        Initializes CommunityAnalysis, finds method files, and loads data.
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.fig_dir = fig_dir
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.fig_dir, exist_ok=True)

        # Find and sort method files using parameterized pattern
        pattern = os.path.join(self.data_dir, METHOD_FILE_PATTERN)
        files = glob.glob(pattern)
        numbered_files = []
        for f in files:
            match = re.search(r"method_(\d+)\.parquet", os.path.basename(f))
            if match:
                numbered_files.append((int(match.group(1)), f))
        numbered_files.sort()
        self.method_files = [f for _, f in numbered_files]
        self.method_names = [f"Method_{n}" for n, _ in numbered_files]

        if not self.method_files:
            logger.error("No method parquet files found in %s", self.data_dir)
            raise FileNotFoundError(f"No method parquet files found in {self.data_dir}")
        logger.info("Detected %d method files: %s", len(self.method_files), self.method_names)
        self.community_assignments = self.load_data()

    def load_data(self):
        """
        Loads all community assignment .parquet files for each method into DataFrames.
        Returns:
            dict: Mapping of method name to DataFrame.
        """
        logger.info("Loading data from parquet files...")
        assignments = {}
        for name, file in zip(self.method_names, self.method_files):
            if not os.path.exists(file):
                logger.error("File not found: %s", file)
                raise FileNotFoundError(f"Missing data file: {file}")
            df = pd.read_parquet(file)
            logger.info("Loaded %s: %d rows", file, len(df))
            assignments[name] = df
        return assignments

    def summarize_communities(self):
        """
        Computes summary statistics for communities in each method and saves as CSV.
        Also plots the distribution of community sizes.
        Returns:
            pd.DataFrame: DataFrame of summary statistics.
        """
        logger.info("Summarizing communities for each method...")
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
            logger.info("Method %s: %d communities", name, len(sizes))
        summary_df = pd.DataFrame(summary_stats)
        out_path = os.path.join(self.output_dir, COMMUNITY_SUMMARY_CSV)
        summary_df.to_csv(out_path, index=False)
        logger.info("Saved community summary to %s", out_path)
        self.plot_community_size_distributions(community_sizes)
        return summary_df

    def plot_community_size_distributions(self, community_sizes):
        """
        Plots and saves the histogram of community sizes for each method.
        Args:
            community_sizes (dict): Mapping of method name to community size Series.
        """
        logger.info("Plotting community size distributions...")
        plt.figure(figsize=(18, 12))
        for i, name in enumerate(self.method_names):
            plt.subplot(2, 3, i+1)
            plt.hist(community_sizes[name].values, bins=30, alpha=0.7)
            plt.title(f"{name}: Community Size Distribution")
            plt.xlabel("Community Size")
            plt.ylabel("Frequency")
        plt.tight_layout()
        fig_path = os.path.join(self.fig_dir, COMMUNITY_SIZE_PNG)
        plt.savefig(fig_path)
        plt.close()
        logger.info("Saved community size distribution plot to %s", fig_path)

    def compute_nmi_matrix(self):
        """
        Computes the pairwise Normalized Mutual Information (NMI) similarity matrix
        for all community assignment methods and saves as CSV and heatmap PNG.
        Returns:
            pd.DataFrame: NMI similarity matrix (methods x methods)
        """
        logger.info("Computing NMI similarity matrix between all methods...")
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
        out_path = os.path.join(self.output_dir, NMI_MATRIX_CSV)
        nmi_df.to_csv(out_path)
        logger.info("Saved NMI similarity matrix to %s", out_path)
        self.plot_nmi_matrix(nmi_df)
        return nmi_df

    def plot_nmi_matrix(self, nmi_df):
        """
        Plots and saves a heatmap of the NMI similarity matrix.
        Args:
            nmi_df (pd.DataFrame): NMI similarity matrix DataFrame.
        """
        logger.info("Plotting NMI similarity matrix heatmap...")
        plt.figure(figsize=(8, 6))
        sns.heatmap(nmi_df, annot=True, cmap="YlGnBu")
        plt.title("NMI Similarity Matrix")
        plt.tight_layout()
        fig_path = os.path.join(self.fig_dir, NMI_MATRIX_PNG)
        plt.savefig(fig_path)
        plt.close()
        logger.info("Saved NMI matrix heatmap to %s", fig_path)

    def align_communities_jaccard(self, method_A, method_B):
        """
        Aligns communities between two methods using greedy one-to-one Jaccard similarity.
        Saves alignment as CSV and plots a heatmap of top aligned communities.
        Args:
            method_A (str): Name of first method.
            method_B (str): Name of second method.
        Returns:
            pd.DataFrame: DataFrame of best matching communities and similarities.
        """
        logger.info("Aligning communities using Jaccard similarity between %s and %s...", method_A, method_B)

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

        # Use consistent file naming if you parameterize this in constants.py
        out_path = os.path.join(self.output_dir, f"aligned_communities_{method_A}_{method_B}.csv")
        alignment_df = pd.DataFrame([
            {f"Community_{method_A}": a, f"Community_{method_B}": b, "Jaccard_Similarity": score}
            for a, (b, score) in best_matches.items()
        ]).sort_values(by="Jaccard_Similarity", ascending=False)
        alignment_df.to_csv(out_path, index=False)
        logger.info("Saved community alignment to %s", out_path)
        self.plot_alignment_heatmap(alignment_df, method_A, method_B)
        return alignment_df

    def plot_alignment_heatmap(self, alignment_df, method_A, method_B, N_top=10):
        """
        Plots a heatmap of the top N aligned communities by Jaccard similarity.
        Args:
            alignment_df (pd.DataFrame): DataFrame of aligned communities.
            method_A (str): First method name.
            method_B (str): Second method name.
            N_top (int): Number of top matches to plot.
        """
        logger.info("Plotting heatmap of top %d aligned communities between %s and %s...", N_top, method_A, method_B)
        top_matches = alignment_df.head(N_top)
        heatmap_data = pd.pivot_table(
            top_matches,
            values="Jaccard_Similarity",
            index=f"Community_{method_A}",
            columns=f"Community_{method_B}",
            fill_value=0
        )
        fig_path = os.path.join(self.fig_dir, f"aligned_communities_heatmap_{method_A}_{method_B}.png")
        plt.figure(figsize=(10, 8))
        sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="YlGnBu")
        plt.title(f"Top {N_top} Aligned Communities: {method_A} vs {method_B}")
        plt.tight_layout()
        plt.savefig(fig_path)
        plt.close()
        logger.info("Saved aligned communities heatmap to %s", fig_path)

    def run_analysis(self):
        """
        Orchestrates the full analysis workflow: EDA, NMI matrix, and community alignment.
        """
        logger.info("Starting exploratory data analysis...")
        self.summarize_communities()
        logger.info("Computing similarity between methods...")
        nmi_df = self.compute_nmi_matrix()

        # Find most similar methods by NMI (excluding diagonal)
        nmi_mat = nmi_df.values
        np.fill_diagonal(nmi_mat, -1)
        best_i, best_j = np.unravel_index(np.argmax(nmi_mat), nmi_mat.shape)
        method_A, method_B = self.method_names[best_i], self.method_names[best_j]
        logger.info("Most similar methods: %s and %s (NMI=%.3f)", method_A, method_B, nmi_df.iloc[best_i, best_j])

        logger.info("Aligning communities between %s and %s...", method_A, method_B)
        self.align_communities_jaccard(method_A, method_B)

        logger.info("Analysis complete.")
        logger.info("- EDA: see '%s' and '%s'", os.path.join(self.output_dir, COMMUNITY_SUMMARY_CSV), os.path.join(self.fig_dir, COMMUNITY_SIZE_PNG))
        logger.info("- NMI matrix: see '%s' and '%s'", os.path.join(self.output_dir, NMI_MATRIX_CSV), os.path.join(self.fig_dir, NMI_MATRIX_PNG))
        logger.info("- Aligned communities: see 'aligned_communities_{method_A}_{method_B}.csv' and heatmap in 'aligned_communities_heatmap_{method_A}_{method_B}.png'")

if __name__ == "__main__":
    CommunityAnalysis().run_analysis()
    # Optionally, publish to markdown summary if available
    try:
        from publish_output_to_md import publish_to_md
        publish_to_md()
        logger.info("Markdown summary created (output.md)")
    except ImportError:
        logger.warning("publish_output_to_md module not found. Skipping markdown summary.")
