import os
from utils.constants import (
    OUT_DIR, FIG_DIR,
    COMMUNITY_SUMMARY_CSV, NMI_MATRIX_CSV
)

def describe_file(fname):
    """Return a friendly description for well-known output files."""
    if fname == COMMUNITY_SUMMARY_CSV:
        return "Summary statistics for each community detection method."
    elif fname == NMI_MATRIX_CSV:
        return "Pairwise Normalized Mutual Information (NMI) similarity matrix."
    elif fname.startswith("aligned_communities_") and fname.endswith(".csv"):
        return "Aligned communities (greedy Jaccard match) between two most similar methods."
    elif fname.endswith(".pdf"):
        return "PDF summary report with figures and tables."
    elif fname.endswith(".png") and "size_distribution" in fname:
        return "Histogram of community size distributions for all methods."
    elif fname.endswith(".png") and "nmi_similarity" in fname:
        return "Heatmap of NMI similarity matrix."
    elif fname.endswith(".png") and "aligned_communities_heatmap" in fname:
        return "Heatmap of top aligned communities for the two most similar methods."
    else:
        return ""

def publish_to_md(output_dir=OUT_DIR, fig_dir=FIG_DIR, md_file="output.md"):
    """
    Generate a Markdown file listing all key outputs and figures with short descriptions.
    """
    lines = []
    lines.append("# Analysis Outputs\n")
    lines.append("This document lists all key files generated by the analysis, with short descriptions and figures (where available).\n")

    # Output files
    lines.append(f"## Data Outputs (`{output_dir}/`)\n")
    for fname in sorted(os.listdir(output_dir)):
        desc = describe_file(fname)
        lines.append(f"- **`{fname}`**: {desc if desc else ''}")
    lines.append("")

    # Figures
    lines.append(f"## Figures (`{fig_dir}/`)\n")
    for fname in sorted(os.listdir(fig_dir)):
        if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        desc = describe_file(fname)
        lines.append(f"### {fname}")
        if desc:
            lines.append(f"*{desc}*")
        # Markdown image embedding (relative paths)
        lines.append(f"![{fname}]({fig_dir}/{fname})")
        lines.append("")

    # Write to output.md (in current working directory or as specified)
    with open(md_file, "w") as f:
        f.write("\n".join(lines))
    print(f"Wrote Markdown summary to {md_file}")

if __name__ == "__main__":
    publish_to_md()
