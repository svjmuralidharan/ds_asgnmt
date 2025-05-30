import os
import pytest
import pandas as pd
import numpy as np

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from utils.constants import DATA_DIR, OUT_DIR, FIG_DIR
from community_analysis import CommunityAnalysis


@pytest.fixture(scope="module")
def ca():
    # Remove old output files for a clean test
    for fname in ["community_summary.csv", "nmi_similarity_matrix.csv"]:
        fpath = os.path.join(OUT_DIR, fname)
        if os.path.exists(fpath):
            os.remove(fpath)
    return CommunityAnalysis(data_dir=DATA_DIR, output_dir=OUT_DIR, fig_dir=FIG_DIR)

def test_load_data(ca):
    # Ensure data loads for all five methods
    assert len(ca.community_assignments) == 5
    for name, df in ca.community_assignments.items():
        assert isinstance(df, pd.DataFrame)
        assert not df.empty

def test_summarize_communities(ca):
    summary_df = ca.summarize_communities()
    # Output file exists and has correct columns
    out_path = os.path.join(OUT_DIR, "community_summary.csv")
    assert os.path.exists(out_path)
    df = pd.read_csv(out_path)
    assert "Method" in df.columns
    assert df.shape[0] == 5

def test_nmi_matrix(ca):
    nmi_df = ca.compute_nmi_matrix()
    out_path = os.path.join(OUT_DIR, "nmi_similarity_matrix.csv")
    assert os.path.exists(out_path)
    df = pd.read_csv(out_path, index_col=0)
    assert df.shape[0] == 5
    assert df.shape[0] == df.shape[1]
    # Use np.allclose to check diagonal is (approximately) all 1s
    assert np.allclose(np.diag(df.values), 1.0)

def test_run_analysis_and_figures(tmp_path):
    # Use temporary folders for outputs to avoid clobbering real outputs
    temp_out = tmp_path / "output"
    temp_fig = tmp_path / "figures"
    temp_out.mkdir()
    temp_fig.mkdir()
    ca = CommunityAnalysis(output_dir=str(temp_out), fig_dir=str(temp_fig))
    ca.run_analysis()
    # Output and figure files should be created
    assert any(fname.endswith(".csv") for fname in os.listdir(temp_out))
    assert any(fname.endswith(".png") for fname in os.listdir(temp_fig))
