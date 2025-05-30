import os
import pytest
import pandas as pd
import numpy as np
import glob

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from utils.constants import (
    DATA_DIR, OUT_DIR, FIG_DIR,
    METHOD_FILE_PATTERN, COMMUNITY_SUMMARY_CSV, NMI_MATRIX_CSV
)
from community_analysis import CommunityAnalysis

def count_method_files():
    # Count how many method parquet files are in the data directory, using the pattern
    pattern = os.path.join(DATA_DIR, METHOD_FILE_PATTERN)
    return len(glob.glob(pattern))

@pytest.fixture(scope="module")
def ca():
    # Remove old output files for a clean test
    for fname in [COMMUNITY_SUMMARY_CSV, NMI_MATRIX_CSV]:
        fpath = os.path.join(OUT_DIR, fname)
        if os.path.exists(fpath):
            os.remove(fpath)
    return CommunityAnalysis(data_dir=DATA_DIR, output_dir=OUT_DIR, fig_dir=FIG_DIR)

def test_load_data(ca):
    # Ensure data loads for all detected method files
    num_methods = count_method_files()
    assert len(ca.community_assignments) == num_methods
    for name, df in ca.community_assignments.items():
        assert isinstance(df, pd.DataFrame)
        assert not df.empty

def test_summarize_communities(ca):
    summary_df = ca.summarize_communities()
    # Output file exists and has correct columns
    out_path = os.path.join(OUT_DIR, COMMUNITY_SUMMARY_CSV)
    assert os.path.exists(out_path)
    df = pd.read_csv(out_path)
    assert "Method" in df.columns
    num_methods = count_method_files()
    assert df.shape[0] == num_methods

def test_nmi_matrix(ca):
    nmi_df = ca.compute_nmi_matrix()
    out_path = os.path.join(OUT_DIR, NMI_MATRIX_CSV)
    assert os.path.exists(out_path)
    df = pd.read_csv(out_path, index_col=0)
    num_methods = count_method_files()
    assert df.shape[0] == num_methods
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
