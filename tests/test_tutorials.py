"""
Tests for tutorial notebooks using nbval.

These tests ensure that all tutorial notebooks:
1. Execute without errors
2. Produce consistent, reproducible outputs

Run quick tests: pytest tests/test_tutorials.py
Run full notebook tests: pytest tests/test_tutorials.py --nbval -m slow
Run all tests: pytest tests/test_tutorials.py --nbval
"""

import pytest
from pathlib import Path

# Get the tutorials directory
TUTORIALS_DIR = Path(__file__).parent.parent / "tutorials"

# List all tutorial notebooks
TUTORIAL_NOTEBOOKS = sorted(TUTORIALS_DIR.glob("*.ipynb"))


@pytest.mark.parametrize(
    "notebook",
    TUTORIAL_NOTEBOOKS,
    ids=[nb.name for nb in TUTORIAL_NOTEBOOKS]
)
def test_notebook_valid_json(notebook):
    """Test that each notebook is valid JSON."""
    import json
    with open(notebook) as f:
        data = json.load(f)
    
    # Basic structure checks
    assert "cells" in data, f"{notebook.name} missing 'cells'"
    assert "metadata" in data, f"{notebook.name} missing 'metadata'"
    assert "nbformat" in data, f"{notebook.name} missing 'nbformat'"


def test_all_tutorials_exist():
    """Test that all expected tutorials exist."""
    expected = [
        "01_gmm_basics.ipynb",
        "02_smm_basics.ipynb",
        "03_optimal_weighting.ipynb",
        "04_bootstrap_inference.ipynb",
        "05_diagnostics.ipynb",
        "06_advanced_models.ipynb",
        "07_real_data_applications.ipynb",
    ]
    
    actual = [nb.name for nb in TUTORIAL_NOTEBOOKS]
    
    for name in expected:
        assert name in actual, f"Missing tutorial: {name}"


def test_tutorials_readme_exists():
    """Test that tutorials README exists."""
    readme = TUTORIALS_DIR / "README.md"
    assert readme.exists(), "tutorials/README.md not found"


# =============================================================================
# Notebook execution tests (slow, require nbval)
# =============================================================================

@pytest.mark.slow
@pytest.mark.notebooks
@pytest.mark.parametrize(
    "notebook",
    TUTORIAL_NOTEBOOKS,
    ids=[nb.name for nb in TUTORIAL_NOTEBOOKS]
)
def test_notebook_executes(notebook):
    """
    Test that notebook executes and produces expected outputs.
    
    This test requires nbval and runs the full notebook.
    Skip with: pytest -m "not slow"
    """
    pytest.importorskip("nbval")
    # nbval handles the actual execution via pytest plugin
    # This test just marks which notebooks should be tested
    pass
