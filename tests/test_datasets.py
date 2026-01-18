"""Tests for the datasets module."""

import numpy as np
import pytest
from momentest import (
    load_econ381,
    load_econ381_bundle,
    load_consumption,
    load_labor_supply,
    load_asset_pricing,
    load_dataset,
    list_datasets,
    DatasetBundle,
)


class TestListDatasets:
    """Tests for list_datasets function."""

    def test_returns_list(self):
        """Should return a list."""
        result = list_datasets()
        assert isinstance(result, list)

    def test_contains_all_datasets(self):
        """Should contain all expected datasets."""
        result = list_datasets()
        expected = ['econ381', 'consumption', 'labor_supply', 'asset_pricing']
        for name in expected:
            assert name in result, f"Missing dataset: {name}"


class TestLoadEcon381:
    """Tests for load_econ381 function."""

    def test_returns_dict(self):
        """Should return a dictionary."""
        result = load_econ381()
        assert isinstance(result, dict)

    def test_has_required_keys(self):
        """Should have all required keys."""
        result = load_econ381()
        required_keys = ['data', 'n', 'bounds', 'mle_params', 'description']
        for key in required_keys:
            assert key in result, f"Missing key: {key}"

    def test_data_is_numpy_array(self):
        """Data should be a numpy array."""
        result = load_econ381()
        assert isinstance(result['data'], np.ndarray)

    def test_data_shape(self):
        """Data should have 161 observations."""
        result = load_econ381()
        assert result['n'] == 161
        assert len(result['data']) == 161

    def test_data_bounds(self):
        """Data should be within bounds [0, 450]."""
        result = load_econ381()
        data = result['data']
        lower, upper = result['bounds']
        assert data.min() >= lower
        assert data.max() <= upper

    def test_data_statistics(self):
        """Data should have expected statistics."""
        result = load_econ381()
        data = result['data']
        assert 340 < data.mean() < 345
        assert 7500 < data.var() < 8000

    def test_mle_params(self):
        """MLE params should have mu and sigma."""
        result = load_econ381()
        assert 'mu' in result['mle_params']
        assert 'sigma' in result['mle_params']
        assert result['mle_params']['mu'] == 622.16
        assert result['mle_params']['sigma'] == 198.76


class TestLoadEcon381Bundle:
    """Tests for load_econ381_bundle function."""

    def test_returns_dataset_bundle(self):
        """Should return a DatasetBundle."""
        result = load_econ381_bundle()
        assert isinstance(result, DatasetBundle)

    def test_has_correct_name(self):
        """Should have correct name."""
        result = load_econ381_bundle()
        assert result.name == "econ381"

    def test_has_data(self):
        """Should have data dictionary."""
        result = load_econ381_bundle()
        assert 'scores' in result.data
        assert len(result.data['scores']) == 161

    def test_has_exercises(self):
        """Should have exercises list."""
        result = load_econ381_bundle()
        assert len(result.exercises) > 0

    def test_difficulty_is_beginner(self):
        """Should be beginner difficulty."""
        result = load_econ381_bundle()
        assert result.difficulty == "beginner"


class TestLoadConsumption:
    """Tests for load_consumption function."""

    def test_returns_dataset_bundle(self):
        """Should return a DatasetBundle."""
        result = load_consumption()
        assert isinstance(result, DatasetBundle)

    def test_has_correct_name(self):
        """Should have correct name."""
        result = load_consumption()
        assert result.name == "consumption"

    def test_has_required_data(self):
        """Should have required data arrays."""
        result = load_consumption()
        assert 'c_growth' in result.data
        assert 'c_growth_lag1' in result.data
        assert 'c_growth_lag2' in result.data

    def test_data_shapes_match(self):
        """All data arrays should have same length."""
        result = load_consumption()
        n = result.n
        assert len(result.data['c_growth']) == n
        assert len(result.data['c_growth_lag1']) == n
        assert len(result.data['c_growth_lag2']) == n

    def test_consumption_growth_reasonable(self):
        """Consumption growth should be around 1 (quarterly)."""
        result = load_consumption()
        cg = result.data['c_growth']
        # Quarterly growth should be close to 1 (0-2% typical)
        assert 0.95 < cg.mean() < 1.05
        assert cg.min() > 0.8  # No huge drops
        assert cg.max() < 1.2  # No huge jumps

    def test_difficulty_is_intermediate(self):
        """Should be intermediate difficulty."""
        result = load_consumption()
        assert result.difficulty == "intermediate"


class TestLoadLaborSupply:
    """Tests for load_labor_supply function."""

    def test_returns_dataset_bundle(self):
        """Should return a DatasetBundle."""
        result = load_labor_supply()
        assert isinstance(result, DatasetBundle)

    def test_has_correct_name(self):
        """Should have correct name."""
        result = load_labor_supply()
        assert result.name == "labor_supply"

    def test_has_required_data(self):
        """Should have required data arrays."""
        result = load_labor_supply()
        assert 'log_hours' in result.data
        assert 'log_wage' in result.data
        assert 'age' in result.data
        assert 'education' in result.data
        assert 'experience' in result.data

    def test_data_shapes_match(self):
        """All data arrays should have same length."""
        result = load_labor_supply()
        n = result.n
        for key, arr in result.data.items():
            assert len(arr) == n, f"data['{key}'] has wrong length"

    def test_has_benchmark_params(self):
        """Should have benchmark parameters."""
        result = load_labor_supply()
        assert result.benchmark_params is not None
        assert 'gamma' in result.benchmark_params  # wage elasticity

    def test_difficulty_is_intermediate(self):
        """Should be intermediate difficulty."""
        result = load_labor_supply()
        assert result.difficulty == "intermediate"

    def test_has_instruments(self):
        """Should have husband's characteristics as instruments."""
        result = load_labor_supply()
        assert 'heducation' in result.data
        assert 'hage' in result.data

    def test_working_sample_positive(self):
        """Working sample should have positive hours and wages."""
        result = load_labor_supply()
        assert result.data['hours'].min() > 0
        assert result.data['wage'].min() > 0


class TestLoadAssetPricing:
    """Tests for load_asset_pricing function."""

    def test_returns_dataset_bundle(self):
        """Should return a DatasetBundle."""
        result = load_asset_pricing()
        assert isinstance(result, DatasetBundle)

    def test_has_correct_name(self):
        """Should have correct name."""
        result = load_asset_pricing()
        assert result.name == "asset_pricing"

    def test_has_required_data(self):
        """Should have required data arrays."""
        result = load_asset_pricing()
        assert 'consumption_growth' in result.data
        assert 'gross_return' in result.data
        assert 'gross_rf' in result.data

    def test_returns_positive(self):
        """Gross returns should be positive."""
        result = load_asset_pricing()
        assert result.data['gross_return'].min() > 0
        assert result.data['gross_rf'].min() > 0

    def test_has_benchmark_params(self):
        """Should have benchmark parameters."""
        result = load_asset_pricing()
        assert result.benchmark_params is not None
        assert 'beta' in result.benchmark_params
        assert 'gamma' in result.benchmark_params

    def test_difficulty_is_advanced(self):
        """Should be advanced difficulty."""
        result = load_asset_pricing()
        assert result.difficulty == "advanced"

    def test_data_shapes_match(self):
        """All data arrays should have same length."""
        result = load_asset_pricing()
        n = result.n
        for key, arr in result.data.items():
            assert len(arr) == n, f"data['{key}'] has wrong length"


class TestLoadDataset:
    """Tests for load_dataset function."""

    def test_loads_econ381(self):
        """Should load econ381 dataset."""
        result = load_dataset('econ381')
        assert result.name == 'econ381'

    def test_loads_consumption(self):
        """Should load consumption dataset."""
        result = load_dataset('consumption')
        assert result.name == 'consumption'

    def test_loads_labor_supply(self):
        """Should load labor_supply dataset."""
        result = load_dataset('labor_supply')
        assert result.name == 'labor_supply'

    def test_loads_asset_pricing(self):
        """Should load asset_pricing dataset."""
        result = load_dataset('asset_pricing')
        assert result.name == 'asset_pricing'

    def test_raises_on_unknown(self):
        """Should raise ValueError for unknown dataset."""
        with pytest.raises(ValueError, match="Unknown dataset"):
            load_dataset('nonexistent')


class TestDatasetBundle:
    """Tests for DatasetBundle dataclass."""

    def test_repr(self):
        """Should have readable repr."""
        result = load_consumption()
        repr_str = repr(result)
        assert 'consumption' in repr_str
        assert 'intermediate' in repr_str

    def test_info_runs(self, capsys):
        """info() should run without error."""
        result = load_consumption()
        result.info()
        captured = capsys.readouterr()
        assert 'consumption' in captured.out.lower()

    def test_true_params_alias(self):
        """true_params should be alias for benchmark_params."""
        result = load_consumption()
        assert result.true_params == result.benchmark_params


class TestDatasetCompleteness:
    """
    Property test: all datasets should have complete information.
    
    **Feature: momentest, Property 17: Dataset Completeness**
    **Validates: Requirements 10.4, 10.5**
    """

    @pytest.mark.parametrize("name", list_datasets())
    def test_dataset_has_all_fields(self, name):
        """
        Each dataset should have all required fields.
        
        Property 17: Dataset Completeness
        *For any* built-in dataset, the DatasetBundle SHALL contain non-empty
        data, description, economic_model, and moment_conditions fields.
        """
        dataset = load_dataset(name)
        
        # Required fields
        assert dataset.name is not None
        assert dataset.data is not None
        assert dataset.n > 0
        assert len(dataset.description) > 0
        assert len(dataset.moment_conditions) > 0
        assert len(dataset.exercises) > 0
        assert len(dataset.references) > 0
        assert dataset.difficulty in ['beginner', 'intermediate', 'advanced']

    @pytest.mark.parametrize("name", list_datasets())
    def test_dataset_data_is_dict(self, name):
        """Each dataset's data should be a dictionary of arrays."""
        dataset = load_dataset(name)
        assert isinstance(dataset.data, dict)
        for key, value in dataset.data.items():
            assert isinstance(value, np.ndarray), f"data['{key}'] should be numpy array"

    @pytest.mark.parametrize("name", list_datasets())
    def test_dataset_n_matches_data(self, name):
        """n should match the length of data arrays."""
        dataset = load_dataset(name)
        for key, arr in dataset.data.items():
            assert len(arr) == dataset.n, f"data['{key}'] length doesn't match n"
