"""Tests for the DGP (Data Generating Process) module."""

import numpy as np
import pytest
from hypothesis import given, strategies as st, settings

from momentest import (
    DGPResult,
    list_dgps,
    linear_iv,
    consumption_savings,
    dynamic_discrete_choice,
    load_dgp,
)


class TestListDGPs:
    """Tests for list_dgps function."""

    def test_returns_list(self):
        """Should return a list."""
        result = list_dgps()
        assert isinstance(result, list)

    def test_contains_all_dgps(self):
        """Should contain all expected DGPs."""
        result = list_dgps()
        expected = ['linear_iv', 'consumption_savings', 'dynamic_discrete_choice']
        for name in expected:
            assert name in result, f"Missing DGP: {name}"


class TestLinearIV:
    """Tests for linear_iv DGP."""

    def test_returns_dgp_result(self):
        """Should return a DGPResult."""
        result = linear_iv(n=100, seed=42)
        assert isinstance(result, DGPResult)

    def test_has_correct_name(self):
        """Should have correct name."""
        result = linear_iv(n=100, seed=42)
        assert result.name == "linear_iv"

    def test_has_required_data(self):
        """Should have required data arrays."""
        result = linear_iv(n=100, seed=42)
        assert 'Y' in result.data
        assert 'X' in result.data
        assert 'Z' in result.data

    def test_data_shapes_match(self):
        """All data arrays should have same length."""
        result = linear_iv(n=100, seed=42)
        n = result.n
        for key, arr in result.data.items():
            assert len(arr) == n, f"data['{key}'] has wrong length"

    def test_true_theta_shape(self):
        """true_theta should have correct shape."""
        result = linear_iv(n=100, seed=42)
        assert result.true_theta.shape == (result.p,)
        assert len(result.param_names) == result.p

    def test_moment_function_shape(self):
        """Moment function should return correct shape."""
        result = linear_iv(n=100, seed=42)
        moments = result.moment_function(result.data, result.true_theta)
        assert moments.shape == (result.n, result.k)

    def test_custom_parameters(self):
        """Should accept custom parameters."""
        result = linear_iv(n=100, seed=42, beta0=5.0, beta1=3.0)
        assert result.true_theta[0] == 5.0
        assert result.true_theta[1] == 3.0

    def test_difficulty_is_beginner(self):
        """Should be beginner difficulty."""
        result = linear_iv(n=100, seed=42)
        assert result.difficulty == "beginner"


class TestConsumptionSavings:
    """Tests for consumption_savings DGP."""

    def test_returns_dgp_result(self):
        """Should return a DGPResult."""
        result = consumption_savings(n=100, seed=42)
        assert isinstance(result, DGPResult)

    def test_has_correct_name(self):
        """Should have correct name."""
        result = consumption_savings(n=100, seed=42)
        assert result.name == "consumption_savings"

    def test_has_required_data(self):
        """Should have required data arrays."""
        result = consumption_savings(n=100, seed=42)
        assert 'C1' in result.data
        assert 'C2' in result.data
        assert 'Y1' in result.data
        assert 'Y2' in result.data

    def test_data_shapes_match(self):
        """All data arrays should have same length."""
        result = consumption_savings(n=100, seed=42)
        n = result.n
        for key, arr in result.data.items():
            assert len(arr) == n, f"data['{key}'] has wrong length"

    def test_consumption_positive(self):
        """Consumption should be positive."""
        result = consumption_savings(n=100, seed=42)
        assert result.data['C1'].min() > 0
        assert result.data['C2'].min() > 0

    def test_moment_function_shape(self):
        """Moment function should return correct shape."""
        result = consumption_savings(n=100, seed=42)
        moments = result.moment_function(result.data, result.true_theta)
        assert moments.shape == (result.n, result.k)

    def test_difficulty_is_intermediate(self):
        """Should be intermediate difficulty."""
        result = consumption_savings(n=100, seed=42)
        assert result.difficulty == "intermediate"


class TestDynamicDiscreteChoice:
    """Tests for dynamic_discrete_choice DGP."""

    def test_returns_dgp_result(self):
        """Should return a DGPResult."""
        result = dynamic_discrete_choice(n=50, T=10, seed=42)
        assert isinstance(result, DGPResult)

    def test_has_correct_name(self):
        """Should have correct name."""
        result = dynamic_discrete_choice(n=50, T=10, seed=42)
        assert result.name == "dynamic_discrete_choice"

    def test_has_required_data(self):
        """Should have required data arrays."""
        result = dynamic_discrete_choice(n=50, T=10, seed=42)
        assert 'state' in result.data
        assert 'action' in result.data
        assert 'agent_id' in result.data
        assert 'time' in result.data

    def test_data_length(self):
        """Data length should be n * T."""
        n, T = 50, 10
        result = dynamic_discrete_choice(n=n, T=T, seed=42)
        assert result.n == n * T

    def test_actions_binary(self):
        """Actions should be binary (0 or 1)."""
        result = dynamic_discrete_choice(n=50, T=10, seed=42)
        actions = result.data['action']
        assert set(np.unique(actions)).issubset({0, 1})

    def test_states_bounded(self):
        """States should be bounded."""
        result = dynamic_discrete_choice(n=50, T=10, seed=42)
        states = result.data['state']
        assert states.min() >= 0
        assert states.max() <= 10  # X_max

    def test_moment_function_shape(self):
        """Moment function should return correct shape."""
        result = dynamic_discrete_choice(n=50, T=10, seed=42)
        moments = result.moment_function(result.data, result.true_theta)
        assert moments.shape == (result.n, result.k)

    def test_difficulty_is_advanced(self):
        """Should be advanced difficulty."""
        result = dynamic_discrete_choice(n=50, T=10, seed=42)
        assert result.difficulty == "advanced"


class TestLoadDGP:
    """Tests for load_dgp function."""

    def test_loads_linear_iv(self):
        """Should load linear_iv DGP."""
        result = load_dgp('linear_iv', n=100, seed=42)
        assert result.name == 'linear_iv'

    def test_loads_consumption_savings(self):
        """Should load consumption_savings DGP."""
        result = load_dgp('consumption_savings', n=100, seed=42)
        assert result.name == 'consumption_savings'

    def test_loads_dynamic_discrete_choice(self):
        """Should load dynamic_discrete_choice DGP."""
        result = load_dgp('dynamic_discrete_choice', n=50, T=10, seed=42)
        assert result.name == 'dynamic_discrete_choice'

    def test_raises_on_unknown(self):
        """Should raise ValueError for unknown DGP."""
        with pytest.raises(ValueError, match="Unknown DGP"):
            load_dgp('nonexistent')


class TestDGPResult:
    """Tests for DGPResult dataclass."""

    def test_repr(self):
        """Should have readable repr."""
        result = linear_iv(n=100, seed=42)
        repr_str = repr(result)
        assert 'linear_iv' in repr_str
        assert 'beginner' in repr_str

    def test_info_runs(self, capsys):
        """info() should run without error."""
        result = linear_iv(n=100, seed=42)
        result.info()
        captured = capsys.readouterr()
        assert 'linear_iv' in captured.out.lower()


class TestDGPReproducibility:
    """
    Property test: DGPs should be reproducible with same seed.
    
    **Feature: momentest, Property 18: DGP Reproducibility**
    **Validates: Requirements 11.5**
    """

    @given(seed=st.integers(min_value=0, max_value=2**31 - 1))
    @settings(max_examples=100)
    def test_linear_iv_reproducibility(self, seed):
        """
        Property 18: DGP Reproducibility
        
        *For any* DGP function called twice with the same seed,
        the returned data and true_theta SHALL be identical.
        """
        dgp1 = linear_iv(n=100, seed=seed)
        dgp2 = linear_iv(n=100, seed=seed)
        
        # true_theta should be identical
        np.testing.assert_array_equal(dgp1.true_theta, dgp2.true_theta)
        
        # All data arrays should be identical
        for key in dgp1.data.keys():
            np.testing.assert_array_equal(
                dgp1.data[key], dgp2.data[key],
                err_msg=f"data['{key}'] differs for seed={seed}"
            )

    @given(seed=st.integers(min_value=0, max_value=2**31 - 1))
    @settings(max_examples=100)
    def test_consumption_savings_reproducibility(self, seed):
        """
        Property 18: DGP Reproducibility for consumption_savings.
        """
        dgp1 = consumption_savings(n=100, seed=seed)
        dgp2 = consumption_savings(n=100, seed=seed)
        
        np.testing.assert_array_equal(dgp1.true_theta, dgp2.true_theta)
        
        for key in dgp1.data.keys():
            np.testing.assert_array_equal(
                dgp1.data[key], dgp2.data[key],
                err_msg=f"data['{key}'] differs for seed={seed}"
            )

    @given(seed=st.integers(min_value=0, max_value=2**31 - 1))
    @settings(max_examples=100)
    def test_dynamic_discrete_choice_reproducibility(self, seed):
        """
        Property 18: DGP Reproducibility for dynamic_discrete_choice.
        """
        dgp1 = dynamic_discrete_choice(n=50, T=10, seed=seed)
        dgp2 = dynamic_discrete_choice(n=50, T=10, seed=seed)
        
        np.testing.assert_array_equal(dgp1.true_theta, dgp2.true_theta)
        
        for key in dgp1.data.keys():
            np.testing.assert_array_equal(
                dgp1.data[key], dgp2.data[key],
                err_msg=f"data['{key}'] differs for seed={seed}"
            )

    @pytest.mark.parametrize("dgp_name", list_dgps())
    def test_all_dgps_reproducible(self, dgp_name):
        """
        All DGPs should be reproducible with same seed.
        
        Property 18: DGP Reproducibility
        """
        seed = 12345
        
        if dgp_name == 'dynamic_discrete_choice':
            dgp1 = load_dgp(dgp_name, n=50, T=10, seed=seed)
            dgp2 = load_dgp(dgp_name, n=50, T=10, seed=seed)
        else:
            dgp1 = load_dgp(dgp_name, n=100, seed=seed)
            dgp2 = load_dgp(dgp_name, n=100, seed=seed)
        
        np.testing.assert_array_equal(dgp1.true_theta, dgp2.true_theta)
        
        for key in dgp1.data.keys():
            np.testing.assert_array_equal(dgp1.data[key], dgp2.data[key])


class TestDGPCompleteness:
    """Tests that all DGPs have complete information."""

    @pytest.mark.parametrize("dgp_name", list_dgps())
    def test_dgp_has_all_fields(self, dgp_name):
        """Each DGP should have all required fields."""
        if dgp_name == 'dynamic_discrete_choice':
            dgp = load_dgp(dgp_name, n=50, T=10, seed=42)
        else:
            dgp = load_dgp(dgp_name, n=100, seed=42)
        
        # Required fields
        assert dgp.name is not None
        assert dgp.data is not None
        assert dgp.n > 0
        assert dgp.true_theta is not None
        assert len(dgp.param_names) == dgp.p
        assert dgp.moment_function is not None
        assert dgp.k > 0
        assert dgp.p > 0
        assert len(dgp.description) > 0
        assert dgp.difficulty in ['beginner', 'intermediate', 'advanced']

    @pytest.mark.parametrize("dgp_name", list_dgps())
    def test_dgp_moment_function_works(self, dgp_name):
        """Moment function should work at true parameters."""
        if dgp_name == 'dynamic_discrete_choice':
            dgp = load_dgp(dgp_name, n=50, T=10, seed=42)
        else:
            dgp = load_dgp(dgp_name, n=100, seed=42)
        
        moments = dgp.moment_function(dgp.data, dgp.true_theta)
        
        assert moments.shape == (dgp.n, dgp.k)
        assert np.all(np.isfinite(moments))
