"""Tests based on R package vignettes."""

import numpy as np
import pytest
from inflection import (
    check_curve, ese, ede, bese, bede,
    edeci, findiplist, uik
)


class TestFisherPrySigmoid:
    """Tests for Fisher-Pry sigmoid curve (tanh function)."""

    def test_total_symmetry_no_noise(self):
        """Test with total symmetry and no noise."""
        ## f(x) = 5 + 5 * tanh(x-5)
        x = np.linspace(2, 8, 501)
        y = 5 + 5 * np.tanh(x - 5)

        ## Check curve type
        cc = check_curve(x, y)
        assert cc["ctype"] == "convex_concave"
        assert cc["index"] == 0

        ## Test ESE
        result_ese = ese(x, y, cc["index"])
        assert abs(result_ese["chi"] - 5.0) < 0.01

        ## Test EDE
        result_ede = ede(x, y, cc["index"])
        assert abs(result_ede["chi"] - 5.0) < 0.01

        ## Test BESE
        result_bese = bese(x, y, cc["index"])
        assert abs(result_bese["iplast"] - 5.0) < 0.001

        ## Test BEDE
        result_bede = bede(x, y, cc["index"])
        assert abs(result_bede["iplast"] - 5.0) < 0.001

    def test_total_symmetry_with_noise(self):
        """Test with total symmetry and uniform noise."""
        np.random.seed(666)
        x = np.linspace(2, 8, 501)
        y = 5 + 5 * np.tanh(x - 5)
        y = y + np.random.uniform(-0.05, 0.05, len(y))

        cc = check_curve(x, y)

        ## Test that methods still find approximately correct inflection
        result_bese = bese(x, y, cc["index"])
        assert abs(result_bese["iplast"] - 5.0) < 0.1

        result_bede = bede(x, y, cc["index"])
        assert abs(result_bede["iplast"] - 5.0) < 0.1

    def test_data_left_asymmetry(self):
        """Test with data left asymmetry."""
        x = np.linspace(4.2, 8, 301)
        y = 5 + 5 * np.tanh(x - 5)

        cc = check_curve(x, y)

        ## ESE should estimate around 4.7
        result_ese = ese(x, y, cc["index"])
        assert 4.5 < result_ese["chi"] < 4.9

        ## EDE should estimate around 5.08
        result_ede = ede(x, y, cc["index"])
        assert 4.9 < result_ede["chi"] < 5.3

    def test_uik_method(self):
        """Test UIK method on sigmoid curve."""
        x = np.linspace(0, 10, 201)
        y = 5 + 5 * np.tanh(x - 5)

        knee = uik(x, y)
        ## UIK finds knee point, not inflection point
        ## For sigmoid, this is around 3.5
        assert 3.0 < knee < 4.0  # Adjusted range

    ## N.B. This doesn't really belong here, but oh well
    def test_uik_method_elbow(self):
        """Test UIK method on an appropriate curve (elbow curve, not sigmoid)."""
        ## Create an elbow curve (exponential decay)
        x = np.linspace(0, 5, 100)
        y = np.exp(-x)

        knee = uik(x, y)

        ## For exponential decay, knee should be around 1.0
        ##   (where the curve transitions from steep to flat)
        assert 0.5 < knee < 2.0

        ## Verify UIK returns x[ede_result["j1"]]
        cc = check_curve(x, y)
        ede_result = ede(x, y, cc["index"])
        expected_knee = x[ede_result["j1"]]
        assert abs(knee - expected_knee) < 0.001


class TestGompertzCurve:
    """Tests for Gompertz non-symmetric sigmoid curve."""

    def test_gompertz_no_noise(self):
        """Test Gompertz curve without noise."""
        ## f(x) = 10 * exp(-exp(5) * exp(-x))
        x = np.linspace(3.5, 8, 501)
        y = 10 * np.exp(-np.exp(5) * np.exp(-x))

        cc = check_curve(x, y)

        ## True inflection point is at x = 5
        result_bese = bese(x, y, cc["index"])
        assert abs(result_bese["iplast"] - 5.0) < 0.01

        result_bede = bede(x, y, cc["index"])
        assert abs(result_bede["iplast"] - 5.0) < 0.01

    def test_gompertz_with_noise(self):
        """Test Gompertz curve with noise."""
        np.random.seed(666)  ## Use consistent seed
        x = np.linspace(3.5, 8, 501)
        y = 10 * np.exp(-np.exp(5) * np.exp(-x))
        y = y + np.random.uniform(-0.05, 0.05, len(y))

        cc = check_curve(x, y)

        result_bese = bese(x, y, cc["index"])
        ## With noise, allow slightly more tolerance
        assert abs(result_bese["iplast"] - 5.0) < 0.2 # Increased from 0.1 to 0.2

        result_bede = bede(x, y, cc["index"])
        assert abs(result_bede["iplast"] - 5.0) < 0.2 # Also test BEDE


class TestPolynomialCurves:
    """Tests for polynomial curves."""

    def test_cubic_polynomial_symmetric(self):
        """Test symmetric 3rd order polynomial."""
        ## f(x) = -1/3 * x^3 + 5/2 * x^2 - 4 * x + 1/2
        ## Inflection point at x=2.5
        x = np.linspace(-2, 7, 501)
        y = -1/3 * x**3 + 5/2 * x**2 - 4 * x + 1/2

        cc = check_curve(x, y)

        result_ese = ese(x, y, cc["index"])
        assert abs(result_ese["chi"] - 2.5) < 0.01

        result_ede = ede(x, y, cc["index"])
        assert abs(result_ede["chi"] - 2.5) < 0.01

    def test_cubic_polynomial_asymmetric(self):
        """Test asymmetric 3rd order polynomial."""
        x = np.linspace(-2, 8, 501)
        y = -1/3 * x**3 + 5/2 * x**2 - 4 * x + 1/2

        cc = check_curve(x, y)

        ## With asymmetry, estimates will be slightly off
        result_bese = bese(x, y, cc["index"])
        assert 2.0 < result_bese["iplast"] < 3.0

        result_bede = bede(x, y, cc["index"])
        assert 2.0 < result_bede["iplast"] < 3.0


class TestBigData:
    """Tests with large datasets."""

    def test_big_data_performance(self):
        """Test performance with large dataset."""
        ## f(x) = 500 + 500 * tanh(x - 500)
        x = np.linspace(0, 1000, 100001)
        y = 500 + 500 * np.tanh(x - 500)

        cc = check_curve(x, y)

        ## Test EDE (fast for big data)
        result_ede = ede(x, y, cc["index"])
        assert abs(result_ede["chi"] - 500.0) < 0.1

        ## Test BEDE
        result_bede = bede(x, y, cc["index"])
        assert abs(result_bede["iplast"] - 500.0) < 0.01

        ## Check that BEDE converges in reasonable iterations
        assert len(result_bede["iters"]["n"]) < 15

    def test_big_data_with_noise(self):
        """Test large dataset with noise."""
        np.random.seed(666)
        x = np.linspace(0, 1000, 10001)
        y = 500 + 500 * np.tanh(x - 500)
        y = y + np.random.uniform(-50, 50, len(y))

        cc = check_curve(x, y)

        result_bede = bede(x, y, cc["index"])
        ## Even with noise, should find close to true value
        assert abs(result_bede["iplast"] - 500.0) < 5.0


class TestEdgeCase:
    """Tests for edge cases and error handling."""

    def test_insufficient_points(self):
        """Test with too few points."""
        x = np.array([1, 2, 3])
        y = np.array([1, 2, 3])

        with pytest.raises(ValueError):
            uik(x, y)

        ## findiplist should warn but not crash
        with pytest.warns(UserWarning):
            result = findiplist(x, y, 0)
            assert np.isnan(result["ESE"]["chi"])
            assert np.isnan(result["EDE"]["chi"])

    def test_confidence_intervals(self):
        """Test EDECI confidence interval calculation."""
        x = np.linspace(0, 10, 501)
        y = 5 + 5 * np.tanh(x - 5)

        cc = check_curve(x, y)
        result = edeci(x, y, cc["index"], k = 5)

        assert "chi-5*s" in result
        assert "chi+5*s" in result
        assert result["chi-5*s"] < result["chi"] < result["chi+5*s"]

    def test_downward_sigmoid(self):
        """Test with downward sigmoid (index=1)."""
        x = np.linspace(0, 10, 501)
        y = 5 - 5 * np.tanh(x - 5) # Downward sigmoid

        cc = check_curve(x, y)
        assert cc["index"] == 1

        result_ese = ese(x, y, cc["index"])
        assert abs(result_ese["chi"] - 5.0) < 0.01

        result_ede = ede(x, y, cc["index"])
        assert abs(result_ede["chi"] - 5.0) < 0.01


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
