from contextlib import nullcontext
from fractions import Fraction
import math
from unittest.mock import patch

import numpy as np
import pytest

from scipy.interpolate import PPoly
from scipy._external.array_api_compat import array_namespace


@pytest.fixture(params=["numpy", "mlx"])
def native_xp(request):
    if request.param == "numpy":
        xp, scope = array_namespace(np.empty(0)), nullcontext()
    else:
        mx = pytest.importorskip("mlx.core")
        pytest.importorskip("scipy.interpolate._ppoly_mlx")
        xp, scope = array_namespace(mx.array(0)), mx.stream(mx.cpu)
    with scope:
        yield xp


@pytest.mark.parametrize("descending", [False, True])
@pytest.mark.parametrize("axis", [0, 1])
def test_cubic_polynomial_calculus(native_xp, descending, axis):
    xp = native_xp
    knots = np.array([-4.0, -1.0, 0.0, 2.0, 8.0])
    if descending:
        knots = knots[::-1].copy()
    origin = knots[:-1]
    coefficients = np.stack(
        (np.ones_like(origin), 3 * origin, 3 * origin**2, origin**3)
    )
    coefficients = np.stack((coefficients, 2 * coefficients), axis=-1)
    if axis:
        coefficients = np.moveaxis(coefficients, -1, 0)
    poly = PPoly(
        xp.asarray(coefficients, dtype=xp.float64),
        xp.asarray(knots, dtype=xp.float64),
        axis=axis,
    )
    query = np.array([-3.5, -0.5, 0.0, 0.5, 1.0, 4.0, 8.0])
    t = xp.asarray(query, dtype=xp.float64)
    for derivative in range(5):
        expected = (
            query ** (3 - derivative) * math.prod(range(4 - derivative, 4))
            if derivative <= 3
            else np.zeros_like(query)
        )
        expected = np.stack((expected, 2 * expected), axis=0 if axis else -1)
        actual = poly(t, nu=derivative)
        np.testing.assert_array_equal(actual, expected)
        assert array_namespace(actual) is xp
    expected = np.stack((3 * query**2, 6 * query**2), axis=0 if axis else -1)
    np.testing.assert_array_equal(poly.derivative()(t), expected)
    expected = (query**4 - knots[0] ** 4) / 4
    expected = np.stack((expected, 2 * expected), axis=0 if axis else -1)
    np.testing.assert_array_equal(poly.antiderivative()(t), expected)
    integral = poly.integrate(0.0, 2.0)
    term_magnitude = 4 + 16 + 24 + 16
    bound = 16 * np.finfo(float).eps * term_magnitude * np.array([1.0, 2.0])
    assert np.all(np.abs(np.asarray(integral) - [4.0, 8.0]) <= bound)


@pytest.mark.parametrize("periodic", [False, True])
def test_interval_boundaries_and_extrapolation(native_xp, periodic):
    xp = native_xp
    c = xp.asarray([[2.0, 5.0, 7.0]], dtype=xp.float64)
    knots = xp.asarray([0.0, 1.0, 1.0, 2.0], dtype=xp.float64)
    poly = PPoly(c, knots, extrapolate="periodic" if periodic else False)
    query = xp.asarray(
        [-1.0, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, float("nan")], dtype=xp.float64
    )
    expected = (
        [7.0, 2.0, 2.0, 7.0, 7.0, 2.0, 7.0, np.nan]
        if periodic
        else [np.nan, 2.0, 2.0, 7.0, 7.0, 7.0, np.nan, np.nan]
    )
    np.testing.assert_array_equal(poly(query), expected)
    np.testing.assert_array_equal(
        poly(xp.empty((0, 3), dtype=xp.float64)), np.empty((0, 3))
    )


@pytest.mark.parametrize("descending", [False, True])
def test_irregular_interval_search_with_discontinuous_values(native_xp, descending):
    xp = native_xp
    rng = np.random.default_rng(831)
    knots = np.cumsum(rng.integers(1, 1024, 257)) / 1024
    query = np.concatenate(
        (knots, (knots[:-1] + knots[1:]) / 2, [knots[0] - 1, knots[-1] + 1])
    )
    query = rng.permutation(query)
    sign = -1 if descending else 1
    knots = knots[::sign].copy()
    coefficients = np.arange(knots.size - 1, dtype=float)[None, :]
    expected = np.clip(
        np.sum(sign * query[:, None] >= sign * knots, axis=1) - 1,
        0,
        knots.size - 2,
    )
    poly = PPoly(xp.asarray(coefficients), xp.asarray(knots))
    np.testing.assert_array_equal(poly(xp.asarray(query)), expected)


@pytest.mark.parametrize("dtype", ["float32", "int32", "float64"])
def test_strided_inputs_and_declared_conversion(native_xp, dtype):
    xp = native_xp
    coefficients = xp.asarray([[3, 0, 3, 0], [2, 0, 5, 0]], dtype=getattr(xp, dtype))[
        :, ::2
    ]
    knots = xp.asarray([0, 9, 1, 9, 2, 9], dtype=getattr(xp, dtype))[::2]
    query = xp.asarray(
        [0.0, 0.0, 0.5, 0.0, 1.0, 0.0, 1.5, 0.0, 2.0, 0.0], dtype=xp.float64
    )[::2]
    poly = PPoly(coefficients, knots)
    np.testing.assert_array_equal(poly(query), [2.0, 3.5, 5.0, 6.5, 8.0])


def test_mlx_gpu_producer_cpu_polynomial_gpu_consumer():
    mx = pytest.importorskip("mlx.core")
    pytest.importorskip("scipy.interpolate._ppoly_mlx")
    with mx.stream(mx.gpu):
        coefficients = mx.array([[1.5], [1.0]]) * 2
        query = mx.arange(32, dtype=mx.float32) / 32
    with patch.object(np, "asarray", side_effect=AssertionError("host conversion")):
        with mx.stream(mx.cpu):
            knots = mx.array([0.0, 1.0], dtype=mx.float64)
            result = PPoly(coefficients, knots)(query)
            result = result.astype(mx.float32)
        with mx.stream(mx.gpu):
            result = result * 3
            mx.eval(result)
    np.testing.assert_array_equal(result, 9 * np.arange(32) / 32 + 6)


@pytest.mark.parametrize("integrations", [1, 2, 3, 25])
def test_multiple_integrals_have_zero_initial_conditions(native_xp, integrations):
    xp = native_xp
    knots = np.array([-2.0, -0.5, 1.0, 4.0])
    degree = 3
    coefficients = np.array(
        [
            [math.comb(degree, row) * origin**row for origin in knots[:-1]]
            for row in range(degree + 1)
        ]
    )
    poly = PPoly(xp.asarray(coefficients), xp.asarray(knots))
    integral = poly.antiderivative(integrations)
    query = [Fraction(-2), Fraction(-1), Fraction(0), Fraction(2), Fraction(4)]
    expected, magnitudes = [], []
    for t in query:
        terms = [
            Fraction(math.factorial(degree), math.factorial(degree + integrations))
            * t ** (degree + integrations)
        ]
        for k in range(integrations):
            terms.append(
                -Fraction(
                    math.factorial(degree),
                    math.factorial(degree + integrations - k) * math.factorial(k),
                )
                * Fraction(-2) ** (degree + integrations - k)
                * (t + 2) ** k
            )
        expected.append(float(sum(terms)))
        magnitudes.append(float(sum(abs(term) for term in terms)))
    actual = integral(xp.asarray([float(t) for t in query], dtype=xp.float64))
    operations = 2 * (degree + integrations + 1) ** 2 * (len(knots) - 1)
    gamma = operations * np.finfo(float).eps / (1 - operations * np.finfo(float).eps)
    np.testing.assert_array_less(
        np.abs(np.asarray(actual) - expected), gamma * np.asarray(magnitudes)
    )
    np.testing.assert_array_equal(
        integral(xp.asarray([-2.0]), nu=integrations - 1), [0.0]
    )


def test_subnormal_knot_spacing():
    tiny = np.nextafter(0.0, 1.0)
    poly = PPoly([[4.0, 8.0]], [0.0, tiny, 2 * tiny])
    np.testing.assert_array_equal(poly([0.0, tiny, 2 * tiny]), [4.0, 8.0, 8.0])


def test_mlx_spline_conversion_rejects_without_host_conversion():
    mx = pytest.importorskip("mlx.core")
    with (
        mx.stream(mx.cpu),
        patch.object(np, "asarray", side_effect=AssertionError("host conversion")),
    ):
        with pytest.raises(NotImplementedError, match="from_spline"):
            PPoly.from_spline((mx.array([0.0, 0.0, 1.0, 1.0]), mx.array([1.0, 2.0]), 1))
