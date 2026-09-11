import numpy as np
import pytest

from scipy.interpolate import uniform_grid_span, uniform_integral


@pytest.fixture(params=["numpy", "mlx"])
def xp(request):
    if request.param == "numpy":
        yield np
    else:
        mx = pytest.importorskip("mlx.core")
        with mx.stream(mx.cpu):
            yield mx


@pytest.mark.parametrize("linear", [False, True])
@pytest.mark.parametrize("first", [-5, 0, 2**30])
@pytest.mark.parametrize("derivative", [False, True])
def test_affine_table_integral(xp, linear, first, derivative):
    cells, tables, step = 8, 3, 0.25
    knots = (first * cells + np.arange(tables * cells)).reshape(tables, cells) * step
    values = np.concatenate((2 * knots + 3, 2 * (knots[:, -1:] + step) + 3), axis=1)
    prefixes = np.arange(cells)[None, :] * step * (2 * knots[:, :1] + 3)
    prefixes = (
        prefixes
        + np.arange(cells)[None, :]
        * (np.arange(cells)[None, :] - (not linear))
        * step**2
    )
    anchors = np.array([-2.0, 0.0, 5.0])
    local = np.array([0.0, 0.125, 0.25, 1.875])
    times = (first + np.arange(tables))[:, None] * cells * step + local
    actual = uniform_integral(
        *[xp.array(a, dtype=xp.float64) for a in (times, values, prefixes, anchors)],
        first=first,
        step=step,
        linear=linear,
        derivative=derivative,
    )
    left = np.floor(local / step) * step
    y0 = 2 * knots[:, :1] + 3
    if derivative:
        expected = y0 + 2 * (local if linear else left)
    elif linear:
        expected = anchors[:, None] + y0 * local + local**2
    else:
        expected = (
            anchors[:, None]
            + y0 * local
            + left * (left - step)
            + 2 * left * (local - left)
        )
    assert type(actual).__module__.split(".")[0] == xp.__name__.split(".")[0]
    np.testing.assert_array_equal(actual, expected)


def test_strides_and_empty_queries(xp):
    values = xp.array([[0.0, 2.0, 4.0, 6.0, 8.0]])
    prefixes = xp.array([[0.0, 0.5, 2.0, 4.5]])
    anchors = xp.array([0.0])
    times = xp.array([0.0, -1.0, 0.25, -1.0, 1.875, -1.0])[::2]
    result = uniform_integral(times, values, prefixes, anchors, step=0.5)
    np.testing.assert_array_equal(result, np.array([0.0, 0.125, 7.03125]))
    empty = uniform_integral(times[:0], values, prefixes, anchors, step=0.5)
    assert empty.shape == (0,)


def test_mlx_cpu_integral_consumed_on_gpu():
    mx = pytest.importorskip("mlx.core")
    with mx.stream(mx.cpu):
        t = mx.array([0.0, 0.25, 0.5], dtype=mx.float64)
        values = mx.array([[2.0, 2.0, 2.0]], dtype=mx.float64)
        prefixes = mx.array([[0.0, 1.0]], dtype=mx.float64)
        integral = uniform_integral(
            t, values, prefixes, mx.array([0.0], dtype=mx.float64), step=0.5
        )
        phase = integral.astype(mx.float32)
    with mx.stream(mx.gpu):
        signal = mx.cos(phase) + 1j * mx.sin(phase)
        mx.eval(signal)
    np.testing.assert_array_equal(integral, [0.0, 0.5, 1.0])
    assert isinstance(signal, mx.array)


def test_grid_span_uses_integer_layout_without_converting_arrays(xp):
    times = xp.array([-2.25, 3.125, 0.0], dtype=xp.float64)
    assert uniform_grid_span(times, 0.5) == (-5, 6)
    assert type(times).__module__.split(".")[0] == xp.__name__.split(".")[0]


def test_subnormal_step_preserves_signed_integrals(xp):
    step = np.nextafter(0.0, 1.0)
    t = xp.array([0.0, step, 2 * step], dtype=xp.float64)
    values = xp.array([[-1.0, -2.0, -3.0, 0.0]], dtype=xp.float64)
    prefixes = xp.array([[0.0, -step, -3 * step]], dtype=xp.float64)
    result = uniform_integral(
        t, values, prefixes, xp.array([0.0], dtype=xp.float64), step=step, linear=False
    )
    np.testing.assert_array_equal(result, [0.0, -step, -3 * step])


def test_grid_boundary_neighbors_have_the_same_selection_on_both_backends(xp):
    times = np.array([0.3, np.nextafter(0.3, 0.0), np.nextafter(0.3, np.inf)])
    assert uniform_grid_span(xp.array(times, dtype=xp.float64), 0.1) == (3, 3)


@pytest.mark.parametrize(
    "invalid", [np.nan, np.inf, -np.inf, float(2**63), -float(2**63)]
)
def test_grid_span_rejects_nonfinite_and_unrepresentable_indices(xp, invalid):
    with pytest.raises(ValueError, match="finite.*int64"):
        uniform_grid_span(xp.array([0.0, invalid], dtype=xp.float64), 1.0)


def test_parallel_integral_preserves_exact_cell_values(xp):
    t = xp.array(np.arange(262144) % 16, dtype=xp.float64) * 0.125
    values = xp.array([[2.0] * 17], dtype=xp.float64)
    prefixes = xp.array((np.arange(16) * 0.25)[None, :], dtype=xp.float64)
    actual = uniform_integral(
        t, values, prefixes, xp.array([0.0], dtype=xp.float64), step=0.125
    )
    np.testing.assert_array_equal(actual, (np.arange(262144) % 16) * 0.25)


def test_strided_tables_and_endpoint_extension(xp):
    values = xp.array([[2.0, -99.0, 4.0, -99.0, 6.0, -99.0]], dtype=xp.float64)[:, ::2]
    prefixes = xp.array([[0.0, -99.0, 1.5, -99.0]], dtype=xp.float64)[:, ::2]
    anchors = xp.array([1.0, -99.0], dtype=xp.float64)[::2]
    times = xp.array([-1.0, 0.0, 0.5, 1.0, 2.0], dtype=xp.float64)
    result = uniform_integral(times, values, prefixes, anchors, step=0.5)
    np.testing.assert_array_equal(result, [1.0, 1.0, 2.5, 5.0, 5.0])


@pytest.mark.parametrize("step", [0.0, -1.0, np.inf, np.nan])
def test_invalid_table_spacing_is_rejected(xp, step):
    values, prefixes, anchors = [
        xp.array(a, dtype=xp.float64) for a in ([[1.0, 2.0]], [[0.0]], [0.0])
    ]
    with pytest.raises(ValueError, match="positive finite step"):
        uniform_integral(
            xp.array([0.0], dtype=xp.float64), values, prefixes, anchors, step=step
        )


@pytest.mark.parametrize("first,step", [(2**63 - 1, 1.0), (-(2**63), 1.0), (2, 1e308)])
def test_unrepresentable_table_grids_are_rejected(xp, first, step):
    values, prefixes, anchors = [
        xp.array(a, dtype=xp.float64) for a in ([[1.0, 2.0]], [[0.0]], [0.0])
    ]
    with pytest.raises(ValueError, match="finite grid endpoints"):
        uniform_integral(
            xp.array([0.0], dtype=xp.float64),
            values,
            prefixes,
            anchors,
            first=first,
            step=step,
        )
