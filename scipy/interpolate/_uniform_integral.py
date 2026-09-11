from scipy._external.array_api_compat import array_namespace, is_numpy_namespace


def uniform_grid_span(t, step):
    """Return the first and last integer cells intersected by finite queries.

    The integer bounds describe table layout. Query arrays retain their array
    namespace and storage; MLX synchronizes their CPU producer before returning
    these structural bounds. The query array must be nonempty.
    """
    xp = array_namespace(t)
    if xp.__name__ == "mlx.core" or xp.__name__.endswith(".mlx"):
        from ._uniform_integral_mlx import grid_span
    elif is_numpy_namespace(xp):
        from ._uniform_integral_numpy import grid_span
    else:
        raise NotImplementedError("uniform_grid_span supports NumPy and MLX arrays")
    return grid_span(t, step)


def uniform_integral(
    t, values, prefixes, anchors, *, first=0, step=1.0, linear=True, derivative=False
):
    """Evaluate a piecewise-uniform function's anchored integral in float64.

    ``values`` contains tables of shape ``(m, n + 1)``. Table ``j`` begins
    at ``(first + j) * n * step``. Its last value is the right endpoint.
    ``prefixes[j, k]`` is the integral within the table up to knot ``k``;
    ``anchors[j]`` is the integral at its first knot. Linear interpolation
    integrates each segment exactly; ``linear=False`` uses left-constant
    segments. ``derivative=True`` returns the interpolated function instead.

    The result has the shape and array namespace of ``t``. NumPy and MLX
    arrays are supported, including strided inputs. MLX evaluates on its CPU
    stream and retains its own storage and graph dependencies. Queries outside
    the table support are clamped to its endpoints. Nonfinite queries return
    NaN. Grid knots are selected with a bound derived from the floating-point
    spacing of their reconstruction. Table indices must lie in float64's
    consecutive integer range, and the grid endpoints must be finite.
    """
    xp = array_namespace(t, values, prefixes, anchors)
    if xp.__name__ == "mlx.core" or xp.__name__.endswith(".mlx"):
        from ._uniform_integral_mlx import evaluate
    elif is_numpy_namespace(xp):
        from ._uniform_integral_numpy import evaluate
    else:
        raise NotImplementedError("uniform_integral supports NumPy and MLX arrays")
    return evaluate(t, values, prefixes, anchors, first, step, linear, derivative)
