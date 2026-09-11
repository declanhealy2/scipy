from contextlib import nullcontext
from importlib.util import find_spec
import numpy as np
from scipy.interpolate import PPoly

from .common import Benchmark


class PiecewisePolynomial(Benchmark):
    params = [
        ["numpy", "mlx"],
        [1000, 1000000],
        ["uniform", "irregular"],
        ["sorted", "shuffled"],
    ]
    param_names = ["backend", "points", "grid", "query_order"]

    def setup(self, backend, points, grid, query_order):
        if backend == "mlx":
            if (
                find_spec("mlx") is None
                or find_spec("scipy.interpolate._ppoly_mlx") is None
            ):
                raise NotImplementedError
            import mlx.core as xp

            scope = xp.stream(xp.cpu)
        else:
            xp, scope = np, nullcontext()
        self.xp, self.scope = xp, scope
        rng = np.random.default_rng(617)
        knots = np.arange(16385) / 1024 - 8
        if grid == "irregular":
            knots[1:-1] = np.sort(rng.uniform(-8, 8, knots.size - 2))
        query = rng.uniform(-8, 8, points)
        if query_order == "sorted":
            query.sort()
        with scope:
            self.c = xp.asarray(
                rng.uniform(-1, 1, (3, knots.size - 1)), dtype=xp.float64
            )
            self.x = xp.asarray(knots, dtype=xp.float64)
            self.query = xp.asarray(query, dtype=xp.float64)
            self.poly = PPoly(self.c, self.x)
            result = self.poly(self.query)
            if backend == "mlx":
                xp.eval(result)

    def time_evaluate(self, backend, points, grid, query_order):
        with self.scope:
            result = self.poly(self.query)
            if backend == "mlx":
                self.xp.eval(result)

    def peakmem_construct_and_evaluate(self, backend, points, grid, query_order):
        with self.scope:
            result = PPoly(self.c, self.x)(self.query)
            if backend == "mlx":
                self.xp.eval(result)
            return result

    def time_construct_and_evaluate(self, backend, points, grid, query_order):
        with self.scope:
            result = PPoly(self.c, self.x)(self.query)
            if backend == "mlx":
                self.xp.eval(result)
