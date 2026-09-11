from contextlib import nullcontext

import numpy as np

from .common import Benchmark, safe_import

with safe_import():
    from scipy.interpolate import uniform_integral


class UniformIntegral(Benchmark):
    param_names = ("namespace", "count")
    params = (("numpy", "mlx"), (1000, 1000000))

    def setup(self, namespace, count):
        if namespace == "mlx":
            try:
                import mlx.core as xp
            except ImportError:
                raise NotImplementedError("MLX is unavailable")
            scope, self.evaluate = xp.stream(xp.cpu), xp.eval
        else:
            xp = np
            scope, self.evaluate = nullcontext(), np.asarray
        self.xp = xp
        self.step, self.cells, self.tables = 0.125, 1024, 64
        with scope:
            self.t = xp.array(
                np.random.default_rng(42).uniform(0, 8192, count), dtype=xp.float64
            )
            self.values = xp.array(np.ones((64, 1025)), dtype=xp.float64)
            self.prefixes = xp.array(
                np.broadcast_to(np.arange(1024) * self.step, (64, 1024)),
                dtype=xp.float64,
            )
            self.anchors = xp.array(np.arange(64) * 128, dtype=xp.float64)

    def time_uniform_integral(self, namespace, count):
        self.evaluate(
            uniform_integral(
                self.t, self.values, self.prefixes, self.anchors, step=self.step
            )
        )

    def time_array_expression(self, namespace, count):
        xp = self.xp
        scope = xp.stream(xp.cpu) if namespace == "mlx" else nullcontext()
        with scope:
            table = xp.floor(self.t / (self.cells * self.step)).astype(xp.int64)
            local = (self.t - table * (self.cells * self.step)) / self.step
            cell = xp.floor(local).astype(xp.int64)
            fraction = local - cell
            y = self.values[table, cell]
            delta = self.values[table, cell + 1] - y
            result = self.anchors[table] + self.prefixes[table, cell]
            result = result + self.step * fraction * (y + 0.5 * fraction * delta)
            self.evaluate(result)

    peakmem_uniform_integral = time_uniform_integral
    peakmem_array_expression = time_array_expression
