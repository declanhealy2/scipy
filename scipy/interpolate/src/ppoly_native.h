#ifndef SCIPY_PPOLY_NATIVE_H
#define SCIPY_PPOLY_NATIVE_H
#include <math.h>
#include <stdint.h>

static inline void scipy_ppoly_evaluate(
    const double* c, const double* knots, const double* query,
    int64_t components, int64_t cells, int64_t orders, int64_t count,
    int derivative, int extrapolate, int64_t order_stride, int64_t cell_stride,
    int64_t output_stride, double* output) {
    const double sign = knots[cells] >= knots[0] ? 1.0 : -1.0;
    const double left = sign * knots[0], right = sign * knots[cells];
    const double inverse_spacing = cells / (right - left);
    int64_t cell = 0;
    double cell_left = left, cell_right = sign * knots[1];
    for (int64_t i = 0; i < count; ++i) {
        const double z = sign * query[i];
        if (isnan(z) || (!extrapolate && (z < left || z > right))) {
            for (int64_t j = 0; j < components; ++j) output[i * output_stride + j] = NAN;
            continue;
        }
        if (z < cell_left || z >= cell_right) {
            if (z >= right) cell = cells - 1;
            else if (z < left) cell = 0;
            else {
                int64_t low = z >= cell_right ? cell + 1 : 0;
                int64_t high = z < cell_left ? cell - 1 : cells - 1;
                const double position = (z - left) * inverse_spacing;
                cell = isfinite(position) && position >= 0 && position < cells ? (int64_t)position : 0;
                if (z < sign * knots[cell] || z >= sign * knots[cell + 1]) {
                    if (z < sign * knots[cell]) high = cell - 1;
                    else low = cell + 1;
                    while (low < high) {
                        const int64_t mid = low + (high - low) / 2;
                        if (z < sign * knots[mid]) high = mid - 1;
                        else if (z >= sign * knots[mid + 1]) low = mid + 1;
                        else { low = mid; break; }
                    }
                    cell = low;
                }
            }
            cell_left = sign * knots[cell];
            cell_right = sign * knots[cell + 1];
        }
        const double delta = query[i] - sign * cell_left;
        for (int64_t j = 0; j < components; ++j) {
            double value = 0.0;
            for (int64_t row = 0; row < orders - derivative; ++row) {
                double factor = 1.0;
                for (int k = 0; k < derivative; ++k) factor *= orders - row - 1 - k;
                const double coefficient = c[row * order_stride + cell * cell_stride + j] * factor;
                value = row == 0 ? coefficient : value * delta + coefficient;
            }
            output[i * output_stride + j] = value;
        }
    }
}
#endif
