#pragma once
#include <cmath>
#include <cstdint>

inline bool valid_uniform_grid(int64_t first, int64_t cells, int64_t tables, double step) {
    const double width = static_cast<double>(cells) * step;
    const double start = static_cast<double>(first);
    return step > 0 && std::isfinite(width) &&
           std::abs(start) + static_cast<double>(tables) <= 9007199254740992.0 &&
           std::isfinite(start * width) &&
           std::isfinite((start + static_cast<double>(tables)) * width);
}

extern "C" void scipy_uniform_integral(
    const double*, const double*, const double*, const double*,
    int64_t, int64_t, int64_t, int64_t, double, int, int, double*);

extern "C" int scipy_uniform_grid_span(const double*, int64_t, double, int64_t*, int64_t*);
