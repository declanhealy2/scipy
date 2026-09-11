#include <cmath>
#include <nanobind/nanobind.h>
#include "mlx/ops.h"
#include "mlx/transforms.h"
#include "mlx/primitives.h"
#include "mlx/allocator.h"
#include "mlx/backend/cpu/encoder.h"
#include "uniform_integral.h"

namespace mx = mlx::core;
namespace nb = nanobind;

class UniformIntegral : public mx::Primitive {
    int64_t first_;
    double step_;
    bool linear_, derivative_;
public:
    UniformIntegral(mx::Stream stream, int64_t first, double step, bool linear, bool derivative)
        : mx::Primitive(stream), first_(first), step_(step), linear_(linear), derivative_(derivative) {}
    const char* name() const override { return "UniformIntegral"; }
    void eval_cpu(const std::vector<mx::array>& inputs, std::vector<mx::array>& outputs) override {
        auto& out = outputs[0];
        out.set_data(mx::allocator::malloc(out.nbytes()));
        auto& encoder = mx::cpu::get_command_encoder(stream());
        for (const auto& input : inputs) encoder.set_input_array(input);
        encoder.set_output_array(out);
        encoder.dispatch([t = inputs[0].data<double>(), values = inputs[1].data<double>(),
                          prefixes = inputs[2].data<double>(), anchors = inputs[3].data<double>(),
                          result = out.data<double>(), count = out.size(), cells = inputs[2].shape(1),
                          tables = inputs[1].shape(0), first = first_, step = step_,
                          linear = linear_, derivative = derivative_]() {
            scipy_uniform_integral(t, values, prefixes, anchors, count, cells, tables,
                                   first, step, linear, derivative, result);
        });
    }
    void eval_gpu(const std::vector<mx::array>&, std::vector<mx::array>&) override {
        throw std::runtime_error("UniformIntegral requires the CPU stream for float64.");
    }
    bool is_equivalent(const mx::Primitive& other) const override {
        const auto& rhs = static_cast<const UniformIntegral&>(other);
        return first_ == rhs.first_ && step_ == rhs.step_ && linear_ == rhs.linear_ && derivative_ == rhs.derivative_;
    }
};

mx::array evaluate(const mx::array& t, const mx::array& values, const mx::array& prefixes,
                   const mx::array& anchors, int64_t first, double step, bool linear, bool derivative) {
    if (values.ndim() != 2 || prefixes.ndim() != 2 || anchors.ndim() != 1 ||
        values.shape(1) < 2 || values.shape(0) < 1 || values.shape(0) != prefixes.shape(0) ||
        values.shape(0) != anchors.size() || values.shape(1) != prefixes.shape(1) + 1 ||
        !valid_uniform_grid(first, prefixes.shape(1), values.shape(0), step))
        throw std::invalid_argument("Expected values (tables, cells+1), prefixes (tables, cells), anchors (tables), positive finite step, and finite grid endpoints with consecutive float64 integer table indices.");
    auto stream = mx::default_stream(mx::Device::cpu);
    std::vector<mx::array> inputs;
    for (const auto& input : {t, values, prefixes, anchors}) {
        if (input.dtype() == mx::complex64)
            throw std::invalid_argument("Uniform integral inputs must be real.");
        inputs.push_back(mx::contiguous(mx::astype(input, mx::float64, stream), false, stream));
    }
    return mx::array(t.shape(), mx::float64,
                    std::make_shared<UniformIntegral>(stream, first, step, linear, derivative), inputs);
}

NB_MODULE(_uniform_integral_mlx, m) {
    m.def("evaluate", &evaluate);
    m.def("grid_span", [](const mx::array& t, double step) {
        if (t.size() == 0 || !std::isfinite(step) || step <= 0)
            throw std::invalid_argument("Expected nonempty finite queries and positive finite grid spacing.");
        auto stream = mx::default_stream(mx::Device::cpu);
        auto input = mx::contiguous(mx::astype(t, mx::float64, stream), false, stream);
        mx::eval({input});
        int64_t first, last;
        if (scipy_uniform_grid_span(input.data<double>(), input.size(), step, &first, &last) != 0)
            throw std::invalid_argument("Queries must be finite with grid indices in the signed int64 range.");
        return nb::make_tuple(first, last);
    });
}
