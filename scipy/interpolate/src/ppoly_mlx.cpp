#include <nanobind/nanobind.h>
#include "mlx/ops.h"
#include "mlx/primitives.h"
#include "mlx/allocator.h"
#include "mlx/backend/cpu/encoder.h"
#include "ppoly_native.h"

namespace mx = mlx::core;
namespace nb = nanobind;

class PPolyEvaluate : public mx::Primitive {
    int derivative_;
    bool extrapolate_;
public:
    PPolyEvaluate(mx::Stream stream, int derivative, bool extrapolate)
        : mx::Primitive(stream), derivative_(derivative), extrapolate_(extrapolate) {}
    const char* name() const override { return "PPolyEvaluate"; }
    void eval_cpu(const std::vector<mx::array>& inputs, std::vector<mx::array>& outputs) override {
        auto& out = outputs[0];
        out.set_data(mx::allocator::malloc(out.nbytes()));
        auto& encoder = mx::cpu::get_command_encoder(stream());
        for (const auto& input : inputs) encoder.set_input_array(input);
        encoder.set_output_array(out);
        encoder.dispatch([c = inputs[0].data<double>(), knots = inputs[1].data<double>(),
                          query = inputs[2].data<double>(), result = out.data<double>(),
                          components = inputs[0].shape(2), cells = inputs[0].shape(1),
                          orders = inputs[0].shape(0), count = inputs[2].size(),
                          derivative = derivative_, extrapolate = extrapolate_]() {
            scipy_ppoly_evaluate(c, knots, query, components, cells, orders, count,
                                 derivative, extrapolate, int64_t(cells) * components,
                                 components, components, result);
        });
    }
    void eval_gpu(const std::vector<mx::array>&, std::vector<mx::array>&) override {
        throw std::runtime_error("PPoly float64 evaluation requires the CPU stream.");
    }
    bool is_equivalent(const mx::Primitive& other) const override {
        const auto& rhs = static_cast<const PPolyEvaluate&>(other);
        return derivative_ == rhs.derivative_ && extrapolate_ == rhs.extrapolate_;
    }
};

mx::array evaluate(const mx::array& c, const mx::array& knots, const mx::array& query,
                   int derivative, bool extrapolate) {
    if (c.ndim() != 3 || c.shape(0) < 1 || c.shape(1) < 1 || knots.ndim() != 1 ||
        knots.size() != c.shape(1) + 1 || query.ndim() != 1 || derivative < 0)
        throw std::invalid_argument("Expected coefficients (order, intervals, components), matching knots, flat queries and nonnegative derivative order.");
    auto stream = mx::default_stream(mx::Device::cpu);
    std::vector<mx::array> inputs;
    for (const auto& input : {c, knots, query}) {
        if (input.dtype() == mx::complex64)
            throw std::invalid_argument("Real PPoly evaluation requires real inputs.");
        inputs.push_back(mx::contiguous(mx::astype(input, mx::float64, stream), false, stream));
    }
    return mx::array({query.shape(0), c.shape(2)}, mx::float64,
                    std::make_shared<PPolyEvaluate>(stream, derivative, extrapolate), inputs);
}

NB_MODULE(_ppoly_mlx, m) {
    m.def("evaluate", &evaluate);
}
