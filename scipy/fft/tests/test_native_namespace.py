import numpy as np
import pytest

from scipy import fft
from scipy._lib import _array_api_override


@pytest.mark.parametrize("name", ["fft", "ifft", "fftn", "ifftn"])
def test_mlx_fft_dispatch_without_environment_switch(monkeypatch, name):
    mx = pytest.importorskip("mlx.core")
    monkeypatch.setattr(_array_api_override, "SCIPY_ARRAY_API", False)
    values = mx.array([1, 0, 0, 0], dtype=mx.complex64)
    original = np.asarray

    def forbid_host_conversion(value, *args, **kwargs):
        assert not isinstance(value, mx.array), "FFT converted an MLX array to NumPy"
        return original(value, *args, **kwargs)

    with monkeypatch.context() as context:
        context.setattr(np, "asarray", forbid_host_conversion)
        with fft.set_workers(2):
            result = getattr(fft, name)(values)
            mx.eval(result)
    assert isinstance(result, mx.array)
    assert result.dtype == mx.complex64
    expected = 0.25 if name.startswith("i") else 1.0
    np.testing.assert_array_equal(result, np.full(4, expected))


def test_numpy_arraylike_and_worker_context():
    with fft.set_workers(2):
        result = fft.fft([1, 0, 0, 0])
    np.testing.assert_array_equal(result, np.ones(4))
