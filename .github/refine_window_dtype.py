from pathlib import Path


source_path = Path("scipy/signal/windows/_windows.py")
source = source_path.read_text()

anchor = '''def _general_cosine_impl(M, a, xp, device, sym=True, dtype=None):
    dtype = xp.float64 if dtype is None else dtype
'''
replacement = '''def _validate_window_dtype(xp, dtype):
    dtype = xp.float64 if dtype is None else dtype
    if not xp.isdtype(dtype, "real floating"):
        raise ValueError("`dtype` must be a real floating data type.")
    return dtype


def _general_cosine_impl(M, a, xp, device, sym=True, dtype=None):
    dtype = _validate_window_dtype(xp, dtype)
'''
if source.count(anchor) != 1:
    raise RuntimeError("general cosine dtype anchor changed")
source = source.replace(anchor, replacement)

old_resolution = "    dtype = xp.float64 if dtype is None else dtype\n"
new_resolution = "    dtype = _validate_window_dtype(xp, dtype)\n"
head, separator, tail = source.partition("def _general_cosine_impl")
if not separator:
    raise RuntimeError("general cosine implementation missing")
count = tail.count(old_resolution)
if count != 6:
    raise RuntimeError(f"expected six public dtype resolutions, found {count}")
tail = tail.replace(old_resolution, new_resolution)
source = head + separator + tail

old_doc = '''    dtype : dtype, optional
        Data type of the returned window. The default is ``float64``.
'''
new_doc = '''    dtype : dtype, optional
        Real floating data type of the returned window. The default is ``float64``.
'''
count = source.count(old_doc)
if count != 8:
    raise RuntimeError(f"expected eight dtype doc entries, found {count}")
source = source.replace(old_doc, new_doc)
source_path.write_text(source)

test_path = Path("scipy/signal/tests/test_windows_dtype.py")
tests = test_path.read_text()
addition = '''

@make_xp_test_case(*WINDOW_FUNCTIONS)
@pytest.mark.parametrize("window, extra_args", WINDOW_CASES)
@pytest.mark.parametrize("dtype_name", ("int32", "complex64"))
def test_basic_window_rejects_nonfloating_dtype(xp, window, extra_args, dtype_name):
    with pytest.raises(ValueError, match="real floating"):
        window(8, *extra_args, xp=xp, dtype=getattr(xp, dtype_name))
'''
if "test_basic_window_rejects_nonfloating_dtype" in tests:
    raise RuntimeError("invalid dtype test already exists")
test_path.write_text(tests + addition)
