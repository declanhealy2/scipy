#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
#include <cmath>
#include "uniform_integral.h"

static PyObject* evaluate(PyObject*, PyObject* args) {
    PyObject *objects[4];
    long long first;
    double step;
    int linear, derivative;
    if (!PyArg_ParseTuple(args, "OOOOLdpp", &objects[0], &objects[1],
                          &objects[2], &objects[3], &first, &step, &linear, &derivative))
        return nullptr;
    PyArrayObject* inputs[4] = {nullptr, nullptr, nullptr, nullptr};
    PyArrayObject* output = nullptr;
    for (int i = 0; i < 4; ++i) {
        inputs[i] = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(objects[i], NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
        if (!inputs[i]) goto cleanup;
    }
    if (PyArray_NDIM(inputs[1]) != 2 || PyArray_NDIM(inputs[2]) != 2 ||
        PyArray_NDIM(inputs[3]) != 1 || PyArray_DIM(inputs[1], 1) < 2 ||
        PyArray_DIM(inputs[1], 0) < 1 ||
        PyArray_DIM(inputs[1], 0) != PyArray_DIM(inputs[2], 0) ||
        PyArray_DIM(inputs[1], 0) != PyArray_SIZE(inputs[3]) ||
        PyArray_DIM(inputs[1], 1) != PyArray_DIM(inputs[2], 1) + 1 ||
        !valid_uniform_grid(first, PyArray_DIM(inputs[2], 1), PyArray_DIM(inputs[1], 0), step)) {
        PyErr_SetString(PyExc_ValueError, "Expected values (tables, cells+1), prefixes (tables, cells), anchors (tables), positive finite step, and finite grid endpoints with consecutive float64 integer table indices.");
        goto cleanup;
    }
    output = reinterpret_cast<PyArrayObject*>(PyArray_SimpleNew(
        PyArray_NDIM(inputs[0]), PyArray_DIMS(inputs[0]), NPY_DOUBLE));
    if (output) {
        Py_BEGIN_ALLOW_THREADS
        scipy_uniform_integral(
            static_cast<const double*>(PyArray_DATA(inputs[0])),
            static_cast<const double*>(PyArray_DATA(inputs[1])),
            static_cast<const double*>(PyArray_DATA(inputs[2])),
            static_cast<const double*>(PyArray_DATA(inputs[3])),
            PyArray_SIZE(inputs[0]), PyArray_DIM(inputs[2], 1),
            PyArray_DIM(inputs[1], 0), first, step, linear, derivative,
            static_cast<double*>(PyArray_DATA(output)));
        Py_END_ALLOW_THREADS
    }
cleanup:
    for (auto input : inputs) Py_XDECREF(input);
    return reinterpret_cast<PyObject*>(output);
}

static PyObject* grid_span(PyObject*, PyObject* args) {
    PyObject* object;
    double step;
    if (!PyArg_ParseTuple(args, "Od", &object, &step)) return nullptr;
    auto input = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(object, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    if (!input) return nullptr;
    if (!std::isfinite(step) || step <= 0 || PyArray_SIZE(input) == 0) {
        Py_DECREF(input);
        PyErr_SetString(PyExc_ValueError, "Expected nonempty finite queries and positive finite grid spacing.");
        return nullptr;
    }
    int64_t first, last;
    int status;
    Py_BEGIN_ALLOW_THREADS
    status = scipy_uniform_grid_span(static_cast<double*>(PyArray_DATA(input)), PyArray_SIZE(input), step, &first, &last);
    Py_END_ALLOW_THREADS
    Py_DECREF(input);
    if (status != 0) {
        PyErr_SetString(PyExc_ValueError, "Queries must be finite with grid indices in the signed int64 range.");
        return nullptr;
    }
    return Py_BuildValue("LL", static_cast<long long>(first), static_cast<long long>(last));
}

static PyMethodDef methods[] = {
    {"evaluate", evaluate, METH_VARARGS, nullptr},
    {"grid_span", grid_span, METH_VARARGS, nullptr},
    {nullptr, nullptr, 0, nullptr}
};
static PyModuleDef module = {PyModuleDef_HEAD_INIT, "_uniform_integral_numpy", nullptr, -1, methods};
PyMODINIT_FUNC PyInit__uniform_integral_numpy() {
    import_array();
    return PyModule_Create(&module);
}
