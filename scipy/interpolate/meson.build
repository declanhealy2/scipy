uniform_kernel = shared_library('uniform_integral', 'src/uniform_integral.f90',
  dependencies: dependency('openmp', language: 'fortran'),
  fortran_args: ff.get_supported_arguments('-ffp-contract=off'),
  install: true, install_dir: py3.get_install_dir() / 'scipy/interpolate',
)
py3.extension_module('_uniform_integral_numpy', 'src/uniform_integral_numpy.cpp',
  dependencies: np_dep, link_with: uniform_kernel,
  install_rpath: host_machine.system() == 'darwin' ? '@loader_path' : '$ORIGIN',
  install: true, subdir: 'scipy/interpolate',
)
py3.install_sources('_uniform_integral.py', subdir: 'scipy/interpolate')

if get_option('uniform-mlx') and host_machine.system() == 'darwin'
  mlx_dir = run_command(py3, '-c',
    'import mlx.core, pathlib; print(pathlib.Path(mlx.core.__file__).parent)',
    check: true).stdout().strip()
  nb_dir = run_command(py3, '-c',
    'import nanobind, pathlib; print(pathlib.Path(nanobind.__file__).parent)',
    check: true).stdout().strip()
  mlx_dep = cpp.find_library('mlx', dirs: mlx_dir / 'lib')
  py3.extension_module('_uniform_integral_mlx',
    ['src/uniform_integral_mlx.cpp', nb_dir / 'src/nb_combined.cpp'],
    include_directories: include_directories(mlx_dir / 'include', nb_dir / 'include',
                                            nb_dir / 'ext/robin_map/include'),
    cpp_args: ['-DNB_DOMAIN=mlx'],
    override_options: ['cpp_std=c++20'],
    dependencies: mlx_dep, link_with: uniform_kernel,
    install_rpath: '@loader_path:@loader_path/../../mlx/lib',
    install: true, subdir: 'scipy/interpolate',
  )
endif

py3.extension_module('_interpnd',
  spt_cython_gen.process('_interpnd.pyx'),
  c_args: [Wno_discarded_qualifiers, cython_c_args],
  dependencies: np_dep,
  link_args: version_link_args,
  install: true,
  subdir: 'scipy/interpolate'
)

py3.extension_module('_ppoly',
  linalg_cython_gen.process('_ppoly.pyx'),
  c_args: cython_c_args,
  dependencies: np_dep,
  link_args: version_link_args,
  install: true,
  subdir: 'scipy/interpolate'
)

py3.extension_module('_rgi_cython',
  linalg_cython_gen.process('_rgi_cython.pyx'),
  c_args: cython_c_args,
  dependencies: np_dep,
  link_args: version_link_args,
  install: true,
  subdir: 'scipy/interpolate'
)

__fitpack_lib = static_library('__fitpack',
    ['src/__fitpack.h', 'src/__fitpack.cc'],
    dependencies:[lapack_dep, np_dep, py3_dep],
)

__fitpack_dep = declare_dependency(
    link_with: __fitpack_lib,
)


py3.extension_module('_dierckx',
    ['src/_dierckxmodule.cc'],
    include_directories: 'src/',
    dependencies: [np_dep, __fitpack_dep],
    link_args: version_link_args,
    install: true,
    subdir: 'scipy/interpolate'
)

py3.extension_module('_fitpack',
  ['src/_fitpackmodule.c', 'src/dfitpack.c'],
  include_directories: 'src/',
  dependencies: np_dep,
  link_args: version_link_args,
  install: true,
  subdir: 'scipy/interpolate'
)

if use_pythran
  py3.extension_module('_rbfinterp_pythran',
    pythran_gen.process('_rbfinterp_pythran.py'),
    cpp_args: cpp_args_pythran,
    dependencies: [pythran_dep, np_dep],
    link_args: version_link_args,
    install: true,
    subdir: 'scipy/interpolate'
  )
else
  py3.install_sources(
    ['_rbfinterp_pythran.py'],
    subdir: 'scipy/interpolate'
  )
endif

py3.install_sources([
    '__init__.py',
    '_bsplines.py',
    '_bary_rational.py',
    '_cubic.py',
    '_fitpack_impl.py',
    '_fitpack_py.py',
    '_fitpack2.py',
    '_fitpack_repro.py',
    '_interpolate.py',
    '_ndgriddata.py',
    '_pade.py',
    '_polyint.py',
    '_rbf.py',
    '_rbfinterp.py',
    '_rbfinterp_common.py',
    '_rbfinterp_np.py',
    '_rbfinterp_xp.py',
    '_regrid.py',
    '_rgi.py',
    '_ndbspline.py',
  ],
  subdir: 'scipy/interpolate'
)

subdir('tests')
