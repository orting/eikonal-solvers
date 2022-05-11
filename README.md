# Python implementations of eikonal solvers
Intended to simplify experimentation and comparison of different algorithms for solving the eikonal equation in 2D and 3D.
FMM implemented in `numpy`. FIM implemented in `numpy`and `torch`.
See `doc/examples` and `test` for example usage.

## Contents
* Fast Marching Method (FMM)
* Fast Iterative Method (FIM), really the Improved Fast Iterative Method 


## Configuration
* `pyproject.toml` : Project configuration (build, test, lint)
* `setup.cfg`      : More project configuration (package info, dependencies). Most importantly the package name which needs to match the namespace hiearchy.
* `setup.py`       : Auto-generated to allow builds assuming `setup.py` existence.
* `tox.ini`        : Configuration for `tox`, see [https://tox.readthedocs.io/en/latest/examples.html](https://tox.readthedocs.io/en/latest/examples.html)
* `MANIFEST.in`    : Include extra files in distribution, see [https://packaging.python.org/guides/using-manifest-in/](https://packaging.python.org/guides/using-manifest-in/)


## Building and installing
We use [`pypa-build`](https://pypa-build.readthedocs.io/en/latest/index.html)

    python -m build
        
This will create a wheel and a tar ball in the directory `dist`, either of which can be distributed and installed using `pip`. To install from the `dist` dir

    pip install eikonal_solvers -f dist


## Testing
[`pytest`](https://docs.pytest.org/en/stable/contents.html) for running tests and [`tox`](https://tox.readthedocs.io/en/latest/) for automating the testing.

Configuration for `pytest` is in `pyproject.toml`. Some tests are slow and can be disabled by removing `--runslow` from the `addopts` variable. Some tests generate images that by default are stored in `test/out`. Alternative directory can be specified in the `addopts` variable with `--outdir=<path-to-directory>`.

Configuration for `tox` is in `tox.ini`.

Tests are run by executing `tox` without parameters from the root of the repository.


### Benchmarks
There are some benchmarks in the test directory for comparing seqeuential/parallel FIM, and for comparing first/second order FMM. These can be run directly from the test dir.


## Development
After building the package, install development dependencies with

    pip install <path-to-wheel>[dev]
    
We use [`pylint`](https://pylint.org/) to do static check and enforce coding standards. Configuration is in `pyproject.toml`. To run do

    pylint src/breathct/classification

and either fix the issues or explicitly ignore them. See the documentation for details on controling linting, including which naming convention to enforce.
