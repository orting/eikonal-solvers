#!/bin/bash
pkg=eikonal_solvers
python3 -m build && 
    pip uninstall -y ${pkg} &&
    pip install ${pkg} -f dist
