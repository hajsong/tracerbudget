#!/bin/bash

rm -f *.c
rm -f *.so

python setup.py build_ext --inplace

mv tracerbudget/* .
\rm -r tracerbudget

