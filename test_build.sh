#!/usr/bin/env bash

if [ "$1" != "-docs" ]; then
    echo "Running Tests...";
    pytest mitiq/tests;
    pytest mitiq/benchmarks/tests;
    pytest mitiq/mitiq_qiskit/tests;
elif [ "$1" != "-tests" ]; then
    echo "Building and Testing Docs...";
    cd docs && make html;
    make doctest;
fi
