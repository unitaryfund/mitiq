.PHONY: all
all: dist

.PHONY: build
build: check-all docs test-all

.PHONY: check-all
check-all: check-format check-style check-types

.PHONY: check-format
check-format:
	black --check --diff mitiq

.PHONY: check-style
check-style:
	flake8

.PHONY: check-types
check-types:
	mypy .

.PHONY: clean
clean:
	rm -rf dist
	rm -rf mitiq.egg-info
	rm -rf .pytest_cache/

.PHONY: dist
dist:
	python setup.py sdist

.PHONY: docs
docs:
	make -C docs html

.PHONY: docs-clean
docs-clean:
	make -C docs clean
	make -C docs html

.PHONY: doctest
doctest:
	make -C docs doctest

.PHONY: format
format:
	black mitiq

.PHONY: install
install:
	pip install -e .[development]

.PHONY: pdf
pdf:
	echo "s" | make -C docs latexpdf

.PHONY: requirements
requirements: requirements.txt
	pip install -r requirements.txt

.PHONY: test
test:
	pytest -n auto -v --cov=mitiq --cov-report=term --cov-report=xml mitiq/benchmarks mitiq/pec mitiq/tests mitiq/zne

.PHONY: test-pyquil
test-pyquil:
	pytest -v --cov=mitiq --cov-report=term --cov-report=xml mitiq/mitiq_pyquil

.PHONY: test-qiskit
test-qiskit:
	pytest -v --cov=mitiq --cov-report=term --cov-report=xml mitiq/mitiq_qiskit

.PHONY: test-cirq
test-cirq:
	pytest -v --cov=mitiq --cov-report=term --cov-report=xml mitiq/mitiq_cirq

.PHONY: test-all
test-all:
	pytest -n auto -v --cov=mitiq --cov-report=term --cov-report=xml
