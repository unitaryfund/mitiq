.PHONY: all
all: dist

.PHONY: check-style
check-style:
	flake8

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
	make -C docs doctest

.PHONY: install
install:
	pip install -e .

.PHONY: requirements
requirements: requirements.txt
	pip install -r requirements.txt

.PHONY: test
test:
	pytest -v --cov=mitiq mitiq/tests mitiq/benchmarks/tests

.PHONY: test-pyquil
test-pyquil:
	pytest -v --cov=mitiq mitiq/mitiq_pyquil/tests

.PHONY: test-qiskit
test-qiskit:
	pytest -v --cov=mitiq mitiq/mitiq_qiskit/tests

.PHONY: test-all
test-all:
	pytest --cov=mitiq
