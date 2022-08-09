.PHONY: all
all: dist

.PHONY: build
build: check-all docs test-all

.PHONY: check-all
check-all: check-format check-style check-types

.PHONY: check-format
check-format:
	poetry run black --check --diff mitiq

.PHONY: check-style
check-style:
	poetry run flake8

.PHONY: check-types
check-types:
	poetry run mypy mitiq --show-error-codes

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

.PHONY: linkcheck
linkcheck:
	make -C docs linkcheck

.PHONY: format
format:
	poetry run black mitiq

.PHONY: install
install:
	poetry install

.PHONY: pdf
pdf:
	echo "s" | make -C docs latexpdf

.PHONY: requirements
requirements:
	poetry install --no-dev

.PHONY: test
test:
	poetry run pytest -n auto -v --cov=mitiq --cov-report=term --cov-report=xml --ignore=mitiq/interface/mitiq_pyquil

.PHONY: test-pyquil
test-pyquil:
	poetry run pytest -v --cov=mitiq --cov-report=term --cov-report=xml mitiq/interface/mitiq_pyquil

.PHONY: test-all
test-all:
	poetry run pytest -n auto -v --cov=mitiq --cov-report=term --cov-report=xml
