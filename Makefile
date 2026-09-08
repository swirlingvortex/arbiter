ARBITER_BOOTSTRAP_PYTHON ?= python3.12
VENV := .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
PYTEST := $(VENV)/bin/pytest
RUFF := $(VENV)/bin/ruff
MYPY := $(VENV)/bin/mypy

.PHONY: install lint typecheck test verify

$(PYTHON):
	$(ARBITER_BOOTSTRAP_PYTHON) -m venv $(VENV)

install: $(PYTHON)
	$(PYTHON) -m pip install --upgrade pip
	$(PIP) install -e ".[dev]"

lint:
	$(RUFF) check .
	$(RUFF) format --check src tests

typecheck:
	$(MYPY) src

test:
	$(PYTEST)

verify: lint typecheck test
