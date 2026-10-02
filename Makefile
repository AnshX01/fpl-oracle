.PHONY: help setup run train test verify clean

PYTHON ?= python

help:
	@echo "FPL Oracle Commands:"
	@echo "  make setup   - Install dependencies and prepare local directories"
	@echo "  make run     - Run the web application at http://localhost:8000"
	@echo "  make train   - Download historical data, build features, train ML models"
	@echo "  make test    - Run unit tests with pytest"
	@echo "  make verify  - Run full self-check verification suite"
	@echo "  make clean   - Clean temporary and cache files"

setup:
	$(PYTHON) run.py setup

run:
	$(PYTHON) run.py run

train:
	$(PYTHON) run.py train

test:
	$(PYTHON) run.py test

verify:
	$(PYTHON) run.py verify

clean:
	$(PYTHON) run.py clean
