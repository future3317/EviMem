# Convenience aliases for common matmem development tasks.
# Run from the repository root or from this directory.

REPO_ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/..)
PYTHON := python
RUFF := $(PYTHON) -m ruff
PYTEST := $(PYTHON) -m pytest

.PHONY: help test lint format-check run-matpes-exploratory run-campaign-gated run-dual-horizon-attribution

help:
	@echo "Available targets:"
	@echo "  test                         - run the full test suite"
	@echo "  lint                         - run ruff check on src/, tests/, tools/"
	@echo "  format-check                 - run ruff format --check"
	@echo "  run-matpes-exploratory       - example MatPES exploratory run (requires --task/--development-vault/--output)"
	@echo "  run-campaign-gated           - example campaign-gated IC-SARR run (requires --task/--transport-model/--output)"
	@echo "  run-dual-horizon-attribution - example dual-horizon attribution (requires --task/--vault/--transport-model/--source-trace/--output)"

test:
	$(PYTEST) -q $(REPO_ROOT)/tests

lint:
	$(RUFF) check $(REPO_ROOT)/src/matmem $(REPO_ROOT)/tests $(REPO_ROOT)/tools

format-check:
	$(RUFF) format --check $(REPO_ROOT)/src/matmem $(REPO_ROOT)/tests $(REPO_ROOT)/tools

run-matpes-exploratory:
	$(PYTHON) $(REPO_ROOT)/tools/run_matpes_protocol_closed_loop_exploratory.py \
		--task $(TASK) \
		--development-vault $(VAULT) \
		--output $(OUTPUT)

run-campaign-gated:
	$(PYTHON) $(REPO_ROOT)/tools/run_campaign_gated_ic_sarr.py \
		--task $(TASK) \
		--transport-model $(TRANSPORT) \
		--output $(OUTPUT)

run-dual-horizon-attribution:
	$(PYTHON) $(REPO_ROOT)/tools/run_dual_horizon_attribution.py \
		--task $(TASK) \
		--vault $(VAULT) \
		--transport-model $(TRANSPORT) \
		--source-trace $(SOURCE_TRACE) \
		--output $(OUTPUT)
