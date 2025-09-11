include .env

VENV_PREFIX = poetry run

init: init_venv

init-ci: init-poetry init_venv

init-poetry:
	python3 -m pip install poetry

init_venv:
	@echo "Make sure you are using pyenv otherwise make sure python version match .python-version" && python3 -V
	poetry env use python3
	poetry install

lint: lint_ruff lint_pylint lint_mypy format_check_ruff

lint_ruff:
	$(VENV_PREFIX) ruff check

lint_ruff_fix:
	echo "Be careful for automatic fixes..."
	$(VENV_PREFIX) ruff check --fix

lint_pylint:
	$(VENV_PREFIX) pylint ltt_ff_frontend

lint_mypy:
	$(VENV_PREFIX) mypy ltt_ff_frontend

format_check_ruff:
	$(VENV_PREFIX) ruff format --check

preview_format: preview_isort_format preview_ruff_format

preview_isort_format:
	$(VENV_PREFIX) ruff check --select I

preview_ruff_format:
	$(VENV_PREFIX) ruff format --diff

formater: formater_isort formater_ruff

formater_isort:
	$(VENV_PREFIX) ruff check --select I --fix

formater_ruff:
	$(VENV_PREFIX) ruff format

DEV_PORT = $(shell echo 8590+$(DEV_NUM) | bc)

run_dev_defect_ui_gpu01:
	FF_ENV="dev" $(VENV_PREFIX) streamlit run ltt_ff_frontend/run_defect_ui.py --browser.gatherUsageStats false --server.port $(DEV_PORT) --server.address 192.168.201.11

run_dev_defect_ui_gpu02:
	FF_ENV="dev" $(VENV_PREFIX) streamlit run ltt_ff_frontend/run_defect_ui.py --browser.gatherUsageStats false --server.port $(DEV_PORT) --server.address 192.168.201.12

run_dev_defect_ui_gpu04:
	FF_ENV="dev" $(VENV_PREFIX) streamlit run ltt_ff_frontend/run_defect_ui.py --browser.gatherUsageStats false --server.port $(DEV_PORT) --server.address 192.168.201.14

run_dev_defect_ui_internal_gpu01:
	FF_ENV="dev" $(VENV_PREFIX) streamlit run ltt_ff_frontend/run_defect_ui_internal.py --browser.gatherUsageStats false --server.port $(DEV_PORT) --server.address 192.168.201.11

run_dev_defect_ui_internal_gpu02:
	FF_ENV="dev" $(VENV_PREFIX) streamlit run ltt_ff_frontend/run_defect_ui_internal.py --browser.gatherUsageStats false --server.port $(DEV_PORT) --server.address 192.168.201.12

run_dev_defect_ui_internal_gpu04:
	FF_ENV="dev" $(VENV_PREFIX) streamlit run ltt_ff_frontend/run_defect_ui_internal.py --browser.gatherUsageStats false --server.port $(DEV_PORT) --server.address 192.168.201.14
