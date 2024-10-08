SHELL := /bin/bash

init:
	poetry install
	poetry run pre-commit install
	poetry env info
	@echo "Created virtual environment"

test:
	poetry run pytest --cov=tests/ --no-cov-on-fail

format:
	ruff format --line-length=80
	ruff check --fix
	poetry run mypy --ignore-missing-imports src/

clean:
	rm -rf .venv
	rm -rf .mypy_cache
	rm -rf .pytest_cache
	rm -rf build/
	rm -rf dist/
	rm -rf juninit-pytest.xml
	find . -name ".coverage*" -delete
	find . -name --pycache__ -exec rm -r {} +
