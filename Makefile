# .PHONY: target dev format lint test coverage-html pr  build build-docs build-docs-api build-docs-website
# .PHONY: docs-local docs-api-local security-baseline complexity-baseline release-prod release-test release

# target:
# 	@$(MAKE) pr

init:
	mv myapp ${appname}
	sed -i "" 's/myapp/${appname}/g' pyproject.toml terraform/*.tf terraform/environments/global.tfvars terraform/environments/*/*.tfvars ${appname}/*.py tests/*.py build.gradle
	sed -i "" 's/###PROJECT_NAME###/${reponame}/g' settings.gradle scripts/entrypoint.bash
	sed -i "" 's/###VERTICAL_NAME###/${vertical}/g' build.gradle
	mv .github/workflows/pytest.txt .github/workflows/pytest.yml

dev:
	poetry config virtualenvs.in-project true --local
	poetry install
	poetry run pre-commit install
	

format:
	poetry run black .

lint: format
    # excludes via .flake8 file
	poetry run flake8

test:
	poetry run pytest tests --cov=${appname}
	poetry run pytest tests

requirements_dev:
	poetry export --output requirements-dev.txt --dev --without-hashes

requirements:
	poetry export --output requirements.txt --without-hashes
