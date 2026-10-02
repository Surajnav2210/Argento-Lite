PY := .venv/bin/python

.PHONY: setup fetch test app

setup:
	python3.11 -m venv .venv
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -e ".[dev]"

fetch:
	$(PY) scripts/fetch_data.py

test:
	$(PY) -m pytest -q

app:
	.venv/bin/streamlit run app/streamlit_app.py
