.PHONY: serve build monitor

serve:
	pip install -r requirements-doc.txt
	mkdocs serve

monitor:
	python tools/generate_monitoring.py

build:
	python tools/generate_monitoring.py
	mkdocs build --clean
