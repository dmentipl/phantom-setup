#
# Makefile for phantom-setup examples
#
# The Python "percent" formatted files can be converted to Jupyter notebooks,
# reStructuredText for documentation, or static HTML pages.

.PHONY: help all notebook rst html

help:
	@echo "Makefile targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  %-20s %s\n", $$1, $$2}'

all: notebook rst html ## Convert to all

notebook: $(patsubst %.py,%.ipynb,$(wildcard *.py)) ## Convert to Jupyter notebooks

rst: $(patsubst %.py,%.rst,$(wildcard *.py)) ## Convert to RST

html: $(patsubst %.py,%.html,$(wildcard *.py)) ## Convert to HTML

clean:
	rm *.ipynb; rm *.py; rm *.html

%.ipynb: %.py
	jupytext --to ipynb $<

%.rst: %.py
	jupytext --to rst $<

%.html: %.ipynb
	jupyter nbconvert --to html --execute $< --ExecutePreprocessor.timeout=180
