#
# Makefile for phantom-setup examples
#
# The markdown files can be converted to Jupyter notebooks, Python scripts, or
# static HTML pages.

.PHONY: help notebook python html

help:
	@echo "Makefile targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  %-20s %s\n", $$1, $$2}'

notebook: $(patsubst %.md,%.ipynb,$(wildcard *.md)) ## Convert to Jupyter notebooks

python: $(patsubst %.md,%.py,$(wildcard *.md)) ## Convert to Python scripts

html: $(patsubst %.md,%.html,$(wildcard *.md)) ## Convert to HTML

%.ipynb: %.md
	jupytext --to ipynb $<

%.py: %.md
	jupytext --to py $<

%.html: %.ipynb
	jupyter nbconvert --to html --execute $<
