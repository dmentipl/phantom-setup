Examples
--------

Examples for using `phantom-setup`.

They are written as markdown files which can be converted to Jupyter notebooks, Python scripts, or HTML pages. The requirements to do this are Jupyter and Jupytext. They can be installed with Conda.

If you have Jupyter lab and Jupytext you can run the markdown directly in Jupyter lab without first converting to the ipynb format.

If you do want to convert to other formats, there is a Makefile to facilitate this. To see what the Makefile can to type the following.

```
make help
```

To execute the markdown files directly at the command line without converting first, do

```
jupytext --execute the_name_of_the_file.md
```

See https://github.com/mwouts/jupytext for details on Jupytext.
