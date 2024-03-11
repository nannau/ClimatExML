---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.5
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# How to build docs

Policy is to checkout `docs_source` before developing the docs. 

```
git checkout docs_source
```

## Jupyter Book
https://jupyterbook.org/en/stable/publish/gh-pages.html

## Python Environment

Create a documentation virtual environment in Python, we will call it `jbook`.

```bash
python -m venv jbook
source jbook/bin/activate
pip install jupyter-book
```

## Building the docs
In the ClimatExML repo, the subdirectory `docs` has the relevent docs information. Edit these values on the branch `docs_source`. To build the docs from source:

```
jupyter-book build docs/
```

## Publishing docs
https://jupyterbook.org/en/stable/start/publish.html

Install ghp-import with
```
pip install ghp-import
```

Push built docs to published branch with

```
ghp-import -n -p -f _build/html
```