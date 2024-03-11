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

## Jupyter Book
https://jupyterbook.org/en/stable/publish/gh-pages.html

## Python Environment

Create a documentation virtual environment in Python, we will call it `jbook`.

```bash
python -m venv jbook
source jbook/bin/activate
pip install jupyter-book
```