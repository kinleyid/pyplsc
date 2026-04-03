# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'pyplsc'
copyright = '2026, Isaac Kinley'
author = 'Isaac Kinley'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
	'sphinx.ext.autodoc',
	'sphinx.ext.napoleon',
	'nbsphinx'
]

napoleon_numpy_docstring = True
napoleon_use_ivar = False

# nbsphinx_execute = "never"
nbsphinx_execute = "always"
exclude_patterns = ["build", "**.ipynb_checkpoints"]

import os
import sys
sys.path.insert(0, os.path.abspath('../..'))

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']
