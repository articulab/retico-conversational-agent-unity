# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join("../src")))
sys.path.insert(0, os.path.abspath(".."))

import retico_conversational_agent

autoapi_dirs = ["../src"]
autodoc_inherit_docstrings = True
autoapi_add_toctree_entry = False
autoapi_template_dir = "_templates/"
autoapi_options = [
    "members",
    "private-members",
    "show-inheritance",
    "show-module-summary",
    "special-members",
    "imported-members",
    # "undoc-members",
    # "inherited-members",
]

# -- Project information -----------------------------------------------------

project = "retico-conversational-agent"
copyright = "2024, Marius Le Chapelier"
author = "Marius Le Chapelier"

# The full version, including alpha/beta/rc tags
# release = retico_core.__version__


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "autoapi.extension",
    # "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "myst_parser",
    "sphinx.ext.autosectionlabel",
]

# MyST-Parser configurations (optional, for customizations)
myst_enable_extensions = [
    "colon_fence",  # Allows the use of :: fenced code blocks
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
