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
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))

# -- Project information -----------------------------------------------------
# TODO: change to minigrid version
# from TODO import __version__ as minigrid_version

from __future__ import annotations

import os
import sys
from typing import Any

project = "MiniGrid"
copyright = "2022"
author = "Farama Foundation"

# The full version, including alpha/beta/rc tags
# TODO: change to minigrid version
release = "1.2.1"

sys.path.insert(0, os.path.abspath("../.."))

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    "sphinx.ext.githubpages",
    "myst_parser",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# Napoleon settings
napoleon_use_ivar = True
napoleon_use_admonition_for_references = True
# See https://github.com/sphinx-doc/sphinx/issues/9119
napoleon_custom_sections = [("Returns", "params_style")]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "furo"
html_title = "MiniGrid Documentation"
html_baseurl = "https://minigrid.farama.org/"
html_copy_source = False
html_favicon = "_static/img/minigrid-favicon.png"
html_theme_options = {
    "light_logo": "img/minigrid.svg",
    "dark_logo": "img/minigrid-white.svg",
    "gtag": "G-FBXJQQLXKD",
}
html_context: dict[str, Any] = {}
html_context["conf_py_path"] = "/docs/"
html_context["display_github"] = True
html_context["github_user"] = "Farama-Foundation"
html_context["github_repo"] = "Minigrid"
html_context["github_version"] = "master"
html_context["slug"] = "minigrid"

html_static_path = ["_static"]
html_css_files = []
