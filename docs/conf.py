# This code is part of qcr.
#
# (C) Copyright Arnav Kapoor 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.

"""Sphinx configuration."""

import qcr

project = "qcr"
author = "Arnav Kapoor"
copyright = "2026, Arnav Kapoor"  # noqa: A001
release = qcr.__version__
version = ".".join(release.split(".")[:2])

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "reno.sphinxext",
]

autodoc_default_options = {"members": True, "undoc-members": False, "show-inheritance": True}
autodoc_typehints = "description"
autodoc_mock_imports = ["qiskit_ibm_runtime", "qiskit_aer", "pytket"]
napoleon_google_docstring = True
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "qiskit": ("https://quantum.cloud.ibm.com/docs/api/qiskit", None),
}
templates_path = ["_templates"]
exclude_patterns = ["_build"]
nitpicky = False
suppress_warnings = ["intersphinx"]

html_theme = "furo"
html_title = f"qcr {release}"
html_static_path = ["_static"]
