# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'DAISY'
copyright = '2025, Fabian Hofmann, Seraphin Zunzer, Jonathan Ackerschewski, Lotta Fejzula'
author = 'Fabian Hofmann, Seraphin Zunzer, Jonathan Ackerschewski, Lotta Fejzula'
release = '1.0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_baseurl = "https://daisy-field.github.io/"
html_logo= "_static/favicon2.ico"
html_favicon = "_static/favicon2.ico"
html_static_path = ['https://daisy-field.github.io/_static']


extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon'
]

html_theme = 'sphinx_rtd_theme'

show_authors = False


def remove_author_and_modified_lines(app, what, name, obj, options, lines):
    # Remove lines that start with 'Author' or 'Modified'
    if lines:
        lines[:] = [line for line in lines if not (line.strip().startswith("Author") or line.strip().startswith("Modified"))]

def setup(app):
    app.connect("autodoc-process-docstring", remove_author_and_modified_lines)