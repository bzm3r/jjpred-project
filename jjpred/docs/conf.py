# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "jjpred"
copyright = "2024, JJ"
author = "JJ"

# -- Project setup -----------------------------------------------------
#

exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    ".venv",
    ".vscode",
    "analysis_*",
    "dist",
    "excel",
    "jjpred-notebooks",
    "*.md",
    "*.sh",
    "*.toml",
    "./**/__pycache__",
    "*.pyc",
    "docs",
]


# jjpred_path = get_main_folder("jjpred-project").joinpath("jjpred")
# print(f"{jjpred_path=}")
# source_folders = [jjpred_path]


# def find_sub_modules(path: Path) -> list[Path]:
#     sub_modules = []

#     for inner in path.iterdir():
#         if inner.is_dir():
#             if not (inner.name.startswith("__") and inner.name.endswith("__")):
#                 sub_modules.append(inner)
#                 sub_modules += find_sub_modules(inner)

#     return sub_modules


# print(f"using source folders: {source_folders}")

# for folder in source_folders:
#     sys.path.insert(0, folder.as_posix())


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
]

templates_path = ["_templates"]

autosummary_generate = True  # Turn on autosummary

autodoc_default_options = {
    "members": True,  # Include members in the documentation
    "undoc-members": True,  # Include undocumented members
    "private-members": False,  # Include private members (starting with _)
    "show-inheritance": True,  # Show base classes
}
autodoc_typehints = "both"

todo_include_todos = True

viewcode_follow_imported_members = True

# https://www.sphinx-doc.org/en/master/usage/extensions/intersphinx.html#configuration

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "polars": ("https://docs.pola.rs/api/python/stable", None),
}

# https://www.sphinx-doc.org/en/master/usage/extensions/todo.html#configuration

todo_include_todos = True


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "alabaster"
html_theme_options = {
    "caption_font_family": "Inconsolata, monospace",
    "font_family": "Inconsolata, monospace",
    "head_font_family": "Inconsolata, monospace",
}
html_static_path = ["static"]
# html_css_files = [
#     "style.css",
# ]
