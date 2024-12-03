#!/bin/bash

source ../../.venv/bin/activate

sphinx-apidoc --force --remove-old --output-dir ./ ../ "docs"

cp index_template index.rst

make clean

make html
