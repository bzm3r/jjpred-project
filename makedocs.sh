#!/bin/bash

# save any changes in the main repository
git status
git add .
git commit -m "wip"

# change to the jjpred docs folder
cd ./jjpred/docs || exit

# activate the virtual environment
source ../../.venv/bin/activate

rm -rf ./_*/
rm -rf ./*.rst
Copy-Item -Path "index_template" "index.rst"

# clean + produce HTML
make clean
make html

# get back to the main jjpred-project folder
cd ../../

# checkout the pages branch
git checkout pages

# remove any stuff from the old build
rm -rf ./_*/
rm -f ./*.html
rm -f ./*.inv
rm -f ./*.js

# in the pages branch, all the other stuff from the project is ignored
# so we copy from the ignored docs folder into the main folder
cp -r ./jjpred/docs/_build/html/* ./

# commit, push, and switch back to main
git add .
git commit -m "update documentation"
git push
git checkout main
