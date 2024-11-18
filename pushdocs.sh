#!/bin/bash

cd ./jjpred/docs || exit ; ./generatedocs.sh

cd ../../
git checkout pages
cp -r ./jjpred/docs/_build/html/* ./
git add .
git commit -m "update documentation"
git push
git checkout main
