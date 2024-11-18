git status
git add .
git commit -m "wip"

# change to the jjpred docs folder
Set-Location -Path ".\jjpred\docs"

# activate the virtual environment
..\..\.venv\Scripts\activate

Remove-Item -Path ".\_*\" -Recurse -Force
Remove-Item -Path "*.rst" -Recurse -Force
Copy-Item -Path "index_template" "index.rst"
# sphinx-apidoc --force --remove-old --output-dir ./_modules ../ "docs"
# might have to do `./make clean` if you are on Windows+PowerShell
./make clean
./make html
# get back to the main jjpred-project folder
Set-Location -Path "..\..\"

# checkout the pages branch
git checkout pages

Remove-Item -Path ".\_*\" -Recurse -Force
Remove-Item -Path "*.html" -Recurse -Force
Remove-Item -Path "*.inv" -Recurse -Force
Remove-Item -Path "*.js" -Recurse -Force

# in the pages branch, all the other stuff from the project is ignored
# so we copy from the ignored docs folder into the main folder
Copy-Item -Path ".\jjpred\docs\_build\html\*" -Destination ".\" -Recurse

# commit, push, and switch back to main
git add .
git commit -m "update documentation"
git push
git checkout main
