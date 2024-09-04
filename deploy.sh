#!/bin/bash

julia -e 'using Franklin; optimize()'
cd ../deploy/enve160b/
git checkout gh-pages
cp -r ../../enve160b/__site/* .
git add --all
git commit -m "update gh-pages"
git push -u origin gh-pages
cd ../../env160b
