#!/usr/bin/env bash

sphinx-build -b html doc/source doc/build/html
sphinx-build -b latex doc/source doc/build/latex
sphinx-build -b man doc/source doc/build/man
cd doc/build/latex
latexmk -pdf fdasrsf.tex
cd ..