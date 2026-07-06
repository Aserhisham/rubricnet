#!/bin/bash
# Compilation script for the thesis LaTeX project

# Exit on error
set -e

echo "=== [1/4] Running pdflatex (first pass) ==="
pdflatex -interaction=nonstopmode main.tex

echo "=== [2/4] Running bibtex (citations) ==="
bibtex main

echo "=== [3/4] Running pdflatex (second pass) ==="
pdflatex -interaction=nonstopmode main.tex

echo "=== [4/4] Running pdflatex (third pass) ==="
pdflatex -interaction=nonstopmode main.tex

echo "=== Compilation Complete! main.pdf generated successfully. ==="
