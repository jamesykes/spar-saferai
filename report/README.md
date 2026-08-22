# Report

`main.tex` is the current manuscript. Its figures and generated tables are kept in the adjacent `figures/` and `tables/` directories.

## Build

Build from this directory:

```bash
latexmk -pdf main.tex
```

If `latexmk` is unavailable, use:

```bash
pdflatex main.tex
biber main
pdflatex main.tex
pdflatex main.tex
```

The manuscript uses `biblatex`; a full build therefore requires Biber rather than BibTeX.
