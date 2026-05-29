# Report build notes

Build from this directory:

```bash
latexmk -pdf main.tex
```

If `latexmk` is unavailable, use:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

The draft includes copied figures from `../figures/` and generated LaTeX tables in `tables/`.
