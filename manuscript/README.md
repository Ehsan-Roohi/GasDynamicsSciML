# Manuscript files

The revision now uses the exact submitted Elsevier source as its base. Its
title, section order, equations, and scientific figure sequence are retained;
reviewer-requested corrections are applied in place.

- `main.tex`: blue-highlighted revision of the submitted source.
- `main_clean.tex`: clean wrapper for the same `main.tex` source.
- `ref.bib`: submitted bibliography plus reviewer-requested references.
- `response_to_reviewers.tex`: point-by-point response for AITF-D-26-00044.
- `*_Revised.pdf`: corrected vector figures for Rayleigh, Fanno, and oblique
  mappings. The submitted nozzle, shock-tube, and Fanno process figures remain.

Compile from `manuscript/` with pdfLaTeX and BibTeX:

```bash
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex

pdflatex -interaction=nonstopmode -halt-on-error -jobname=main_clean main_clean.tex
bibtex main_clean
pdflatex -interaction=nonstopmode -halt-on-error -jobname=main_clean main_clean.tex
pdflatex -interaction=nonstopmode -halt-on-error -jobname=main_clean main_clean.tex

pdflatex -interaction=nonstopmode -halt-on-error response_to_reviewers.tex
```

Accuracy, ablations, timing, uncertainty, edge holdout, and dimensional scaling
are reported as tables. No new benchmark bar chart or dashboard figure is used
in the article.
