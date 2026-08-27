# Manuscript files

The manuscript uses the exact submitted Elsevier source as its base. Its title,
section order, and equations are retained; reviewer-requested corrections are
applied in place.

- `main.tex`: blue-highlighted revision of the submitted source.
- `main_clean.tex`: clean wrapper for the same `main.tex` source.
- `ref.bib`: submitted bibliography plus reviewer-requested references.
- `response_to_reviewers.tex`: point-by-point response for AITF-D-26-00044.
- `*_Revised.pdf`: corrected vector figures for Rayleigh, Fanno, and oblique
  mappings. The nozzle and shock-tube scientific figures remain.

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

Accuracy, ablations, timing, uncertainty, edge holdout, dimensional scaling,
gradient-based nozzle inversion, and the 100,000-state workload are reported as
tables. No new benchmark bar chart or dashboard figure is used in the article.
Only the overall workflow remains in the manuscript; five problem-specific TikZ
workflows are documented in `../docs/workflows/`.
