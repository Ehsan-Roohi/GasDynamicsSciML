# Manuscript files

- `main_revised_highlighted.tex`: wrapper that renders all revised text in blue.
- `main_revised_clean.tex`: clean wrapper for the same source.
- `main_revised_body.tex`: single source of truth for both manuscript variants.
- `response_to_reviewers.tex`: point-by-point response for AITF-D-26-00044.

Compile from the repository root so the generated figures resolve correctly:

```bash
mkdir -p manuscript/build
pdflatex -interaction=nonstopmode -halt-on-error -output-directory manuscript/build manuscript/main_revised_highlighted.tex
pdflatex -interaction=nonstopmode -halt-on-error -output-directory manuscript/build manuscript/main_revised_clean.tex
pdflatex -interaction=nonstopmode -halt-on-error -output-directory manuscript/build manuscript/response_to_reviewers.tex
```

