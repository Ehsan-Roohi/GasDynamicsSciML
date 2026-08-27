# Overleaf build instructions

This project is based on the original submitted Elsevier source.

- Set `main.tex` as the Main document for the blue-highlighted revision.
- Set `main_clean.tex` as the Main document for the clean article.
- Set `response_to_reviewers.tex` as the Main document for the response letter.
- Compiler: pdfLaTeX.

`main_clean.tex` is a wrapper around `main.tex`; the two article versions
therefore cannot drift in content. Run BibTeX after the first article compile,
then run pdfLaTeX twice. The response letter does not require BibTeX.

`generated_timing_values.tex` is required by all three main documents. It is
generated from the unified single-process timing CSVs and keeps Tables 9 and 12,
the abstract, and the response letter numerically synchronized.

The article contains only the overall framework workflow diagram. The five
problem-specific TikZ diagrams are maintained in the public repository at
<https://github.com/Ehsan-Roohi/GasDynamicsSciML/tree/main/docs/workflows>.

Machine-readable values cited in the manuscript are included under `evidence/`
in the supplied ZIP and are also maintained in the public repository.
