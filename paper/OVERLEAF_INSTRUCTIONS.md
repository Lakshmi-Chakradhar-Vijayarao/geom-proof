# Overleaf Upload Instructions

## Files to upload

Upload ALL of the following into the root of your Overleaf project:

```
main.tex          ← main paper
references.bib    ← bibliography
figures/
  06_central_figure.png
  03_certificate_calibration.png
  05_depth_fraction_overlay.png
  08_ot_certificates.png
  10_conformal_coverage.png
  11_qwen_scale_curve.png
  12_mlp_probe_comparison.png   ← NEW (Exp 12, added in final version)
```

## Compiler setting

Set compiler to: **pdfLaTeX**

## On first compile

If you see errors about missing packages, Overleaf will auto-install them.
The packages used are all standard (amsmath, booktabs, natbib, etc.).

## Page count

Expected: ~11–12 pages in twocolumn 10pt layout (after workshop acceptance upgrades).

## To switch to single-column

Change line 1 from:
  \documentclass[10pt,twocolumn]{article}
to:
  \documentclass[10pt]{article}

## To use NeurIPS / ICML / ICLR style

Replace \documentclass line with the venue template and remove the
\usepackage[margin=1in]{geometry} line. All content is style-agnostic.
