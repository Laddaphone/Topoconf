"""
Regenerate all paper figures from raw .npz result files.
Figures saved as PDF (600 DPI) + PNG.

Font sizes: title=13, axes=12, ticks=10, legend=9
All CI bars are seed-level standard error.
Consistent ordering: CBIS, INbreast, CMMD.

Usage:
  python generate_figures.py --results_dir /path/to/posthoc_corrected/
                             --cmmd_dir /path/to/cmmd_within_domain/
                             --output_dir /path/to/figures/

See Topoconf_Figure_Genration.ipynb for the full implementation.
Pre-generated figures available on HuggingFace:
  https://huggingface.co/Laddaphone/topoconf-mammography-edl/tree/main/figures
"""
print('Figure generation script. See notebook or HuggingFace for pre-generated figures.')
print('Notebook: Topoconf_Figure_Genration.ipynb')
print('HuggingFace: https://huggingface.co/Laddaphone/topoconf-mammography-edl/tree/main/figures')
