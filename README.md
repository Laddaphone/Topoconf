# TopoConf: Evidential Deep Learning for Graph-Based Mammography

Code for: **Evidence Phenotypes Determine Optimal Uncertainty Channels for Safety-Critical Mammography Triage**
CMBBE: Imaging & Visualization (2026).

## Repository Structure

### Model Code
- `models/topoconf.py` - TopoConfGNN (EDL, returns Dirichlet alphas)
- `models/baselines.py` - Softmax, MC-Dropout, GCN baselines
- `models/backbone.py` - ResNet-18 truncated at layer3
- `models/layers.py` - GAT blocks, gated topology fusion
- `losses.py` - EDL loss (Type II ML + KL regularization)
- `uncertainty.py` - Dirichlet and MC decomposition

### Preprocessing Pipeline
- `preprocessing/pipeline.py` - Full image-to-graph pipeline:
  breast masking (Otsu + largest CC), CLAHE (clip=2.0, grid=8x8),
  patch extraction (128x128, stride=64), TDA features (GUDHI cubical
  persistent homology H0/H1), spatial edges (k-NN k=8),
  topology edges (k=4, cosine > 0.7)
- `preprocessing/load_inbreast.py` - DICOM loader with MONOCHROME1 orientation fix
- `preprocessing/load_cbis.py` - CBIS-DDSM JPEG loader with CSV metadata

### Analysis Scripts
- `analysis/uncertainty_decomposition.py` - Corrected decomposition (pred_ent, exp_ent, MI, vacuity)
- `analysis/triage.py` - FN catch at matched workloads + patient-level Wilcoxon
- `analysis/ablation.py` - Class weight and KL coefficient ablation analysis
- `analysis/subgroup.py` - Clinical subgroup stratification (density, lesion type, subtlety)
- `analysis/bootstrap_ci.py` - Bootstrap CIs + cluster-aware resampling
- `analysis/calibration.py` - ECE, reliability diagrams, sensitivity at fixed specificity
- `analysis/cross_domain.py` - CBIS to INbreast transfer evaluation
- `analysis/node_distribution.py` - Node/edge count statistics

### Reproducibility
- `reproduce_all.py` - Regenerate all tables from raw .npz files
- `generate_figures.py` - Regenerate all figures (see also notebook)
- `verify_splits.py` - Assert zero patient overlap across splits
- `data_utils.py` - Patient-disjoint splitting (50/15/15/20)
- `conformal/conformal_prediction.py` - Split conformal + TACP/EACP adaptive variants
- `requirements.txt` - Pinned dependency versions

## Checkpoints, Results, and Data

Available at: https://huggingface.co/Laddaphone/topoconf-mammography-edl (515 files)
- 137 checkpoints (3 datasets x 6 models x 10 seeds + cross-domain)
- 200 patient ID lists (per-seed train/test manifests)
- All result CSVs, ablation outputs, verification reports
- 8 figures at 600 DPI (PDF + PNG)

## Quick Start

```bash
pip install -r requirements.txt
python reproduce_all.py --results_dir ./posthoc_corrected/ --output_dir ./output/
python verify_splits.py ./posthoc_corrected/
```

## Citation

Douangnouanexay, L. (2026). Evidence Phenotypes Determine Optimal
Uncertainty Channels for Safety-Critical Mammography Triage.
CMBBE: Imaging & Visualization.

## License

MIT
