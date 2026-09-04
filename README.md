# TopoConf: Evidential Deep Learning for Graph-Based Mammography

Code repository for the paper:
**Evidence Phenotypes Determine Optimal Uncertainty Channels for Safety-Critical Mammography Triage**

Submitted to CMBBE: Imaging & Visualization (2026).

## Key Finding

On digitized-film mammography (CBIS-DDSM), EDL's vacuity signal catches
substantially more false negatives than predictive entropy at matched
review burden. On digital FFDM mammography (INbreast, CMMD), this
advantage vanishes or reverses. The optimal uncertainty channel for
FN triage is dataset-dependent.

## Repository Structure

- models/backbone.py - ResNet-18 truncated at layer3
- models/layers.py - GATBlk, GatedTopoFusion
- models/topoconf.py - TopoConfGNN (EDL, returns Dirichlet alphas)
- models/baselines.py - Softmax/MCDrop/GCN baselines
- losses.py - EDL loss with KL regularization
- uncertainty.py - Dirichlet and MC decomposition
- data_utils.py - Patient-disjoint splitting

## Checkpoints & Results

Available at: https://huggingface.co/Laddaphone/topoconf-mammography-edl

## Citation

Douangnouanexay, L. (2026). Evidence Phenotypes Determine Optimal
Uncertainty Channels for Safety-Critical Mammography Triage.
CMBBE: Imaging & Visualization.

## License

MIT
