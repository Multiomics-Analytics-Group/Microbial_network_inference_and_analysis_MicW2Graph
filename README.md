
# Microbial Network Inference and Analysis - MicW2Graph
This repository contains scripts to infer and analyze **microbial association networks (MANs)** for the [MicW2Graph project][MicW2Graph].

## Microbial association networks
MANs are weighted and undirected networks, defined as *G = (V, E)*, where *V* is a set of nodes and *E* is a set of edges. Nodes in these networks are Operational Taxonomic Units at a specific taxonomic level, while edges indicate substantial co-presence (positive interaction) or mutual exclusion (negative interaction) trends in microorganism abundances across samples. Weights in MANs correspond to association values among species defined by the inference method, and there is an edge between two nodes if this number is greater than or equal to a given cutoff *t*.

## Network Inference
In this project, we selected the Correlation inference for [Compositional data through Lasso (CCLasso)][CCLasso] method. This approach estimates positive definite Pearson correlations from log-ratio variances using a latent variable model based on least squares with an L1 penalty. The parameters required for the CCLasso algorithm included choices for zero-treatment, normalization, sparsification, and dissimilarity methods. In this study, the CCLasso MANs were constructed using the *pseudo count zero* approach for zero-treatment, the *total sum scaling* technique for normalization, the  *signed* strategy for converting associations into dissimilarities, and varying sparsification thresholds.

MANs were inferred from the abundance tables of each sub-biome and experiment type, changing the association cutoff from 0.1 to 0.7 with increments of 0.05. Network inference was conducted using the [NetCoMi][NetCoMi] R package v1.1 and R v4.3.1. Before creating the MANs, abundance, sample, and taxonomic tables were transformed into a Phyloseq object to facilitate data manipulation within NetCoMi. Cytoscape v3.10.1 and RCy3 v2.6 were used for network visualization.

## Network Analysis
Identifying functional microbial communities was the primary research focus of this study, so the optimal association threshold was determined based on community-structure metrics, specifically modularity and average clustering coefficient (ACC). The top three networks with the highest modularity were identified for each sub-biome and experiment type, and from these, the one with the highest ACC was chosen. Furthermore, the Louvain community detection algorithm was used to assign clusters to the nodes of networks with the optimal cutoff. All network metrics were computed using the [NetworkX][NetworkX] v3.3 Python library and summary plots were created with Matplotlib.

## Getting Started
To get started, you can clone this repository and explore the available scripts and data. Ensure that you have the necessary packages and dependencies installed. [Poetry][Poetry] was used for the management of python libraries of the Network Analysis scripts. To create a Python virtual environment with libraries and dependencies required for this project, you should clone this GitHub repository, open a terminal, move to the Network Analysis folder, and run the following commands:

```bash
# Create the Python virtual environment 
$ poetry install

# Activate the Python virtual environment 
$ poetry shell
```

You can find a detailed guide on how to use pipenv [here][Poetry-doc].

## **Credits and Contributors**
- Developed by [Sebastián Ayala Ruano][myweb] under the supervision of [Dr. Alberto Santos][Alberto], head of the [Multiomics Network Analytics Group (MoNA)][Mona] at the [Novo Nordisk Foundation Center for Biosustainability (DTU Biosustain)][Biosustain].
- MicW2Graph was built for the thesis project from the [MSc in Systems Biology][sysbio] at the MoNA group.
- The data for this project was obtained from [Mgnify][Mgnify], using the scripts available in this [GitHub repository][retrieve_info_mgnify].

## **Contact**
If you have comments or suggestions about this project, you can [open an issue][issues] in this repository.

[MicW2Graph]: https://github.com/Multiomics-Analytics-Group/MicW2Graph
[CCLasso]: https://github.com/huayingfang/CCLasso
[NetCoMi]: https://github.com/stefpeschel/NetCoMi
[NetworkX]: https://networkx.org/
[Poetry]: https://python-poetry.org/
[Poetry-doc]: https://python-poetry.org/docs/basic-usage/
[sysbio]: https://www.maastrichtuniversity.nl/education/master/systems-biology
[myweb]: https://sayalaruano.github.io/
[Alberto]: https://orbit.dtu.dk/en/persons/alberto-santos-delgado
[Mona]: https://multiomics-analytics-group.github.io/
[Biosustain]: https://www.biosustain.dtu.dk/
[retrieve_info_mgnify]: https://github.com/Multiomics-Analytics-Group/Retrieve_info_MGnifyAPI
[Mgnify]: https://www.ebi.ac.uk/metagenomics/api/latest/
[issues]: https://github.com/Multiomics-Analytics-Group/Microbial_network_inference_and_analysis_MicW2Graph/issues/new
