<p  align="center">
  <img src='logo.png' width='200'>
</p>
<h1 align="center">Shape Happens</h1>

<div align="center">

  <a href="">[![Arxiv](https://img.shields.io/badge/Arxiv-2510.01025-red?style=flat-square&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2510.01025)</a>
  <a href="">[![License](https://img.shields.io/github/license/UKPLab/arxiv2025-shape-happens)](https://opensource.org/licenses/Apache-2.0)</a>
  <a href="">[![Python Versions](https://img.shields.io/badge/Python-3.12-blue.svg?style=flat&logo=python&logoColor=white)](https://www.python.org/)</a>
  
</div>

This is the official repository of the paper "_Shape Happens: Automatic Feature Manifold Discovery in LLMs_". It contains our implementation of Supervised Multidimensional Scaling (SMDS), as well as the scripts necessary to record activations, score different manifolds, and some utilities to plot the results.

><details>
><summary> <b>Abstract</b> </summary>
>The linear representation hypothesis states that language models (LMs) encode concepts as directions in their latent space, forming organized, multidimensional manifolds. Prior efforts focused on discovering ad-hoc geometries and numerical features, and thus lack generalization for manifold structure discovery. We introduce Supervised Multi-Dimensional Scaling (SMDS), a model-agnostic method to automatically discover feature manifolds. In this paper, we apply it to temporal reasoning as a case study, and find that features form geometric structures such as circles, lines, and clusters. This reveals many insights on these structures: they consistently reflect the properties of the concepts they represent; are stable across model families and sizes; actively support reasoning in models; dynamically reshape in response to context changes. Together, our findings shed light on the functional role of feature manifolds, supporting a model of entity-based reasoning in which LMs encode and transform structured representations.
></details></p>

Contact person: [Federico Tiblias](mailto:federico.tiblias@tu-darmstadt.de) 

[UKP Lab](https://www.ukp.tu-darmstadt.de/) | [TU Darmstadt](https://www.tu-darmstadt.de/
)

Don't hesitate to send us an e-mail or report an issue, if something is broken or if you have further questions.


## Getting Started

Simply create a conda environment and install dependencies:

  ```bash
  conda env create -f environment.yml
  conda activate shape-happens
  ```

## Usage

We offer scripts to automatically record activations, score different hypothesis manifolds, and run interventions. First, activations for a given model on a given dataset should be recorded. Then, after the activations have been stored on disk, scores and interventions can be performed.

```bash
python record_activations.py --config <filename> 
```

- `--config <filename>`: specifies the location of a config file listing all configurations to evaluate and record. See `configs/activations.yaml` for an example;

```bash
python compute_scores.py --config <filename> 
```

- `--config <filename>`: specifies the location of a config file listing all manifold configurations to score. See `configs/scoring.yaml` for an example;
- `--save_path <filename>`: Optional. Specifies a filename where the summary scores are stored. The default path is `results/scores/combined_scores.csv`;
- `--overwrite`: Optional. If passed, overwrites existing scoring results. All manifold scores are stored in `results/scores`;

```bash
python record_interventions.py --config <filename> 
```

- `--config <filename>`: specifies the location of a config file listing all configurations to evaluate and record. See `configs/interventions.yaml` for an example;

## Cite

Please use the following citation:

```
@InProceedings{TODO:ARR2025,
  author    = {Smith, John},
  title     = {My Paper Title},
  booktitle = {Proceedings of the 20XX Conference on XXXX},
  month     = mmm,
  year      = {20xx},
  address   = {Gotham City, USA},
  publisher = {Association for XXX},
  pages     = {XXXX--XXXX},
  url       = {http://xxxx.xxx}
}
```

## Disclaimer

> This repository contains experimental software and is published for the sole purpose of giving additional background details on the respective publication. 
