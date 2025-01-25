# Official code implementation of paper: "Revisiting In-context Learning Inference Circuit in Large Language Models"

This repo contains the official code for the following paper:

> Hakaze Cho, et al. "Revisiting In-context Learning Inference Circuit in Large Language Models." The Thirteenth International Conference on Learning Representations (ICLR), 2025.

Implemented by Hakaze Cho, the main contributor of the paper.

### Abstract

In-context Learning (ICL) is an emerging few-shot learning paradigm on Language Models (LMs) with inner mechanisms un-explored. There are already existing works describing the inner processing of ICL, while they struggle to capture all the inference phenomena in large language models. Therefore, this paper proposes a comprehensive circuit to model the inference dynamics and try to explain the observed phenomena of ICL. In detail, we divide ICL inference into 3 major operations: (1) Input Text Encode: LMs encode every input text (demonstrations and queries) into linear representation in the hidden states with sufficient information to solve ICL tasks. (2) Semantics Merge: LMs merge the encoded representations of demonstrations with their corresponding label tokens to produce joint representations of labels and demonstrations. (3) Feature Retrieval and Copy: LMs search the joint representations similar to the query representation on a task subspace, and copy the searched representations into the query. Then, language model heads capture these copied label representations to a certain extent and decode them into predicted labels. The proposed inference circuit successfully captured many phenomena observed during the ICL process, making it a comprehensive and practical explanation of the ICL inference process. Moreover, ablation analysis by disabling the proposed steps seriously damages the ICL performance, suggesting the proposed inference circuit is a dominating mechanism. Additionally, we confirm and list some bypass mechanisms that solve ICL tasks in parallel with the proposed circuit.

## Get Started

**Notice: GPUs with CUDA are needed to run all the experiments. We recommend using a machine with at least 1 GPU with more than 40GB VRAM.**

### 1. Clone the repository

```bash
git clone https://github.com/hc495/ICL_Circuit.git
```

### 2. Installation

```bash
conda env create -f environment.yaml
conda activate icl_circuit
```

We use our own-developed library `StaICC`(https://github.com/hc495/StaICC) to form the ICL-styled inputs. You can install it by:

```bash
pip install StaICC
```

or use our accompanying library `StaICC`.

### 3. Make Sure Your Working Directory is the Root Directory of the Project

You need to ensure that your working directory is set to the root directory of the project, i.e., the same directory as `README.md`, even if you open a Jupyter notebook from the `Experiments` folder.

We provide a default `os.chdir()` method in every notebook, you should use it to move the working directory to the root directory.

## Experiments

We use Jupyter notebooks to implement all the experiments descirbed in the paper. We index these notebooks here, and leave the detailed experiment instructions in each notebook.

--TBD--

## Citation

If you find this repository helpful, please cite the following paper:

```
@inproceedings{cho2025revisiting,
  title={Revisiting In-context Learning Inference Circuit in Large Language Models},
  author={Cho, Hakaze},
  booktitle={The Thirteenth International Conference on Learning Representations},
  url={https://openreview.net/forum?id=xizpnYNvQq},
  year={2025}
}
```