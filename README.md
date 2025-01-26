# Official code implementation of paper: "Revisiting In-context Learning Inference Circuit in Large Language Models"

**This repo contains the official code for the following paper:**

> Hakaze Cho, et al. "Revisiting In-context Learning Inference Circuit in Large Language Models." The Thirteenth International Conference on Learning Representations (ICLR), 2025.

Implemented by Hakaze Cho, the main contributor of the paper.

### Abstract

*In-context Learning (ICL) is an emerging few-shot learning paradigm on Language Models (LMs) with inner mechanisms un-explored. There are already existing works describing the inner processing of ICL, while they struggle to capture all the inference phenomena in large language models. Therefore, this paper proposes a comprehensive circuit to model the inference dynamics and try to explain the observed phenomena of ICL. In detail, we divide ICL inference into 3 major operations: (1) Input Text Encode: LMs encode every input text (demonstrations and queries) into linear representation in the hidden states with sufficient information to solve ICL tasks. (2) Semantics Merge: LMs merge the encoded representations of demonstrations with their corresponding label tokens to produce joint representations of labels and demonstrations. (3) Feature Retrieval and Copy: LMs search the joint representations similar to the query representation on a task subspace, and copy the searched representations into the query. Then, language model heads capture these copied label representations to a certain extent and decode them into predicted labels. The proposed inference circuit successfully captured many phenomena observed during the ICL process, making it a comprehensive and practical explanation of the ICL inference process. Moreover, ablation analysis by disabling the proposed steps seriously damages the ICL performance, suggesting the proposed inference circuit is a dominating mechanism. Additionally, we confirm and list some bypass mechanisms that solve ICL tasks in parallel with the proposed circuit.*

### Summary figure

![Summary figure](https://s2.loli.net/2025/01/26/vXt2VD1iYQ7rJIZ.png)

*The 3-phase inference diagram of ICL. Step 1: LMs encode every input text into representations, Step 2: LMs merge the encoded text representations of demonstrations with their corresponding label semantics, Step 3: LMs retrieve merged label-text representations similar to the encoded query, and copy the retrieved representations into the query representation.*

## Get Started

### 0. Requirement

1. 1 GPU with CUDA and more than 40GB VRAM with CUDA are strongly recommended to run all the experiments.
2. Network connection to `huggingface` is needed to download the pre-trained model.
3. `Anaconda` or `Miniconda` is needed.

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

We use Jupyter notebooks to implement all the experiments descirbed in the paper. We index these notebooks here with the corresponding result figures in the paper, and leave the detailed experiment instructions in each notebook.

### Figure 2: Left and Middle (`Experiments/Exp1_Kernel_Alignment.ipynb`)

This experiment is to calculate the kernel alignment between the ICL hidden states and the sentence embedding. Control the parameters differently will make you get the Fig. 2 Left (by `ICL_selected_token_type`) and Middle (by `k`).

### Figure 2: Right (`Experiments/Exp1_Query_PPL.ipynb`)

This experiment is to calculate the LM loss of the query on the ICL model, to get the x-axis of Fig. 2 Right. Augmented with the individual kernel alignment data from the results of `Exp1_Kernel_Alignment.ipynb`, you can get the Fig. 2 Right.

## Citation

If you want to cite this work, please use the following BibTeX entry:

```
@inproceedings{cho2025revisiting,
  title={Revisiting In-context Learning Inference Circuit in Large Language Models},
  author={Cho, Hakaze and Kato, Mariko and Sakai, Yoshihiro and Inoue, Naoya},
  booktitle={The Thirteenth International Conference on Learning Representations},
  url={https://openreview.net/forum?id=xizpnYNvQq},
  year={2025}
}
```