# Revisiting In-context Learning Inference Circuit in Large Language Models

<p align="center">
  <a href="https://arxiv.org/abs/2410.04468">arXiv</a> •
  <a href="https://openreview.net/forum?id=xizpnYNvQq">OpenReview</a> •
  <a href="https://huggingface.co/papers/2410.04468">HuggingFace</a>
</p>

**This repo contains the official code for the following paper:**

> Hakaze Cho, et al. **"Revisiting In-context Learning Inference Circuit in Large Language Models."** *The Thirteenth International Conference on Learning Representations (ICLR)*, 2025.

Implemented by [Hakaze Cho](https://www.hakaze-c.com/), the primary contributor of the paper.

## Overview

### Abstract

*In-context Learning (ICL) is an emerging few-shot learning paradigm on Language Models (LMs) with inner mechanisms un-explored. There are already existing works describing the inner processing of ICL, while they struggle to capture all the inference phenomena in large language models. Therefore, this paper proposes a comprehensive circuit to model the inference dynamics and try to explain the observed phenomena of ICL. In detail, we divide ICL inference into 3 major operations: (1) Input Text Encode: LMs encode every input text (demonstrations and queries) into linear representation in the hidden states with sufficient information to solve ICL tasks. (2) Semantics Merge: LMs merge the encoded representations of demonstrations with their corresponding label tokens to produce joint representations of labels and demonstrations. (3) Feature Retrieval and Copy: LMs search the joint representations similar to the query representation on a task subspace, and copy the searched representations into the query. Then, language model heads capture these copied label representations to a certain extent and decode them into predicted labels. The proposed inference circuit successfully captured many phenomena observed during the ICL process, making it a comprehensive and practical explanation of the ICL inference process. Moreover, ablation analysis by disabling the proposed steps seriously damages the ICL performance, suggesting the proposed inference circuit is a dominating mechanism. Additionally, we confirm and list some bypass mechanisms that solve ICL tasks in parallel with the proposed circuit.*

### Summary figure

![Summary figure](https://s2.loli.net/2025/01/26/vXt2VD1iYQ7rJIZ.png)

*The 3-phase inference diagram of ICL. Step 1: LMs encode every input text into representations, Step 2: LMs merge the encoded text representations of demonstrations with their corresponding label semantics, Step 3: LMs retrieve merged label-text representations similar to the encoded query, and copy the retrieved representations into the query representation.*

## Set Up

### 0. Requirement

1. 1 GPU with more than 40GB VRAM and CUDA (12.4 recommended) are strongly required to run all the experiments.
2. Network connection to `huggingface` is needed to download the pre-trained model. And a `huggingface` user token with access to the [`Llama2`](https://huggingface.co/meta-llama/Llama-2-7b) model is recommended to run a part of the experiments.
3. `Anaconda` or `Miniconda` is needed.

### 1. Clone the repository

```bash
git clone https://github.com/hc495/ICL_Circuit.git
```

### 2. Environment Installation

**Direct Installation**

```bash
conda env create -f environment.yaml
conda activate icl_circuit
```

We use our own-developed library [`StaICC`](https://github.com/hc495/StaICC) to form the ICL-styled inputs. You can install it by: `pip install StaICC` or use our accompanying library `StaICC`.

**Image Installation (backup)**

We deeply understand that setting up an environment can be a quite tedious process, and errors may occur at any stage. Therefore, we have made the image of our experimental environment publicly available. You can download it [here](https://drive.google.com/file/d/1rOYz4h-jqEQqsivWuzkS6CPoaVjpTdFe/view?usp=sharing) and `tar -xzf icl_circuit.tar.gz -C ~/anaconda3/envs/icl_circuit`.

### 3. Make Sure Your Working Directory is the Root Directory of the Project

You need to ensure that your working directory is set to the root directory of the project, i.e., the same directory as `README.md`, even if you open a Jupyter notebook from the `Experiments` folder.

We provide a default `os.chdir()` method in every notebook, you should use it to move the working directory to the root directory.

## Experiments

Due to the fact that this paper consists of many relatively independent experiments, we use Jupyter notebooks to implement all the experiments descirbed in the main body (Appendix experiments will be released later). We index these notebooks here with the corresponding result figures/tables in the paper, and leave the detailed experiment instructions in each notebook.

| Index | Notebook | Result Figure | Description |
| :---: | :---: | :---: | :---: |
| 1 | `Exp1_Kernel_Alignment.ipynb` | Fig. 2 (Left, Middle) | This experiment is to calculate the kernel alignment between the ICL hidden states and the sentence embedding. Control the parameters differently will make you get the Fig. 2 Left (by `ICL_selected_token_type`) and Middle (by `k`). |
| 2 | `Exp1_Query_PPL.ipynb` | Fig. 2 (Right) | This experiment is to calculate the LM loss of the query on the ICL model, to get the x-axis of Fig. 2 Right. Augmented with the individual kernel alignment data from the results of `Exp1_Kernel_Alignment.ipynb`, you can get the Fig. 2 Right. |
| 3 | `Exp2_Centroid_Classifier.ipynb` | Fig. 3, Fig. 5 (Right) | This experiment is to train centroid classifiers on the ICL hidden states, then test the accuracies to get whether the information in the hidden states is sufficient for ICL task. Also, by controlling the selection of different hidden states, we can conduct a control experiment as shown in Fig. 5 (Right). |

--- Not completed yet ---

## Citation

If you find this work useful for your research, please cite [our paper](https://openreview.net/forum?id=xizpnYNvQq):

```
@inproceedings{cho2025revisiting,
  title={Revisiting In-context Learning Inference Circuit in Large Language Models},
  author={Cho, Hakaze and Kato, Mariko and Sakai, Yoshihiro and Inoue, Naoya},
  booktitle={The Thirteenth International Conference on Learning Representations},
  url={https://openreview.net/forum?id=xizpnYNvQq},
  year={2025}
}
```