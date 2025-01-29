# Revisiting In-context Learning Inference Circuit in Large Language Models

<p align="center">
  <a href="https://www.hakaze-c.com/">Hakaze Cho</a>, et al.
  <br>
  <a href="https://arxiv.org/abs/2410.04468">arXiv</a> •
  <a href="https://openreview.net/forum?id=xizpnYNvQq">OpenReview</a> •
  <a href="https://huggingface.co/papers/2410.04468">HuggingFace</a>
  <br>
  <br>
  <a href="https://github.com/hc495/ICL_Circuit/blob/master/LICENSE"><img alt="Static Badge" src="https://img.shields.io/badge/license-MIT-yellow?style=flat&link=https%3A%2F%2Fgithub.com%2Fhc495%2FICL_Circuit%2Fblob%2Fmaster%2FLICENSE"></a>
  <a href="https://openreview.net/forum?id=xizpnYNvQq"><img src="https://img.shields.io/badge/ICLR_2025-Accepted-blue?link=https%3A%2F%2Fopenreview.net%2Fforum%3Fid%3DxizpnYNvQq"></a>
  <a href="https://arxiv.org/abs/2410.04468"><img alt="Static Badge" src="https://img.shields.io/badge/arXiv-2410.04468-red?style=flat&link=https%3A%2F%2Farxiv.org%2Fabs%2F2410.04468"></a>
</p>

**This repo contains the official code for the following paper published at ICLR 2025:**

> Hakaze Cho, et al. **"Revisiting In-context Learning Inference Circuit in Large Language Models."** *The Thirteenth International Conference on Learning Representations (ICLR)*, 2025.

Implemented by [Hakaze Cho](https://www.hakaze-c.com/), the primary contributor of the paper.

## Overview

### Abstract

*In-context Learning (ICL) is an emerging few-shot learning paradigm on Language Models (LMs) with inner mechanisms un-explored. There are already existing works describing the inner processing of ICL, while they struggle to capture all the inference phenomena in large language models. Therefore, this paper proposes a comprehensive circuit to model the inference dynamics and try to explain the observed phenomena of ICL. In detail, we divide ICL inference into 3 major operations: (1) Input Text Encode: LMs encode every input text (demonstrations and queries) into linear representation in the hidden states with sufficient information to solve ICL tasks. (2) Semantics Merge: LMs merge the encoded representations of demonstrations with their corresponding label tokens to produce joint representations of labels and demonstrations. (3) Feature Retrieval and Copy: LMs search the joint representations similar to the query representation on a task subspace, and copy the searched representations into the query. Then, language model heads capture these copied label representations to a certain extent and decode them into predicted labels. The proposed inference circuit successfully captured many phenomena observed during the ICL process, making it a comprehensive and practical explanation of the ICL inference process. Moreover, ablation analysis by disabling the proposed steps seriously damages the ICL performance, suggesting the proposed inference circuit is a dominating mechanism. Additionally, we confirm and list some bypass mechanisms that solve ICL tasks in parallel with the proposed circuit.*

### Summary figure

<p align="center">
<img src="https://s2.loli.net/2025/01/26/vXt2VD1iYQ7rJIZ.png" width="60%" />
</p>

*The 3-phase inference diagram of ICL. Step 1: LMs encode every input text into representations, Step 2: LMs merge the encoded text representations of demonstrations with their corresponding label semantics, Step 3: LMs retrieve merged label-text representations similar to the encoded query, and copy the retrieved representations into the query representation.*


## Set Up

### 0. Requirement

1. A GPU with more than 40GB VRAM and CUDA (Ver. `12.4` recommended) are strongly required to run all the experiments.
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

We use our own-developed library [`StaICC`](https://github.com/hc495/StaICC) to form the ICL-styled inputs. You can install it by: `pip install StaICC` or use the accompanying path `StaICC` in this repo (`git submodule update --init`).

**Image Installation (backup)**

We deeply understand that setting up an environment can be a quite tedious process, and errors may occur at any stage. Therefore, we have made the image of our experimental environment publicly available. You can download it [here](https://drive.google.com/file/d/1rOYz4h-jqEQqsivWuzkS6CPoaVjpTdFe/view?usp=sharing) and `tar -xzf icl_circuit.tar.gz -C ~/anaconda3/envs/icl_circuit`.

### 3. Make Sure Your Working Directory is the Root Directory of the Project

You need to ensure that your working directory is set to the root directory of the project, i.e., the same directory as `README.md`, even if you open a Jupyter notebook from the `Experiments` folder.

We provide a default `os.chdir()` method in every notebook, you should use it to move the working directory to the root directory.

## Experiments

Due to the fact that this paper consists of many relatively independent experiments, we use Jupyter notebooks in the `Experiments` folder to implement all the experiments descirbed in the main body (Appendix experiments will be released later). We index these notebooks here with the corresponding result figures/tables in the paper, and leave the detailed experiment instructions in each notebook.

| Index | Notebook | Result Figure | Description |
| :---: | :---: | :---: | :---: |
| 1 | `Exp1_Kernel_Alignment.ipynb` | Fig. 2 (Left, Middle) | This experiment is to calculate the kernel alignment between the ICL hidden states and the sentence embedding. Control the parameters differently will make you get the Fig. 2 Left (by `ICL_selected_token_type`) and Middle (by `k`). |
| 2 | `Exp1_Query_PPL.ipynb` | Fig. 2 (Right) | This experiment is to calculate the LM loss of the query on the ICL model, to get the x-axis of Fig. 2 Right. Augmented with the individual kernel alignment data from the results of `Exp1_Kernel_Alignment.ipynb`, you can get the Fig. 2 Right. |
| 3 | `Exp2_Centroid_Classifier.ipynb` | Fig. 3, Fig. 5 (Right) | This experiment is to train centroid classifiers on the ICL hidden states, then test the accuracies to get whether the information in the hidden states is sufficient for ICL task. Also, by controlling the selection of different hidden states, we can conduct a controlled experiment as shown in Fig. 5 (Right). |
| 4 | `Exp2_ICL_Feature_Similarity.ipynb` | Fig. 4 | This experiment is to calculate the similarity between the hidden states of the forerunner tokens, to directly get the Fig. 4. |
| 5 | `Exp3_Kernel_Alignment_Across_s_and_y.ipynb` | Fig. 5 (Left) | This experiment is to calculate the kernel alignment of the hidden states of forerunner tokens and the label token in the next layer to get the Fig. 5 (Left). |
| 6 | `Exp3_Forerunner_Token_Head_Counting.ipynb` | Fig. 5 (Middle), Fig. 22 - 25 | This experiment is to count the number of forerunner token heads in each layer, and also the maximum copy magnitude. |
| 7 | `Exp4_Induction_Head_Counting.ipynb` | Fig. 6 (Left, Middle) | This experiment is to count the number of induction heads and correct induction heads in each layer, also calculate the accuracy based on various induction. |

--- Not completed yet ---

We are regretful for an iterative release of the codes. We have only one person to maintain the project, and organizing the experiment codes are quite time-consuming. We will release the rest of the experiments as soon as possible. The next release is scheduled to be on 2025/01/31.

### Parameter `dataset_index`

In the experiments, we use the `dataset_index` parameter to control the dataset used in the experiments, defined as follows:

- `0`: `SST-2`
- `1`: `MR` (`rotten_tomatoes`)
- `2`: `FP` (`financial_phrasebank`)
- `3`: `SST5`
- `4`: `TREC`
- `5`: `AGNews`
- `7`: `TEE` (only used in the control experiments of Fig. 2 (Left, Middle))

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