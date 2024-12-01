# RecRanker: Instruction Tuning Large Language Model as Ranker for Top-k Recommendation





![Static Badge](https://img.shields.io/badge/Paper-PDF-blue?style=flat&link=https%3A%2F%2Farxiv.org%2Fpdf%2F2312.16018v2.pdf)


This is the PyTorch implementation for RecRanker model.


> **RecRanker: Instruction Tuning Large Language Model as Ranker for Top-k Recommendation.**  
Sichun Luo, Bowei He, Haohan Zhao, Yinya Huang, Aojun Zhou, Zongpeng Li, Yuanzhang Xiao, Mingjie Zhan, Linqi Song.
*ACM Transactions on Information Systems (TOIS) 2024*


> 
---

## Introduction
In this paper, we introduce **RecRanker**, tailored for instruction tuning LLM to serve as the *Ranker* for top-*k* *Rec*ommendations. Specifically, we introduce importance-aware sampling, clustering-based sampling, and penalty for repetitive sampling for sampling high-quality, representative, and diverse users as training data. To enhance the prompt, we introduce a position shifting strategy to mitigate position bias and augment the prompt with auxiliary information from conventional recommendation models, thereby enriching the contextual understanding of the LLM. Subsequently, we utilize the sampled data to assemble an instruction-tuning dataset with the augmented prompt comprising three distinct ranking tasks: pointwise, pairwise, and listwise rankings. We further propose a hybrid ranking method to enhance the model performance by ensembling these ranking tasks.

![Training](/fig/f17.png)
![Inference](/fig/f16.png)

## Citation
If you find RecRanker useful in your research or applications, please kindly cite:

> @article{luo2023recranker,  
  title={RecRanker: Instruction Tuning Large Language Model as Ranker for Top-k Recommendation},  
  author={Luo, Sichun and He, Bowei and Zhao, Haohan and Huang, Yinya and Zhou, Aojun and Li, Zongpeng and Xiao, Yuanzhang and Zhan, Mingjie and Song, Linqi},  
  journal={arXiv preprint arXiv:2312.16018},  
  year={2023}  
}

Thanks for your interest in our work!


## Acknowledgement
The structure of this code is largely based on [SELFRec](https://github.com/Coder-Yu/SELFRec) and [MathCoder](https://github.com/mathllm/MathCoder). Thank them for their work.
