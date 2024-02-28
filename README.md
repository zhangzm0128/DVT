# DVT 
### [Paper](https://www.ieee-jas.net/en/article/id/ec6a16fa-d348-417a-af0f-dd734c60439c) | [Project Page](https://github.com/zhangzm0128/DVT) 

> DVT: Dendritic Learning-incorporated Vision Transformer for Image Recognition

> [Zhiming Zhang](https://zhangzm0128.github.io/), [Zhenyu Lei](https://scholar.google.com/citations?user=7Ss6peAAAAAJ&hl=zh-CN&oi=sra), Masaaki Omura, [Hideyuki Hasegawa](https://scholar.google.com/citations?hl=zh-CN&user=Qb2bhzcAAAAJ&view_op=list_works&sortby=pubdate), [Shangce Gao](https://toyamaailab.github.io/)


DVT is a groundbreaking Biomimetic Vision Transformer that combines dendritic learning and Vision Transformer architecture, showcasing superior image recognition performance through biologically inspired structures.

## Overview
![demo](./framework.gif)

DVT is an innovative project introducing a Dendritic Learning-incorporated Vision Transformer, specifically designed for universal image recognition tasks inspired by dendritic neurons in neuroscience. The model's architecture incorporates highly biologically interpretable dendritic learning techniques, enabling DVT to excel in handling complex nonlinear classification problems.

The motivation behind DVT stems from the hypothesis that networks with high biological interpretability in architecture also exhibit superior performance in image recognition tasks. Our experimental results, as outlined in the associated [paper](https://www.ieee-jas.net/en/article/id/ec6a16fa-d348-417a-af0f-dd734c60439c), highlight the substantial improvement achieved by DVT compared to the current state-of-the-art methods on four general datasets.

## Getting Started

### Training
Train the DVT on Nvidia GPU.
```
python main.py --mode train --device cuda --config ./configs/DVT_cifar10.json
```
### Evaluation
Test a model on Nvidia GPU.
```
python main.py --mode test --device cuda --checkpoint ./logs/xxx
```

## Related Projects

Our code is based on [PyTorch](https://github.com/pytorch/pytorch).

## Citing DVT

Zhiming Zhang, Zhenyu Lei, Masaaki Omura, Hideyuki Hasegawa, and Shangce Gao,  “Dendritic learning-incorporated vision transformer for image recognition,” IEEE/CAA Journal of Automatica Sinica, vol. 11, no. 2, pp. 1–3, Feb. 2024. DOI: 10.1109/JAS.2023.123978. 


```bib
@article{zhang2024dendritic,
  author={Zhiming Zhang,Zhenyu Lei,Masaaki Omura,Hideyuki Hasegawa,Shangce Gao},
  title={Dendritic Learning-Incorporated Vision Transformer for Image Recognition},
  journal={IEEE/CAA Journal of Automatica Sinica},  
  year={2024},
  volume={11},
  number={2},
  pages={541-543},
  doi={10.1109/JAS.2023.123978}
}
