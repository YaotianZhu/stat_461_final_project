# stat_461_final_project
This repository provides reproducible implementations of seven representative methods that integrate Large Language Models (LLMs) into graph learning for node classification. Based on the survey by Li et al. (2024), we benchmark models across four major paradigms: **LLM as Predictor**, **LLM as Enhancer**, **LLM for Alignment**, and **LLM as Annotator**.

## 🚀 Implemented Methods

### 1. LLM as Predictor
- **GraphGPT**: Graph instruction tuning for large language models, enabling LLMs to understand and reason over graph structures
- **InstructGLM**: Demonstrates that language is all a graph needs, leveraging pure language understanding for graph learning tasks

### 2. LLM as Enhancer  
- **OFA**: Unifying architectures, tasks, and modalities through a simple sequence-to-sequence learning framework for multimodal understanding
- **TAPE**: Uses LLM-generated explanations as features for text-attributed graphs, enhancing graph representation learning

### 3. LLM for Alignment
- **G2P2**: Uses contrastive learning and soft prompt tuning to align node-level graph embeddings and language model representations for graph-enhanced node classification
- **GRAD**: Graph-aware asymmetric distillation framework that transfers GNN teacher knowledge into lightweight language models via soft label matching on textual graphs

### 4. LLM as Annotator
- **LLMGNN**: Label-free node classification on graphs with large language models, using LLMs to generate pseudo-labels for unlabeled nodes

## 📊 Benchmarking

All methods are evaluated on two widely-used datasets:
- **Cora**: Citation network dataset
- **OGBN-Arxiv**: Large-scale academic paper citation network

The `baseline/` directory contains baseline results for comparison across different datasets including Cora, OGBN-Arxiv.

## 🏗️ Repository Structure

```
final_code/
├── LLM_as_Predictor/     # GraphGPT, InstructGLM implementations
├── LLM_as_Enhancer/      # OFA, TAPE implementations  
├── LLM_as_Alignment/     # G2P2, GRAD implementations
├── LLM_as_Annotator/     # LLMGNN implementation
└── baseline/             # Baseline results for comparison
```

## 📈 Results

Preliminary results demonstrate that all LLM-GNN integrated methods outperform strong GCN baselines, confirming the potential of textual augmentation and language understanding in graph learning. However, the degree of improvement varies across methods, suggesting that the design of integration mechanisms plays a critical role.

## 🔬 Experimental Framework

Each method is adapted to a consistent backbone and input format to ensure fair comparison across the four paradigms. Detailed performance comparisons and ablation analyses are available in the respective method directories.

## 📝 Disclaimer

This repository is created for educational purposes as part of a course assignment. All implemented methods are reproductions of existing research works. We acknowledge and respect the original contributions of the authors.

## 📚 Citations

This work is based on the following research papers:

### Survey Paper
```bibtex
@inproceedings{ijcai2024p898,
  title     = {A Survey of Graph Meets Large Language Model: Progress and Future Directions},
  author    = {Li, Yuhan and Li, Zhixun and Wang, Peisong and Li, Jia and Sun, Xiangguo and Cheng, Hong and Yu, Jeffrey Xu},
  booktitle = {Proceedings of the Thirty-Third International Joint Conference on Artificial Intelligence (IJCAI-24), Survey Track},
  pages     = {8123--8131},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},
  editor    = {Kate Larson},
  year      = {2024},
  month     = aug,
  note      = {Survey Track},
  doi       = {10.24963/ijcai.2024/898},
}
```

### LLM as Predictor
- **GraphGPT**: 
```bibtex
@inproceedings{GraphGPT,
  title={Graphgpt: Graph instruction tuning for large language models},
  author={Tang, Jiabin and Yang, Yuhao and Wei, Wei and Shi, Lei and Su, Lixin and Cheng, Suqi and Yin, Dawei and Huang, Chao},
  booktitle={Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval},
  pages={491--500},
  year={2024}
}
```
- **InstructGLM**: 
```bibtex
@article{InstructGLM,
  title={Language is all a graph needs},
  author={Ye, Ruosong and Zhang, Caiqi and Wang, Runhui and Xu, Shuyuan and Zhang, Yongfeng},
  journal={arXiv preprint arXiv:2308.07134},
  year={2023}
}
```

### LLM as Enhancer
- **OFA**: 
```bibtex
@inproceedings{wang2022ofa,
  title={Ofa: Unifying architectures, tasks, and modalities through a simple sequence-to-sequence learning framework},
  author={Wang, Peng and Yang, An and Men, Rui and Lin, Junyang and Bai, Shuai and Li, Zhikang and Ma, Jianxin and Zhou, Chang and Zhou, Jingren and Yang, Hongxia},
  booktitle={International conference on machine learning},
  pages={23318--23340},
  year={2022},
  organization={PMLR}
}
```
- **TAPE**: 
```bibtex
@article{he2023explanations,
  title={Explanations as features: Llm-based features for text-attributed graphs},
  author={He, Xiaoxin and Bresson, Xavier and Laurent, Thomas and Hooi, Bryan and others},
  journal={arXiv preprint arXiv:2305.19523},
  volume={2},
  number={4},
  pages={8},
  year={2023}
}
```

### LLM for Alignment  
- **G2P2**: 
```bibtex
@article{wen2023g2p2,
  title={Prompt Tuning on Graph-augmented Low-resource Text Classification},
  author={Wen, Zhihao and Fang, Yuan},
  journal={arXiv preprint arXiv:2307.10230},
  year={2023}
}
```
- **GRAD**: 
```bibtex
@inproceedings{mavromatis2023grad,
  title={Train Your Own GNN Teacher: Graph-Aware Distillation on Textual Graphs},
  author={Mavromatis, Costas and Ioannidis, Vassilis N and Wang, Shen and others},
  booktitle={Proceedings of the 40th International Conference on Machine Learning (ICML)},
  year={2023}
}
```

### LLM as Annotator
- **LLMGNN**: 
```bibtex
@misc{chen2024labelfree,
  title= {Label-free Node Classification on Graphs with Large Language Models (LLMs)},
  author= {Chen, Zhikai and Mao, Haitao and Wen, Hongzhi and Han, Haoyu and Jin, Wei and Zhang, Haiyang and Liu, Hui and Tang, Jiliang},
  booktitle = {Proceedings of the International Conference on Learning Representations (ICLR)},
  year= {2024}
}
```

## 🙏 Acknowledgments

We thank the original authors of all the methods implemented in this repository for their valuable contributions to the field of graph learning and large language models.
