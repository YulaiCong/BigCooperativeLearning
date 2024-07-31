# Big Cooperative Learning

The official code for the paper "[Big Cooperative Learning](https://arxiv.org/abs/2312.11926)" by Yulai Cong.

## Abstract
Cooperation plays a pivotal role in the evolution of human intelligence; moreover, it also underlies the recent revolutionary advancement of artificial intelligence (AI) that is driven by foundation models. 
Specifically, we reveal that the training of foundation models can be interpreted as a form of big cooperative learning (abbr. big learning), where massive learning individuals/tasks cooperate to approach the unique essence of data from diverse perspectives of data prediction, leveraging a universal model. 
The presented big learning therefore unifies most training objectives of foundation models within a consistent framework, where their underlying assumptions are exposed simultaneously.
We design tailored simulations to demonstrate the principle of big learning, based on which we provide learning-perspective justifications for the successes of foundation models, with interesting side-products. 
Furthermore, we reveal that big learning is a new dimension for upgrading conventional machine learning paradigms, valuable for endowing reinvigorations to associated applications;
as an illustrative example, we propose the BigLearn-GAN, which is a novel adversarially-trained foundation model with versatile data sampling capabilities.

## Directory Explanation
```
filetree 
├── BL_vs_deepClustering
│  ├── dataset
│  ├── function.py
│  ├── method.py
|  ├── main_BigLearnEM.py
├── dataset
├── function.py
├── method.py
├── main_biglearnEM_vs_EM_v1.ipynb
├── main_realworld_clustering.ipynb
```

## Usage

### 1. Joint-EM vs BigLearn-EM on the 25-GMM simulation
- Run `main_biglearnEM_vs_EM_v1.ipynb`

### 2. Real-World Clustering Experiments
- Prepare the dataset: The Glass, Letter, Pendigits, and Vehicle datasets are given in the Directory 'dataset'. [Click here to download other datasets](https://www.csie.ntu.edu.tw/~cjlin/libsvm/index.html)
- Run `main_realworld_clustering.ipynb`.
- Note different hyperparameter settings should be used for different datasets. If 'out of memory' occurs, modify 'data_size' or 'chunk_size'. For datasets without an official testing set, set 'split_data' to 'True' to randomly select data samples to form one.
  
### 3. BigLearn-EM vs Deep Clustering Methods on the FashionMNIST dataset
- Download the [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) dataset and place it into the `BL_vs_deepClustering/dataset` directory
- Run `BL_vs_deepClustering/main_BigLearnEM.py`
- Run `python BL_vs_deepClustering/main_BigLearnEM.py --device cuda:1 --Niter 70 --NITnei 5 --eps 0.05 --P1 0.4 --P2 0.5 --out_dir [path for training weights] --txt_dir [path for training records]`
- The experiment is conducted based on the [code](https://github.com/JinyuCai95/EDESC-pytorch) of the CVPR22 paper ["Efficient Deep Embedded Subspace Clustering"](https://openaccess.thecvf.com/content/CVPR2022/papers/Cai_Efficient_Deep_Embedded_Subspace_Clustering_CVPR_2022_paper.pdf).

## Reference
Please consider citing our paper if you refer to this code in your research.
```
@misc{cong2023big,
      title={Big Cooperative Learning}, 
      author={Yulai Cong},
      year={2024},
      eprint={2312.11926},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

​     
