# *MaxEnt-ARL*: [Mitigating Information Leakage in Image Representations: A Maximum Entropy Approach](https://arxiv.org/abs/arXiv:1904.05514)

By Proteek Chandan Roy and Vishnu Naresh Boddeti

### Introduction

This code archive includes the Python implementation of intrinsic dimensionality estimation for image representation, and the proposed dimensionality reduction method -- DeepMDS. Our work, *ID&DeepMDS*, addressed two basic but fundamental questions in representation learning, i.e., its intrinsic dimensionality and if we can find a mapping between the ambient and intrinsic space while maintaining the discriminative capability of the representation. The proposed *ID&DeepMDS* is able to estimate intrinsic dimensionality for a given image representation, and then transform the ambient space to the intrinsic space based on an unsupervised DNN dimensionality reduction method under the framework of multidimensional scaling.

### Citation

If you think **MaxEnt-ARL** is useful to your research, please cite:

    @article{roy2019mitigating,
        title={Mitigating Information Leakage in Image Representations: A Maximum Entropy Approach},
        author={Roy, Proteek Chandan and Boddeti, Vishnu Naresh},
        booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
        year={2019}
    }

### Usage
    python main.py

By default this will run the experiment on CIFAR-100 dataset as described in the paper. Note that it will generate multiple runs of the trade-off between utility and privacy. The non-dominated solutions across the multiple runs provides the final trade-off front as reported in the paper.

In order to run the experiments on the other datasets in the paper, please edit the "main.py" file.