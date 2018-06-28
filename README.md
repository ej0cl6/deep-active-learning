## Deep Active Learning

Python implementations of the following active learning algorithms:

- Random Sampling
- Least Confidence [1]
- Margin Sampling [1]
- Entropy Sampling [1]
- Uncertainty Sampling with Dropout Estimation [2]
- Bayesian Active Learning Disagreement [2]
- K-Means Sampling [3]
- K-Centers Greedy [3]
- Core-Set [3]
- Adversarial - Basic Iterative Method
- Adversarial - DeepFool [4]

### Prerequisites 
- numpy            1.14.3
- scipy            1.1.0
- pytorch          0.4.0
- torchvision      0.2.1
- scikit-learn     0.19.1
- ipdb             0.11

### Usage 

    $ python run.py

### Reference

[1] A New Active Labeling Method for Deep Learning, IJCNN, 2014

[2] Deep Bayesian Active Learning with Image Data, ICML, 2017

[3] Active Learning for Convolutional Neural Networks: A Core-Set Approach, ICLR, 2018

[4] Adversarial Active Learning for Deep Networks: a Margin Based Approach, arXiv, 2018
