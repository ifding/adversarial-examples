## Adversarial examples


Requirements:

* Python 3
* numpy
* tensorflow (tested with versions 1.2 and 1.4)
* scipy (tested with version 0.19.1)


### Attacks

- **JSMA** ([notebook](notebooks))([code](adversarial_attacks)). 

This part we cite the work of [Papernot et al.](https://github.com/tensorflow/cleverhans).
Default model in the source code is a deep neural network defined in above respository.
This part relies on cleverhans's other files, you my need to install the whole respository for running this code.

- **FGSM** ([notebook](notebooks))([code](adversarial_attacks)). 

We also cite this work from [cleverhans](https://github.com/tensorflow/cleverhans).This tutorial covers how to train a MNIST/CIFAR model using TensorFlow, craft adversarial examples using the fast gradient sign method, and make the model more robust to adversarial examples using adversarial training.

- **CW attack** ([notebook](notebooks))([code](adversarial_attacks)). 

"Towards Evaluating the Robustness of Neural Networks" by Nicholas Carlini and David Wagner, at IEEE Symposium on Security & Privacy, 2017.

CW attack consists of L0 attack,L2 attack and Li attack. In our work, we only test L2 attack.This tutorial covers how to train a MNIST model using TensorFlow, craft adversarial examples using CW attack, and prove that defensive distillation is not robust to adversarial examples.More details in [Nicholas Carlini et al.](https://github.com/carlini/nn_robust_attacks).
 
- **Fast feature fool** ([notebook](notebooks))([code](adversarial_attacks)). 

Test fast feature fool algorithm with MNIST dataset has not been finished yet, there's the source code of [Mopuri et al.](https://github.com/val-iisc/fast-feature-fool).

- **Box-constrained attacks** ([notebook](notebooks/box_constrained_attack.ipynb))([code](adversarial_attacks)). 

NIPS 2017 adversarial attacks/defenses competition:
  * pgd_attack: Uses projected SGD (Stochastic Grandient Descent) as optimizer
  * step_pgd_attcK: Uses a mix of FGSM (Fast Gradient Sign Attack) and SGD. We found this to converge faster if there is a limit of only a few iterations (e.g. 10-15)

For a more comprehensive example, please check the provided [luizgh/adversarial_examples](https://github.com/luizgh/adversarial_examples)


## Physical-World Attacks

Robust Physical-World Attacks on Deep Learning Models

[Article](https://arxiv.org/pdf/1707.08945.pdf)

- [traffic-sign-classifier](traffic-sign-classifier/README.md)