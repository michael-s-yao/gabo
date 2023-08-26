# In-Distribution Objective Optimization for Generative Networks

[![LICENSE](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE.md)
[![CONTACT](https://img.shields.io/badge/contact-michael.yao%40pennmedicine.upenn.edu-blue)](mailto:michael.yao@pennmedicine.upenn.edu)
[![CONTACT](https://img.shields.io/badge/contact-obastani%40seas.upenn.edu-blue)](mailto:obastani@seas.upenn.edu)

Na&iuml;vely training a generative model to optimize an objective function can result in a generated distribution out-of-domain compared to the training distribution. This is a common problem encountered in the protein design space: in trying to maximize protein-target interactions, generated proteins are often non-physiologic and cannot be easily synthesized *in-vivo*. In this work, we aim to solve this problem through proposing a technique to regularize the optimization problem using a source-discriminator similar to that used in [Generative Adversarial Nets (GANs)](https://arxiv.org/abs/1406.2661). We show that our method is able to optimize an objective function subject to staying reasonably in-distribution.

## Installation

To install and run our code, first clone the `OODOptimization` repository.

```
git clone https://github.com/michael-s-yao/OODOptimization
cd OODOptimization
```

Next, create a virtual environment and install the relevant dependencies.

```
python -m venv env
source env/bin/activate
pip install -r requirements.txt
```

Our codebase uses [Weights & Biases](https://wandb.ai/site) for experiment tracking. To properly configure W&B, please run the following command in the root project directory.

```
python setup.py --project [PROJECT_NAME]
```

After successful setup, you can run our code as

```
python main.py --optimizer [OPTIMIZER]
```

If you would prefer not to use W&B for experiment tracking, you can disable it using

```
python main.py --optimizer [OPTIMIZER] --disable_wandb
```

For other script arguments, you can run

```
python main.py --help
```

## Contact

Questions and comments are welcome. Suggestions can be submitted through Github issues. Contact information is linked below.

[Michael Yao](mailto:michael.yao@pennmedicine.upenn.edu)

[Osbert Bastani](mailto:obastani@seas.upenn.edu)

## Citation

When available, relevant citation information will be added in a future commit.

## License

This repository is MIT licensed (see [LICENSE](LICENSE)).
