# Offline Generative Adversarial Bayesian Optimization (GABO)

[![LICENSE](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE.md)
[![CONTACT](https://img.shields.io/badge/contact-michael.yao%40pennmedicine.upenn.edu-blue)](mailto:michael.yao@pennmedicine.upenn.edu)
[![CONTACT](https://img.shields.io/badge/contact-obastani%40seas.upenn.edu-blue)](mailto:obastani@seas.upenn.edu)

Offline model-based policy optimization seeks to optimize a learned surrogate objective function without querying the true oracle objective during optimization. However, inaccurate surrogate model predictions are frequently encountered in this setting. To address this limitation, we propose *adaptive source critic regularization* that utilizes a Lipschitz-constrained source critic agent to constrain the optimization trajectory to regions where the surrogate performs well. We show that under certain assumptions for the continuous input space prior, we can dynamically adjust the strength of the source critic regularization, which consistently outperforms existing baselines on a number of different optimization tasks across a variety of domains. Our work provides a practical framework for offline policy optimization via source critic regularization.

## Installation

To install and run our code, first clone the `gabo` repository.

```
cd ~
git clone https://github.com/michael-s-yao/gabo
cd gabo
```

Next, create a [`conda` environment](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file) from the `environment.yml` file to setup the environment and install the relevant dependencies.

```
conda env create -f environment.yml
conda activate gabo
```

If you are running our codebase on a GPU, please also run the following commands:

```
python -m pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
```

There is also a minor versioning conflict between the installed dependences that requires one line in a package to be modified. More specifically, please navigate to the location of the installed `transformers` package:

```
cd /home/usr/miniconda3/envs/gabo/lib/python3.8/site-packages/transformers
vim trainer_pt_utils.py
```

In the `trainer_pt_utils.py` source code, file, please change the line `if version.parse(torch.__version__) <= version.parse("1.4.1"):` to `if version.parse(torch.__version__) <= version.parse("1.12.1"):`. Next, please copy the [`smiles_vocab.txt`](./data/molecules/smiles_vocab.txt) file to the `design_bench_data` package directory:

```
cd ~/gabo
cp -p data/molecules/smiles_vocab.txt /home/usr/miniconda3/envs/gabo/lib/python3.8/site-packages/design_bench_data/
```

Similarly, please also follow the directions [here](https://github.com/rail-berkeley/design-bench/issues/1) to also download the `design-bench`-associated datasets as well if applicable. Finally, please initialize the submodules associated with the repository. After successful setup, you can run our code as

```
python mbo/run_gabo.py --help
```

To replicate our experiments, please refer to the [`scripts`](./scripts) directory for relevant shell scripts.

### Running BDI Baseline Experiments

[Bidirectional learning for offline infinite-width model-based optimization (BDI)](https://openreview.net/forum?id=_j8yVIyp27Q) is a baseline method for MBO tasks that we compare against in our experiments. The codebase provided by Chen et al. (2022) depends on the `jax` and [`neural-tangents`](https://github.com/google/neural-tangents/tree/main) libraries which are version-incompatible with our own dependencies specified in [environment.yml](environment.yml). Therefore, if you are interested in replicating our BDI experimental results, please create a new `conda` environment separate from the one described above using

```
conda env create -f bdi_environment.yml
conda activate bdi
```

Of note, you may come across versioning issues with `jaxlib`, a required dependency for `jax`. The solution proposed in this [GitHub issue](https://github.com/google/jax/issues/18368#issuecomment-1792381092) worked in our hands.

## Contact

Questions and comments are welcome. Suggestions can be submitted through Github issues. Contact information is linked below.

[Michael Yao](mailto:michael.yao@pennmedicine.upenn.edu)

[Osbert Bastani](mailto:obastani@seas.upenn.edu)

## Citation

When available, relevant citation information will be added in a future commit.

## License

This repository is MIT licensed (see [LICENSE](LICENSE)).
