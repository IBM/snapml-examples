# IBM Snap ML Examples

Example notebooks to demonstrate how to use the IBM Snap Machine Learning (Snap ML) library. 

## Getting started 

### Linux/MacOS/Windows on x86 systems

Install either [Anaconda](https://docs.anaconda.com/anaconda/install/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html).

Open the Terminal (on MacOS or Linux) or the Anaconda/Miniconda prompt (on Windows), and follow the steps below to create a conda environment with everything you need to run the examples.

```bash
conda install git
git clone https://github.com/IBM/snapml-examples
cd snapml-examples
conda env create -f environment.yml
conda activate snapenv
jupyter notebook
```
With the Jupyter UI open in your web browser, navigate to the `examples` folder and explore the various example notebook provided.

### Linux on Z (s390x) systems

Install [Anaconda](https://docs.anaconda.com/anaconda/install/linux-s390x/) for Linux on Z.
 
On s390x some of the dependencies required to run these examples need to be compiled from source. 
This happens automatically when creating the anaconda environment, but it is necessary to install some development tools beforehand.

On Ubuntu systems:
```bash
apt-get install build-essential cmake libssl-dev
```
On RHEL systems:
```bash
yum groupinstall 'Development Tools'
yum install cmake openssl-devel
```
After installing these tools, follow the steps below:
```bash
export GRPC_PYTHON_BUILD_SYSTEM_OPENSSL=True
conda install git
git clone https://github.com/IBM/snapml-examples
cd snapml-examples
conda env create -f environment_z.yml
conda activate snapenvz
jupyter notebook
```

## Troubleshooting

If you experience errors importing `snapml`, you may be missing the OpenMP runtime library. 
In this case, please see the detailed [installation guide](https://snapml.readthedocs.io/en/latest/installation.html) for your platform.

## Datasets

This repository contains code to automatically download and pre-process datasets from a variety of different sources:
1. [Kaggle](https://www.kaggle.com/)
2. [LIBSVM Datasets Repository](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/)
3. [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php)

To use datasets hosted on Kaggle, you will need a Kaggle account and to [install an API token](https://www.kaggle.com/docs/api) on your machine.  

Datasets are downloaded, extracted and preprocessed once, and stored in the cache directory, which by default is set to:
```bash
examples/cache_dir
```
If something goes wrong while extracting the data (e.g. a dependency missing), it may be helpful to clear the corresponding cache directory before trying again.

The `GraphFeaturePreprocessor` example uses a synthethic dataset available here:
```bash
examples/datasets/graph_feature_preprocessor
```

## Resources

Find out more about Snap ML at the following links:

- [Snap ML installation guide](https://snapml.readthedocs.io/en/latest/installation.html)
- [Snap ML documentation](https://snapml.readthedocs.io/en/latest/)
- [Project homepage](https://www.zurich.ibm.com/snapml/)
- [Snap ML on PyPI](https://pypi.org/project/snapml/)

## Contact
 
For questions about the examples or Snap ML in general please contact:
- [Thomas Parnell](mailto:tpa@zurich.ibm.com)
- [Andreea Anghel](mailto:aan@zurich.ibm.com)
- [Martin Petermann](mailto:map@zurich.ibm.com)

