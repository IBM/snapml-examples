# IBM Snap ML Examples

Example notebooks to demonstrate how to use the IBM Snap Machine Learning (Snap ML) library. 

## Getting started 

Install either [Anaconda](https://docs.anaconda.com/anaconda/install/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html).

Opne the Terminal on MacOS or Linux or the Anaconda/Miniconda prompt on Windows, 
and follow the steps below to create an Anaconda environment with everything you need to run the examples.

```bash
conda install git
git clone https://github.com/IBM/snapml-examples
cd snapml-examples
git checkout tpa-win
conda env create -f environment.yml
conda activate snapenv
jupyter notebook
```
With the Jupyter UI open in your web browser, navigate to the `examples` folder and explore the various example notebook provided.

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

