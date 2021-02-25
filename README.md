# IBM Snap ML Examples

Example notebooks to illustrate how to use the IBM Snap ML library. 

## Prerequisites 

Install dependencies using pip:

```bash
pip install -r requirements.txt
```

If you experience errors importing `snapml`, you may be be missing the OpenMP runtime library. 
In this case, please see the detailed [installation guide](https://snapml.readthedocs.io/en/latest/installation.html) for your platform.

## Datasets

This repository contains code to automatically download and pre-process datasets from a variety of different sources:
1. [Kaggle](https://www.kaggle.com/)
2. [LIBSVM Datasets Repository](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/)
3. [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php)

To use datasets hosted on Kaggle, you will need a Kaggle account and to [install an API token](https://www.kaggle.com/docs/api) on your machine.  

The following tools should be installed on your machine in order handle data extraction:
```bash
wget
bunzip2
gunzip
```

Datasets are downloaded and pre-processed once, and stored in the cache directory:
```bash
examples/cache_dir
```
If something goes wrong while extracting the data (e.g. a dependency missing), it may be helpful to clear the corresponding cache directory before trying again.