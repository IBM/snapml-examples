## Train a scikit-learn using the sklearn breast cancer dataset.

We first train a Random Forest Classifier model using the breast cancer sklearn dataset. We save the model as a PMML pipeline.

```
python3 train.py
```

## Define the BentoML service interface

Before model serving, the first step is to create a prediction service class. This class defines the user interface API (the serving logic). The class is defined in the `bento_service.py` file. The sklearn Random Forest model saved in the PMML file will be imported in the Snap ML library which has a high-performance prediction engine that will replace sklearn's prediction engine. Given Snap ML's compatibility with scikit-learn, it is not necessary to define a new BentoML artifact / framework for Snap ML. One can use the BentoML sklearn interface to serve Snap ML models (`from bentoml.frameworks.sklearn import SklearnModelArtifact`).
 
## Save the prediction service for distribution

The command below prepares the BentoML service for deployment. It imports the sklearn model from the PMML file into Snap ML and sets the number of CPU threads to be used by each predict call (each BentoML worker) at inference time. 

```
python3 bento_packer.py
```

## Start the BentoML server

Start the BentoML server running the BreastCancerClassifier model with 1 BentoML worker.

```
bentoml serve-gunicorn --do-not-track -q -w 1 BreastCancerClassifier:latest
```

## Test the BentoML server

Test the BentoML service by sending an HTTP request for classification.

```
python3 test-service.py
```
