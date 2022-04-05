# train.py
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
from sklearn2pmml import sklearn2pmml
from sklearn2pmml import PMMLPipeline

# Load training data
X, y = datasets.load_breast_cancer(return_X_y=True)

# Train a PMML pipeline and save it in PMML format
# To be used by SnapML which imports the Sklearn model into its optimized predict engine

model = RandomForestClassifier(n_estimators = 100, max_depth=4)
pipeline = PMMLPipeline([("model", model)]).fit(X,y)
sklearn2pmml(pipeline, "model.pmml", with_repr=True)

