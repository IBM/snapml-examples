# *****************************************************************
#
# Licensed Materials - Property of IBM
#
# (C) Copyright IBM Corp. 2023. All Rights Reserved.
#
# US Government Users Restricted Rights - Use, duplication or
# disclosure restricted by GSA ADP Schedule Contract with IBM Corp.
#
# *****************************************************************

import numpy as np
import onnx
import onnxmltools
from onnxmltools.convert.common.data_types import FloatTensorType
from snapml import RandomForestClassifier as snapRFC
from sklearn.ensemble import RandomForestClassifier as skRFC
from xgboost import XGBClassifier
from snapml import BoostingMachineClassifier
from sklearn.metrics import accuracy_score
from sklearn2pmml import sklearn2pmml
from sklearn2pmml.pipeline import PMMLPipeline

# Load the train dataset
X_train = np.loadtxt("../data/train_data.csv", dtype=np.float64, delimiter=",", comments="#")
print("Input train data shape: ", X_train.shape)

# Load the test dataset
X_test = np.loadtxt("../data/test_data.csv", dtype=np.float64, delimiter=",", comments="#")
print("Input test data shape:  ", X_test.shape)

# Extract the labels
y_train = X_train[:,0]
y_test  = X_test[:,0]

# Extract the features
X_train = X_train[:, 1:]
X_test  = X_test[:, 1:]

print("# ----------------------------------------")
print("# Train a Scikit-learn Random Forest Model")
print("# ----------------------------------------")
model = skRFC(random_state=42)
pipeline = PMMLPipeline([("model", model)]).fit(X_train, y_train)
sklearn2pmml(pipeline, "../models/model.pmml", with_repr=True)
preds = pipeline.predict(X_test)
print("[Sklearn Random Forest] Accuracy score: ", accuracy_score(y_test, preds))

print("# ----------------------------------------")
print("# Train an XGBoost Model")
print("# ----------------------------------------")
model = XGBClassifier(random_state=36)
model.fit(X_train, y_train)

# Export the model to JSON
model.get_booster().save_model("../models/model.json")

# Export the same xgboost model to ONNX
initial_type = [("float_input", FloatTensorType([None, X_train.shape[1]]))]
onnx_model = onnxmltools.convert_xgboost(model, initial_types=initial_type)
with open("../models/model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())
preds = model.predict(X_test)
print("[XGBoost] Accuracy score: ", accuracy_score(y_test, preds))

print("# --------------------------------------------------")
print("# Load Scikit-learn Random Forest Model into Snap ML")
print("# --------------------------------------------------")

model = snapRFC()
model.import_model("../models/model.pmml", "pmml")
preds = model.predict(X_test)
preds[preds < 0] = 0
preds[preds > 0] = 1
print("[Snap ML Sklearn Random Forest (pmml)] Accuracy score: ", accuracy_score(y_test, preds))

print("# --------------------------------------------------")
print("# Load XGBoost Model into Snap ML")
print("# --------------------------------------------------")
model = BoostingMachineClassifier()
model.import_model("../models/model.json", "xgb_json")
preds = model.predict(X_test)
preds[preds < 0] = 0
preds[preds > 0] = 1
print("[Snap ML XGBoost (json)] Accuracy score: ", accuracy_score(y_test, preds))

model = BoostingMachineClassifier()
model.import_model("../models/model.onnx", "onnx")
preds = model.predict(X_test)
preds[preds < 0] = 0
preds[preds > 0] = 1
print("[Snap ML XGBoost (onnx)] Accuracy score: ", accuracy_score(y_test, preds))

