# Snap ML C++ API Examples

In these examples we demonstrate how to use the Snap ML C++ API 1) to train tree models (decision tree, random forest, boosting machine) in Snap ML, 2) to run inference on Snap ML-trained tree models, and 3) to import tree-ensemble models trained using external frameworks (such as XGBoost) into Snap ML, and run inference on them. We provide three examples: 1) decision tree (`src/capi-dt-example.cpp`), 2) random forest (`src/capi-rf-example.cpp`), and 3) boosting machine (`src/capi-boost-example.cpp`).

## 0.Generate the ML models
To generate the ML models run the following commands:
```
cd gen-models
python gen-models.py
cd ..
```
The models will be stored in the `models` folder.

## 1.Compile the examples
To compile the examples run the following commands:
```
mkdir build
cd build
cmake -DSNAPML_LIB_DIR=$SNAPML_LIB_DIR -DSNAPML_INCLUDE_DIR=$SNAPML_INCLUDE_DIR ..
make -j
```
where `SNAPML\_LIB\_DIR`=`python -c 'import snapml; import os ; print(os.path.dirname(snapml.__file__))'` and `SNAPML\_INCLUDE\_DIR`=`${SNAPML_LIB_DIR}/include`.

## 2. Run the decision tree example
To run the decision tree example:
```
./test-dt
```
Expected output:
```
------------------------------------------------------
Read input data and convert to Snap ML data format
------------------------------------------------------
  Loading train dataset
  Data OK.
  Loading test dataset
  Data OK.

------------------------------------------------------
Scenario 1 (training + inference): Decision Tree Model
------------------------------------------------------
  Training model
  Running inference
  Accuracy score 1
```

## 3. Run the random forest example
To run the random forest example:
```
./test-rf
```
Expected output:
```
------------------------------------------------------
Read input data and convert to Snap ML data format
------------------------------------------------------
  Loading the train dataset
  Data OK.
  Loading the test dataset
  Data OK.

------------------------------------------------------
Scenario 1 (training + inference): Random Forest Model
------------------------------------------------------
  Training model
  Running inference
  Accuracy score 1

------------------------------------------------------
Scenario 2 (import + inference): Random Forest Model
------------------------------------------------------
  Importing model from file
  Number of classes found in the imported model 2
  Running inference
  Accuracy score 1
```
## 4. Run the boosting machine example
To run the boosting machine example:
```
./test-boost
```
Expected output:
```

------------------------------------------------------
Read input data and convert to Snap ML data format
------------------------------------------------------
  Loading the train dataset
  Data OK.
  Loading the test dataset
  Data OK.

------------------------------------------------------
Scenario 1 (training + inference): Boosting Model
------------------------------------------------------
  Training model
  Running inference
  Accuracy score 1

------------------------------------------------------
Scenario 2 (import + inference): Boosting Model
------------------------------------------------------
  Importing model from file
  Number of classes found in the imported model 2
  Running inference
  Accuracy score 1

```

