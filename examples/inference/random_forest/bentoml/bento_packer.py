# bento_packer.py
from snapml import RandomForestClassifier 

# import the BreastCancerClassifier class defined above
from bento_service import BreastCancerClassifier

# Create a breastcancer classifier service instance
breastcancer_classifier_service = BreastCancerClassifier()

# Load the models for later packing
snapml_model = RandomForestClassifier()

# To run Snap ML using the zDNN library (to run on the Z AI accelerator)
snapml_model.import_model("model.pmml", "pmml", "zdnn_tensors")

# To run Snap ML on CPU
# snapml_model.import_model("model.pmml", "pmml")

# Set the number of predict threads used at inference time
snapml_model.set_params(n_jobs=4)

# Pack the newly trained model artifact
breastcancer_classifier_service.pack('model', snapml_model)

# Save the prediction service to disk for model serving
saved_path = breastcancer_classifier_service.save()
