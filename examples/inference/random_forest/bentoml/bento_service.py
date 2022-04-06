# bento_service.py
import pandas as pd

from bentoml import env, artifacts, api, BentoService
from bentoml.adapters import DataframeInput
from bentoml.frameworks.sklearn import SklearnModelArtifact

@env(infer_pip_packages=True)
@artifacts([SklearnModelArtifact('model')])
class BreastCancerClassifier(BentoService):
    """
    A minimum prediction service exposing a Snap ML model
    """

    @api(input=DataframeInput(), batch=True)
    def predict(self, df):
        """
        An inference API named `predict` with Dataframe input adapter, which codifies
        how HTTP requests or CSV files are converted to a pandas Dataframe object as the
        inference API function input
        """
        return self.artifacts.model.predict(df.values)