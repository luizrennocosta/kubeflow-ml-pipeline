import dill
import logging


class Transformer(object):
    def __init__(self):

        with open("/mnt/model/lr_text.model", "rb") as model_file:
            self._lr_model = dill.load(model_file)

    def predict(self, X, feature_names):
        logging.info(X)
        prediction = self._lr_model.predict_proba(X)
        logging.info(prediction)
        return prediction
