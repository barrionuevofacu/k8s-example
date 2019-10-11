#!/bin/python3
import joblib
import numpy as np
import csv
import utils
from sklearn.utils import shuffle
from sklearn.feature_extraction.text import HashingVectorizer

np.set_printoptions(suppress=True, linewidth=1000, threshold=1000)


class Model(object):
    def __init__(self):
        self.clf = joblib.load('model.joblib')

    def predict_raw(self, X):
        doc = X["data"]["tensor"]["values"]
        return self.predict(doc, None)

    def predict(self, X, feature_names):
        # create the transform
        trainx, _ = utils.getVectorizer([X], features=10000)
        # summarize encoded vector
        predictions = self.clf.predict(trainx)
        return predictions[0]


if __name__ == '__main__':
    print("preparing data ...")
    trainX, trainY = utils.getDataFromCsv("data/entrenamiento.tar.gz")
    print("predicting ...")
    serve = Model()
    i = 0
    ok = 0
    total = 0
    errors = []
    for document in trainX:  # [0:500]:
        predicted1 = serve.predict(document, None)
        payload = {"data": {"names": [], "tensor": {
            "shape": "", "values": document}}}
        predicted2 = serve.predict_raw(payload)

        total += 1
        if (predicted1 != predicted2):
            print("!!! Predictions doesn't match on same input")
        else:
            if (trainY[i] == predicted1):
                ok += 1
            else:
                errors.append(trainY[i] + " != " + predicted1)
        i += 1

    print("Summary")
    print("Matches: {} of {} ({})".format(ok, total, ok/total))
    # print("Errors:\n" + str(errors))
