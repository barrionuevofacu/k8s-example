#!/bin/python3
from polyaxon_client.tracking import Experiment, get_outputs_path
from argparse import ArgumentParser
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.utils import shuffle
from sklearn.tree import DecisionTreeClassifier
import os
import joblib
import utils
import numpy as np
from sklearn import datasets, metrics
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import SGDClassifier
# Polyaxon

np.set_printoptions(suppress=True, linewidth=1000, threshold=1000)


def main(args):
    print("args: " + str(args))
    print("preparing data ...")
    trainX, trainY = utils.getDataFromCsv(args.data)
    print("loaded {} records".format(len(trainX)))

    # create the transform
    xtrain, _ = utils.getVectorizer(trainX, useHashing=args.use_hashing,
                                    features=args.features)

    n_samples = xtrain.shape[0]
    train_len = (n_samples//100)*args.sample_size
    print("training samples: " + str(train_len))
    if (args.algorithm == "logistic"):
        clf = LogisticRegression(random_state=0, solver="lbfgs",
                                 multi_class='multinomial',
                                 max_iter=2000, verbose=0)
    elif (args.algorithm == "pagressive"):
        clf = PassiveAggressiveClassifier(max_iter=50, tol=1e-3)
    elif (args.algorithm == "sgd"):
        clf = SGDClassifier(alpha=.0001, max_iter=50, penalty="L2")
    clf.fit(xtrain[:train_len], trainY[:train_len])
    print("predicting ...")
    predicted = clf.predict(xtrain[train_len:])
    expected = trainY[train_len:]

    print("Classification report for classifier %s:\n%s\n"
          % (clf, metrics.classification_report(expected, predicted)))
    print("Confusion matrix:\n%s" %
          metrics.confusion_matrix(expected, predicted))
    # Train and eval the model with given parameters.
    # Polyaxon

    try:
        output_path = os.path.join(get_outputs_path(), "model.joblib")
    except:
        output_path = "model.joblib"
    print("dumping model parameters into '{}'".format(output_path))
    joblib.dump(clf, output_path)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-a", "--algorithm", dest="algorithm",
                        help="classifier algorithm to train",
                        default="logistic")
    parser.add_argument("-s", "--sample_size", dest="sample_size",
                        help="percentage of example data used to training",
                        type=int, default=80)
    parser.add_argument("-f", "--features", dest="features",
                        help="feature dictionary size (default=400)",
                        type=int, default=65535)
    parser.add_argument("-u", "--use_hashing", dest="use_hashing",
                        help="Vectorizer type: true -> *hashing* | " +
                        "false -> TFID",
                        type=utils.str2bool, default=True)
    parser.add_argument("-d", "--data", dest="data",
                        help="file with train/test data",
                        default="data/entrenamiento.tar.gz")

    args = parser.parse_args()
    main(args)
