#!/bin/python3
import logging
import numpy as np
from optparse import OptionParser
import sys
from time import time
import matplotlib.pyplot as plt
import csv

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.utils.extmath import density
from sklearn import metrics

my_stop_words = ['un', 'una', 'unas', 'unos', 'uno', 'sobre', 'todo',
                 'también', 'tras', 'otro', 'algún', 'alguno', 'alguna',
                 'algunos', 'algunas', 'ser', 'es', 'soy', 'eres', 'somos',
                 'sois', 'estoy', 'esta', 'estamos', 'estais', 'estan',
                 'como', 'en', 'para', 'atras', 'porque', 'por', 'qué',
                 'estado', 'estaba', 'ante', 'antes', 'siendo', 'ambos',
                 'pero', 'por', 'poder', 'puede', 'puedo', 'podemos',
                 'podeis', 'pueden', 'fui', 'fue', 'fuimos', 'fueron',
                 'hacer', 'hago', 'hace', 'hacemos', 'haceis', 'hacen',
                 'cada', 'fin', 'incluso', 'primero', 'desde', 'conseguir',
                 'consigo', 'consigue', 'consigues', 'conseguimos',
                 'consiguen', 'ir', 'voy', 'va', 'vamos', 'vais', 'van',
                 'vaya', 'gueno', 'tener', 'tengo', 'tiene', 'tenemos',
                 'teneis', 'tienen', 'el', 'la', 'lo', 'las', 'los', 'su',
                 'aqui', 'mio', 'tuyo', 'ellos', 'ellas', 'nos', 'nosotros',
                 'vosotros', 'vosotras', 'si', 'dentro', 'solo', 'solamente',
                 'saber', 'sabes', 'sabe', 'sabemos', 'sabeis', 'saben',
                 'ultimo', 'largo', 'bastante', 'haces', 'muchos', 'aquellos',
                 'aquellas', 'sus', 'entonces', 'tiempo', 'verdad',
                 'verdadero', 'verdadera', 'cierto', 'ciertos', 'cierta',
                 'ciertas', 'intentar', 'intento', 'intenta', 'intentas',
                 'intentamos', 'intentais', 'intentan', 'dos', 'bajo',
                 'arriba', 'encima', 'usar', 'uso', 'usas', 'usa', 'usamos',
                 'usais', 'usan', 'emplear', 'empleo', 'empleas', 'emplean',
                 'ampleamos', 'empleais', 'valor', 'muy', 'era', 'eras',
                 'eramos', 'eran', 'modo', 'bien', 'cual', 'cuando', 'donde',
                 'mientras', 'quien', 'con', 'entre', 'sin', 'trabajo',
                 'trabajar', 'trabajas', 'trabaja', 'trabajamos', 'trabajais',
                 'trabajan', 'podria', 'podrias', 'podriamos', 'podrian',
                 'podriais', 'yo', 'aquel']

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')


# parse commandline arguments
op = OptionParser()
op.add_option("--report",
              action="store_true", dest="print_report",
              help="Print a detailed classification report.")
op.add_option("--chi2_select",
              action="store", type="int", dest="select_chi2",
              help="Select some number of features using a chi-squared test")
op.add_option("--confusion_matrix",
              action="store_true", dest="print_cm",
              help="Print the confusion matrix.")
op.add_option("--top10",
              action="store_true", dest="print_top10",
              help="Print ten most discriminative terms per class"
                   " for every classifier.")
op.add_option("--use_hashing",
              action="store_true",
              help="Use a hashing vectorizer.")
op.add_option("--n_features",
              action="store", type=int, default=2 ** 16,
              help="n_features when using the hashing vectorizer.")
op.add_option("--filtered",
              action="store_true",
              help="Remove newsgroup information that is easily overfit: "
                   "headers, signatures, and quoting.")
op.add_option("--sample",
              action="store",
              help="train data percentage from total input",
              type=int, default=80)


def is_interactive():
    return not hasattr(sys.modules['__main__'], '__file__')


# work-around for Jupyter notebook and IPython console
argv = [] if is_interactive() else sys.argv[1:]
(opts, args) = op.parse_args(argv)
if len(args) > 0:
    op.error("this script takes no arguments.")
    sys.exit(1)

print(__doc__)
op.print_help()
print()


# #############################################################################
# Load some categories from the training set
categories = None

if opts.filtered:
    remove = ('headers', 'footers', 'quotes')
else:
    remove = ()

trainX = []
trainY = []
with open("data/entrenamiento.csv", mode='r') as infile:
    reader = csv.reader(infile, delimiter=";")
    # map word_list to number_list
    for row in reader:
        trainX.append(row[0])
        trainY.append(row[1])

n_samples = len(trainX)
train_len = (n_samples//100)*opts.sample
test_len = n_samples-(n_samples-train_len)
data_train = trainX[:train_len]
data_test = trainX[test_len:]
print('data loaded: %s (training: %s, eval: %s )' % (n_samples,
                                                     len(data_train),
                                                     len(data_test)))

# order of labels in `target_names` can be different from `categories`
target_names = [
    "Acordadas",
    "Asociaciones Sindicales",
    "Avisos Oficiales",
    "Concursos Oficiales",
    "Convenciones Colectivas de Trabajo",
    "Decisiones Administrativas",
    "Decretos",
    "Disposiciones",
    "Leyes",
    "Remates Oficiales",
    "Resoluciones",
    "Resoluciones Conjuntas",
    "Resoluciones Generales",
    "Resoluciones Sintetizadas",
    "Sentencias",
    "Tratados y Convenios Internacionales"
]


def size_mb(docs):
    return sum(len(s.encode('utf-8')) for s in docs) / 1e6


data_train_size_mb = size_mb(data_train)
data_test_size_mb = size_mb(data_test)

print("%d documents - %0.3fMB (training set)" % (
    len(data_train), data_train_size_mb))
print("%d documents - %0.3fMB (test set)" % (
    len(data_test), data_test_size_mb))
print("%d categories" % len(target_names))
print()

# split a training set and a test set
y_train, y_test = trainY[:train_len], trainY[test_len:]

print("Extracting features from the training data using a sparse vectorizer")
t0 = time()
if opts.use_hashing:
    vectorizer = HashingVectorizer(stop_words=my_stop_words,
                                   alternate_sign=False,
                                   n_features=opts.n_features)
    X_train = vectorizer.transform(data_train)
else:
    vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                                 stop_words=my_stop_words)
    X_train = vectorizer.fit_transform(data_train)
duration = time() - t0
print("done in %fs at %0.3fMB/s" % (duration, data_train_size_mb / duration))
print("n_samples: %d, n_features: %d" % X_train.shape)
print()

print("Extracting features from the test data using the same vectorizer")
t0 = time()
X_test = vectorizer.transform(data_test)
duration = time() - t0
print("done in %fs at %0.3fMB/s" % (duration, data_test_size_mb / duration))
print("n_samples: %d, n_features: %d" % X_test.shape)
print()

# mapping from integer feature name to original token string
if opts.use_hashing:
    feature_names = None
else:
    feature_names = vectorizer.get_feature_names()

if opts.select_chi2:
    print("Extracting %d best features by a chi-squared test" %
          opts.select_chi2)
    t0 = time()
    ch2 = SelectKBest(chi2, k=opts.select_chi2)
    X_train = ch2.fit_transform(X_train, y_train)
    X_test = ch2.transform(X_test)
    if feature_names:
        # keep selected feature names
        feature_names = [feature_names[i] for i
                         in ch2.get_support(indices=True)]
    print("done in %fs" % (time() - t0))
    print()

if feature_names:
    feature_names = np.asarray(feature_names)


def trim(s):
    """Trim string to fit on terminal (assuming 80-column display)"""
    return s if len(s) <= 80 else s[:77] + "..."


# #############################################################################
# Benchmark classifiers
def benchmark(clf):
    print('_' * 80)
    print("Training: ")
    print(clf)
    t0 = time()
    clf.fit(X_train, y_train)
    train_time = time() - t0
    print("train time: %0.3fs" % train_time)

    t0 = time()
    pred = clf.predict(X_test)
    test_time = time() - t0
    print("test time:  %0.3fs" % test_time)

    score = metrics.accuracy_score(y_test, pred)
    print("accuracy:   %0.3f" % score)

    if hasattr(clf, 'coef_'):
        print("dimensionality: %d" % clf.coef_.shape[1])
        print("density: %f" % density(clf.coef_))

        if opts.print_top10 and feature_names is not None:
            print("top 10 keywords per class:")
            for i, label in enumerate(target_names):
                top10 = np.argsort(clf.coef_[i])[-10:]
                print(trim("%s: %s" % (label, " ".join(feature_names[top10]))))
        print()

    if opts.print_report:
        print("classification report:")
        print(metrics.classification_report(y_test, pred,
                                            target_names=target_names))

    if opts.print_cm:
        print("confusion matrix:")
        print(metrics.confusion_matrix(y_test, pred))

    print()
    clf_descr = str(clf).split('(')[0]
    return clf_descr, score, train_time, test_time


results = []
for clf, name in (
        (RidgeClassifier(tol=1e-2, solver="sag"), "Ridge Classifier"),
        (PassiveAggressiveClassifier(max_iter=50, tol=1e-3),
         "Passive-Aggressive")  # ,
):
    print('=' * 80)
    print(name)
    results.append(benchmark(clf))

for penalty in ["l2", "l1"]:
    print('=' * 80)
    print("%s penalty" % penalty.upper())
    # Train Liblinear model
    results.append(benchmark(LinearSVC(penalty=penalty, dual=False,
                                       tol=1e-3)))

    # Train SGD model
    results.append(benchmark(SGDClassifier(alpha=.0001, max_iter=50,
                                           penalty=penalty)))

# Train SGD with Elastic Net penalty
print('=' * 80)
print("Elastic-Net penalty")
results.append(benchmark(SGDClassifier(alpha=.0001, max_iter=50,
                                       penalty="elasticnet")))

# Train sparse Naive Bayes classifiers
print('=' * 80)
print("Naive Bayes")
results.append(benchmark(ComplementNB(alpha=.1)))

print('=' * 80)
print("LinearSVC with L1-based feature selection")
# The smaller C, the stronger the regularization.
# The more regularization, the more sparsity.
results.append(benchmark(Pipeline([
    ('feature_selection', SelectFromModel(LinearSVC(penalty="l1", dual=False,
                                                    tol=1e-3))),
    ('classification', LinearSVC(penalty="l2"))])))

print('=' * 80)
print("LogisticRegression with lbfgs multinomial")
results.append(benchmark(LogisticRegression(random_state=0, solver="lbfgs",
                                            multi_class='multinomial',
                                            max_iter=2000, verbose=0)))
# make some plots

indices = np.arange(len(results))

results = [[x[i] for x in results] for i in range(4)]

clf_names, score, training_time, test_time = results
training_time = np.array(training_time) / np.max(training_time)
test_time = np.array(test_time) / np.max(test_time)

plt.figure(figsize=(12, 8))
plt.title("Score")
plt.barh(indices, score, .2, label="score", color='navy')
plt.barh(indices + .3, training_time, .2, label="training time",
         color='c')
plt.barh(indices + .6, test_time, .2, label="test time", color='darkorange')
plt.yticks(())
plt.legend(loc='best')
plt.subplots_adjust(left=.25)
plt.subplots_adjust(top=.95)
plt.subplots_adjust(bottom=.05)

for i, c in zip(indices, clf_names):
    plt.text(-.3, i, c)

plt.show()
