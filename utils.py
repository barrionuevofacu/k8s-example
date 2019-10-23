import io
import csv
import gzip
import tarfile
from sklearn import metrics
from sklearn.utils import shuffle
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer

SPANISH_STOP_WORDS = ['un', 'una', 'unas', 'unos', 'uno', 'sobre', 'todo',
                      'también', 'tras', 'otro', 'algún', 'alguno', 'alguna',
                      'algunos', 'algunas', 'ser', 'es', 'soy', 'eres',
                      'somos', 'sois', 'estoy', 'esta', 'estamos', 'estais',
                      'estan', 'como', 'en', 'para', 'atras', 'porque', 'por',
                      'qué', 'estado', 'estaba', 'ante', 'antes', 'siendo',
                      'ambos', 'pero', 'por', 'poder', 'puede', 'puedo',
                      'podemos', 'podeis', 'pueden', 'fui', 'fue', 'fuimos',
                      'fueron', 'hacer', 'hago', 'hace', 'hacemos', 'haceis',
                      'hacen', 'cada', 'fin', 'incluso', 'primero', 'desde',
                      'conseguir', 'consigo', 'consigue', 'consigues',
                      'conseguimos', 'consiguen', 'ir', 'voy', 'va', 'vamos',
                      'vais', 'van', 'vaya', 'gueno', 'tener', 'tengo',
                      'tiene', 'tenemos', 'teneis', 'tienen', 'el', 'la',
                      'lo', 'las', 'los', 'su', 'aqui', 'mio', 'tuyo',
                      'ellos', 'ellas', 'nos', 'nosotros', 'vosotros',
                      'vosotras', 'si', 'dentro', 'solo', 'solamente', 'saber',
                      'sabes', 'sabe', 'sabemos', 'sabeis', 'saben',
                      'ultimo', 'largo', 'bastante', 'haces', 'muchos',
                      'aquellos', 'aquellas', 'sus', 'entonces', 'tiempo',
                      'verdad', 'verdadero', 'verdadera', 'cierto', 'ciertos',
                      'cierta', 'ciertas', 'intentar', 'intento', 'intenta',
                      'intentas', 'intentamos', 'intentais', 'intentan', 'dos',
                      'bajo', 'arriba', 'encima', 'usar', 'uso', 'usas', 'usa',
                      'usamos', 'usais', 'usan', 'emplear', 'empleo',
                      'empleas', 'emplean', 'ampleamos', 'empleais', 'valor',
                      'muy', 'era', 'eras', 'eramos', 'eran', 'modo', 'bien',
                      'cual', 'cuando', 'donde', 'mientras', 'quien', 'con',
                      'entre', 'sin', 'trabajo', 'trabajar', 'trabajas',
                      'trabaja', 'trabajamos', 'trabajais', 'trabajan',
                      'podria', 'podrias', 'podriamos', 'podrian', 'podriais',
                      'yo', 'aquel']

CLASSES = [
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


def getDataFromCsv(filename, delimiter=';', useShuffle=True, columns="0,1"):
    trainX = []
    trainY = []

    if (filename.endswith(".tar.gz")):
        with tarfile.open(filename) as tar:
            for member in tar:
                if member.isreg():      # Is it a regular file?
                    print("{} - {} bytes".format(member.name, member.size))
                    csv_file = io.StringIO(tar.extractfile(member).read()
                                           .decode('utf-8'))
                    reader = csv.reader(csv_file, delimiter=delimiter)
                    # this is here because of csv 'I/O op on closed file issue'
                    for row in reader:
                        trainX.append(row[int(columns.split(",")[0])])
                        trainY.append(row[int(columns.split(",")[1])])
    elif (filename.endswith(".csv") or filename.endswith(".txt")):
        with open(filename, mode='r') as infile:
            reader = csv.reader(infile, delimiter=delimiter)
            # hack/fix for I/O op on closed file
            # when using 'with file', file is closed when you leave code block
            for row in reader:
                trainX.append(row[int(columns.split(",")[0])])
                trainY.append(row[int(columns.split(",")[1])])

    else:
        print("file type not readable for training in this model")
        return [], []
    # map word_list to number_list
    if (useShuffle):
        trainX, trainY = shuffle(trainX, trainY)

    return trainX, trainY


def getVectorizer(data, useHashing=True, useStopWords=True, features=2**16):
    if useHashing:
        vectorizer = HashingVectorizer(stop_words=SPANISH_STOP_WORDS,
                                       alternate_sign=False,
                                       n_features=features)
        x_train = vectorizer.transform(data)
    else:
        vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                                     stop_words=SPANISH_STOP_WORDS)
        x_train = vectorizer.fit_transform(data)
    return x_train, vectorizer


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        pass
        # raise argparse.ArgumentTypeError('Boolean value expected.')


def printMeasures(clf, predicted, expected):
    print("Classification report for classifier %s:\n%s\n"
          % (clf, metrics.classification_report(expected, predicted)))
    print("Confusion matrix:\n%s" %
          metrics.confusion_matrix(expected, predicted))
