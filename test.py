#!/bin/python3
import os
import numpy as np
import csv
import requests

np.set_printoptions(suppress=True, linewidth=1000, threshold=1000)


def request(doc):
    payload = {"data": {"names": [], "tensor": {
        "shape": "", "values": doc}}}
    response = requests.post(
        # "http://ia.dom.unitech.com.ar:32135/predict",
        "http://localhost:5000/predict",
        json=payload)
    print(response.text[1:-2])
    return response.text[1:-2]


if __name__ == '__main__':
    test_count = 30
    predictions = []
    trainX = []
    # trainY = []
    with open("data/evaluacion.csv", mode='r') as infile:
        reader = csv.reader(infile, delimiter=";")
        # map word_list to number_list
        for row in reader:
            trainX.append(row[0])
            # trainY.append(row[1])

    for image in trainX[0:test_count]:
        prediction = request(image)
        predictions.append(prediction)

    print(predictions)
