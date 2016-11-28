from __future__ import print_function
from collections import Counter, defaultdict
import csv

from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix

import tqdm

from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.optimizers import Adagrad, RMSprop, SGD


def save_model(model, name):
    # serialize model to JSON
    model_json = model.to_json()
    with open(name + ".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(name + ".h5", overwrite=True)

def load_model(name="model"):
    json_file = open(name + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(name + ".h5")
    return loaded_model

def read_batches(filename):
    with open(filename) as in_file:
        reader = csv.reader(in_file)
        for row in reader:
            x = list(map(int, row[:-1]))
#             y = int(row[-1])
            y = max(0, int(row[-1]))
            yield x, y

def BatchGenerator(filename, batch_size=128*1024):
    X = []
    Y = []
    for x, y in read_batches(filename=filename):
        X.append(x)
        Y.append(y)
        if len(Y) == batch_size:
            yield np.array(X), np.array(Y)
            X = []
            Y = []
    if len(Y):
        yield np.array(X), np.array(Y)

def construct_NN(activation="relu", exit_act="softmax", input_dim=17,
                 inner_layers=2, inner_dim=15):
    # Define a Feed Forward NN
    model = Sequential()
    model.add(Dense(input_dim=input_dim, output_dim=inner_dim))
    model.add(Activation(activation))
    for i in range(inner_layers):
        model.add(Dense(input_dim=inner_dim, output_dim=inner_dim))
        model.add(Activation(activation))
    # Output layer
    model.add(Dense(input_dim=inner_dim, output_dim=1))
    model.add(Activation(exit_act))
    # model.compile(loss='mse',
    #               optimizer=Adagrad(lr=0.02, epsilon=1e-08))
    model.compile(loss='mse',
                  optimizer=RMSprop())

    return model

def train_model(model, epochs, train_filename, nb_epoch=10,
                batch_size=1024):
    losses = []
    for _ in tqdm.tqdm(range(epochs)):
        for X_batch, Y_batch in BatchGenerator(train_filename):
            loss = model.fit(X_batch, Y_batch, batch_size=batch_size,
                             nb_epoch=nb_epoch, verbose=False,
                             validation_split=0.15, )
            losses.extend(loss.history['loss'])
            #             loss = model.train_on_batch(X_batch, Y_batch)
            #             losses.append(loss)
    return model, losses

def plot_counts(counter):
    xs = list(range(0, 101))
    ys = [0] * len(xs)
    for x, y in counter.items():
        ys[x] = y
    plt.bar(xs, ys)

def test_model(model, test_filename):
    acc_counter = Counter()
    pred_counter = Counter()
    prob_counter = Counter()

    for X, y in BatchGenerator(test_filename, batch_size=1024):
        # Model evaluation
        pred_y = model.predict_proba(X, verbose=False)
        preds = model.predict_classes(X, verbose=False)
        pred_counter.update(zip(y, [x[0] for x in preds]))
        prob_counter.update([int(100 * x) for x in pred_y])
        acc = accuracy_score(y, preds)
        acc_counter.update([int(100 * acc)])

    print('CONFUSION MATRIX:\n', pred_counter)

    plt.clf()
    plot_counts(prob_counter)
    plt.title("Predicted p_coop")
    plt.savefig("knn/preds.png")

    plt.clf()
    plot_counts(acc_counter)
    plt.title("Accuracies")
    plt.savefig("knn/test.png")


def train(epochs=1, train_filename  ="/ssd/train1.csv"):
    print("Training...")
    model = construct_NN(activation="relu", exit_act="sigmoid",
                         inner_layers=1, inner_dim=17)
    model, losses = train_model(model, epochs, train_filename, nb_epoch=1)

    print("MSE", losses[-1])
    plt.plot(range(len(losses)), losses)
    plt.savefig("knn/losses.png")
    save_model(model, "model")
    return model


def test(model=None, test_filename="/ssd/test1.csv"):
    print("Testing....")
    if not model:
        model = load_model("model")
    for X in [
        [92,90,2,61,31,1,0,0,1,1,0,1,0,1,0,0,1,0],
        [142,36,106,72,70,1,0,1,0,0,1,1,0,1,0,0,1,1],
        [163,80,83,79,84,0,1,0,1,1,0,1,0,1,0,1,0,0],
        [32,13,19,8,24,1,0,1,0,1,0,0,1,0,1,0,1,1]
        ]:
        print(X)
        print(model.predict_proba(np.array([X[:-1]]), verbose=False))
        print(model.predict_classes(np.array([X[:-1]]), verbose=False))

    test_model(model, test_filename)

if __name__ == "__main__":
    model = train(epochs=10)
    test()



