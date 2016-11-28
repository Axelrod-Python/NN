import csv

from keras.optimizers import Adagrad, RMSprop

import axelrod as axl
from knn_strategy import KNN

from generate_data import yield_data
from train import train_model, test_model, load_model, save_model

def process_data(filename, outfilename):
    with open(outfilename, 'w') as outputfile:
        writer = csv.writer(outputfile)
        for line in yield_data(filename):
            writer.writerow(line)

def interactions_generator(player, opponents=None, repetitions=100, noise=0,
                           turns=200):
    if not opponents:
        opponents = [s() for s in axl.all_strategies if axl.obey_axelrod(s())
               and not s().classifier['long_run_time']]

    for opponent in opponents:
        match = axl.Match((player, opponent), turns, noise=noise)
        for rep in range(repetitions):
            match.play()
            yield (player.history, opponent.history)


def write_interactions(interactions_gen, filename):
    with open(filename, 'w') as handle:
        writer = csv.writer(handle)
        for h1, h2 in interactions_gen:
            row = [0, 0, '', '', ''.join(h1), ''.join(h2)]
            writer.writerow(row)

def create_interactions(s=""):
    g = interactions_generator(KNN(), repetitions=20)
    filename = "/ssd/raw_train_extra.csv{}".format(s)
    write_interactions(g, filename)
    outfilename = "/ssd/train_extra.csv{}".format(s)
    process_data(filename, outfilename)

    g = interactions_generator(KNN(), repetitions=4)
    filename = "/ssd/raw_test_extra.csv{}".format(s)
    write_interactions(g, filename)
    outfilename = "/ssd/test_extra.csv{}".format(s)
    process_data(filename, outfilename)


if __name__ == "__main__":
    # try:
    #     s = sys.argv[1]
    # except IndexError:
    #     s = ""
    # create_interactions(s)


    epochs = 1

    for i in range(0, 10):
        # Generate new data
        create_interactions(str(i))
        # Load
        model = load_model(name="model")
        model.compile(loss='mse',
           optimizer=RMSprop())
        # Train
        train_filename = "/ssd/train_extra.csv{}".format(i)
        model, losses = train_model(model, epochs, train_filename, nb_epoch=2)
        # Test
        print("MSE", losses[-1])
        test_filename = "/ssd/test_extra.csv{}".format(i)
        m = test_model(model, test_filename)
        # Save model
        save_model(model, name="model")
        # if m > 0.93:
        #     break



