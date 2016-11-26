import csv
from collections import defaultdict, Counter
import functools
import multiprocessing
import os

import numpy as np

import axelrod as axl

# Features
# more history?
# longest defection streak?

# mapping = {'C': 1, 'D': -1}
# mapping = {'C': 0, 'D': 1}
# mapping = {'C': 1, 'D': 0}

from features import extract_features

def write_csv(outcomes, filename="outcomes.csv", append=False):
    s = 'w'
    if append:
        s = 'a'
    writer = csv.writer(open(filename, s))
    for row in outcomes:
       writer.writerow(row)

def write_winner(filename, turns, repetitions, noise, i, j, seed=None):
    """
    Write the winner of a Match to file
    """
    if seed:
        axl.seed(seed)  # Seed the process

    pairs = (players[i]().clone(), players[j]().clone())
    match = axl.Match(pairs, turns=turns, noise=noise)
    rs = repetitions
    if not match._stochastic and noise == 0:
        rs = max(1, int(repetitions / 4))
    outcomes = []
    for _ in range(rs):
        match.play()
        outcomes.append([
            ''.join(pairs[0].history),
            ''.join(pairs[1].history)
        ])

    write_csv(outcomes, filename=filename, append=True)

def generate_matchups_indices(num_players):
    # Want the triangular product
    for i in range(num_players):
        for j in range(i, num_players):
            yield i, j

def sample_match_outcomes_parallel(turns, repetitions, filename, noise=0,
                                   processes=None):
    """
    Parallel matches.
    """

    player_indices = range(len(players))
    if processes is None:
        for i in player_indices:
            print(i, len(players))
            for j in player_indices:
                for seed in range(repetitions):
                    write_winner(filename, turns, repetitions, noise, i, j,
                                 seed)
    else:
        func = functools.partial(write_winner, filename, turns, repetitions,
                                 noise)
        p = multiprocessing.Pool(processes)

        args = generate_matchups_indices(len(players))
        p.starmap(func, args)

# def zeros_and_ones(h):
#     return list(map(lambda x: mapping[x], h))
#
# def cumulative_context_counts(h1, h2):
#     counts = []
#     # counts = []
#     d = defaultdict(int)
#     for i, (p1, p2) in enumerate(zip(h1, h2)):
#         d[str(p1) + str(p2)] += 1
#         # if i >= 4:
#         counts.append((d['CC'], d['CD'], d['DC'], d['DD']))
#     return counts
#
# # custom cumsum for zeros and ones Cs and Ds
#
# def cumulative_cooperations(h):
#     coops = []
#     s = 0
#     for play in h:
#         if play == 'C':
#             s += 1
#         coops.append(s)
#     return coops
#
# def cumulative_scores(h1, h2):
#     ss1, ss2 = [], []
#     game = axl.Game()
#     for p1, p2 in zip(h1, h2):
#         s1, s2 = game.score((p1, p2))
#         ss1.append(s1)
#         ss2.append(s2)
#     return np.cumsum(ss1), np.cumsum(ss2)

# def vectorize_interactions(h1, h2):
#     # ds = np.cumsum(h1)
#     # op_ds = np.cumsum(h2)
#     coops = cumulative_cooperations(h1)
#     op_coops = cumulative_cooperations(h2)
#     ccs = cumulative_context_counts(h1, h2)
#     # scores1, scores2 = cumulative_scores(h1, h2)
#     h1 = zeros_and_ones(h1)
#     h2 = zeros_and_ones(h2)
#     # Handle N=0 and 1 separately
#     yield [0] * 17 + [h2[0],
#                       # 0, 0
#                       ]
#     row = [
#         1,
#         coops[0], 1 - coops[0],
#         op_coops[0], 1 - op_coops[0],
#         h1[0], 0, h2[0], 0,
#         0, h1[0], 0, h2[0],
#         # scores1[0], scores2[0]
#     ]
#     row.extend(ccs[0])
#     y = h2[1]
#     row.append(y)
#     yield row
#     for i in range(2, len(h1)):
#         row = [
#             i,
#             coops[i-1], i - coops[i-1],
#             op_coops[i-1], i - op_coops[i-1],
#             h1[0], h1[1], h2[0], h2[1],
#             h1[i-2], h1[i-1], h2[i-2], h2[i-1],
#             # scores1[i-1], scores2[i-1]
#         ]
#         row.extend(ccs[i-1])
#         y = h2[i]
#         row.append(y)
#         yield row


def yield_data(filename):
    with open(filename) as handle:
        for line in handle:
            s = line.strip().split(',')
            h1, h2 = s[-2], s[-1]
            yield from extract_features(h1, h2, include_target=True)

def process_data():
    with open("/ssd/train1.csv", 'w') as outputfile:
        writer = csv.writer(outputfile)
        for line in yield_data("/ssd/interactions-train.csv1"):
            writer.writerow(line)
    with open("/ssd/test1.csv", 'w') as outputfile:
        writer = csv.writer(outputfile)
        for line in yield_data("/ssd/interactions-test.csv1"):
            writer.writerow(line)

def generate_data(turns=200, noise=0., repetitions=40, processes=4):
    output_filename = "/ssd/interactions-train.csv1"
    try:
        os.remove(output_filename)
    except:
        pass

    sample_match_outcomes_parallel(turns, repetitions, output_filename,
                                   noise=noise,
                                   processes=processes)

    output_filename = "/ssd/interactions-test.csv1"
    try:
        os.remove(output_filename)
    except:
        pass
    sample_match_outcomes_parallel(turns, 10, output_filename,
                                   noise=noise,
                                   processes=processes)

if __name__ == "__main__":
    players = [s for s in axl.all_strategies if axl.obey_axelrod(s())
               and not s().classifier['long_run_time']]

    generate_data(repetitions=100)
    process_data()
