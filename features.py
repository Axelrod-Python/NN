"""Extract features from the history of play of a game theory match between
two players.

Potential additionally valuable features:
* Context responses
* Longest Streaks
* Has Cycles
* More history

"""

from collections import defaultdict

import numpy as np

import axelrod as axl

# mapping = {'C': 1, 'D': -1}
mapping = {'C': 1, 'D': 0}

def zeros_and_ones(h, round_num=-1):
    """Translate C/D history to binary."""
    return list(map(lambda x: mapping[x], h[:round_num]))

# def cumulative_context_counts(h1, h2):
#     """Extracts context counts for the finite state process underlying a game:
#     CC, CD, DC, DD."""
#
#     counts = []
#     d = defaultdict(int)
#     for i, (p1, p2) in enumerate(zip(h1, h2)):
#         d[str(p1) + str(p2)] += 1
#         counts.append((d['CC'], d['CD'], d['DC'], d['DD']))
#     return counts

def cumulative_cooperations(h):
    """Computes the cumulative cooperations over the full history."""
    coops = []
    s = 0
    for play in h:
        if play == 'C':
            s += 1
        coops.append(s)
    return coops

def cumulative_scores(h1, h2):
    """Computes the cumulative scores of each player."""
    ss1, ss2 = [], []
    game = axl.Game()
    for p1, p2 in zip(h1, h2):
        s1, s2 = game.score((p1, p2))
        ss1.append(s1)
        ss2.append(s2)
    return np.cumsum(ss1), np.cumsum(ss2)

def starting_moves(h, round_num, depth):
    base = [0] * (2 * depth)
    for i in range(0, min(round_num, depth)):
        if h[i] == 'C':
            base[2 * i] = 1
            base[2 * i + 1] = 0
        else:
            base[2 * i] = 0
            base[2 * i + 1] = 1
    return base

def trailing_moves(h, round_num, depth):
    base = [0] * (2 * depth)
    for i in range(0, min(round_num, depth)):
        if h[round_num - i - 1] == 'C':
            base[2 * i] = 1
            base[2 * i + 1] = 0
        else:
            base[2 * i] = 0
            base[2 * i + 1] = 1
    return base

def num_features(starting=2, trailing=2, include_scores=False):
    length = 5 + 2 * starting + 4 * trailing
    if include_scores:
        length += 2
    return length


def extract_features_single(h1, h2, c1, c2, round_num, starting=2, trailing=2,
                            include_scores=False,
                            include_target=False):
    if include_scores:
        scores1, scores2 = cumulative_scores(h1, h2)

    # h1 = zeros_and_ones(h1)
    # h2 = zeros_and_ones(h2)

    # Default is round_num, player1 coops and defects, player2 coops and defects
    length = num_features(starting=starting, trailing=trailing,
                          include_scores=include_scores)

    # Handle N=0
    if round_num == 0:
        row = [0] * length
        if include_target:
            row += [mapping[h2[0]]]
        return row

    i = round_num - 1
    row = [
        round_num,
        c1, round_num - c1,
        c2, round_num - c2
    ]
    row += starting_moves(h2, round_num, starting)
    row += trailing_moves(h1, round_num, trailing)
    row += trailing_moves(h2, round_num, trailing)
    if include_scores:
        row += [scores1[i], scores2[i]]
    if include_target:
        row += [mapping[h2[round_num]]]
    return row


def extract_features(h1, h2, starting=2, trailing=2, include_scores=False,
                     include_target=False):
    coops = cumulative_cooperations(h1)
    op_coops = cumulative_cooperations(h2)
    # ccs = cumulative_context_counts(h1, h2)
    
    # if include_scores:
    #     scores1, scores2 = cumulative_scores(h1, h2)

    # h1 = zeros_and_ones(h1)
    # h2 = zeros_and_ones(h2)

    # Default is round_num, player1 coops and defects, player2 coops and defects
    # length = num_features(starting=starting, trailing=trailing,
    #                       include_scores=include_scores)

    # # Handle N=0
    # row = [0] * length
    # if include_target:
    #     row += [h2[0]]
    # yield row

    # for round_num in range(1, len(h1)):
    #     i = round_num - 1
    #     row = [
    #         round_num,
    #         coops[i], round_num - coops[i],
    #         op_coops[i], round_num - op_coops[i]
    #     ]
    #     row += starting_moves(h2, round_num, 2)
    #     row += trailing_moves(h1, round_num, 2)
    #     row += trailing_moves(h2, round_num, 2)
    #     if include_scores:
    #         row += [scores1[i], scores2[i]]
    #     if include_target:
    #         row += [h2[1]]
    #     yield row

    for round_num in range(0, len(h1)):
        row = extract_features_single(
            h1, h2,
            coops[round_num-1], op_coops[round_num-1],
            round_num,
            starting=starting, trailing=trailing,
            include_scores=include_scores,
            include_target=include_target)
        yield row