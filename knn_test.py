import tensorflow

import axelrod as axl

from knn_strategy import KNN

# from big_results import selected_strategies

# strategies = list(reversed(axl.ordinary_strategies))
#
# repetitions = 20
#
# def play_matches(player1, player2, repetitions):
#     match = axl.Match((player1, player2), turns=200)
#     for repetition in range(repetitions):
#         match.play()

# player1 = axl.KNN()
# player2 = axl.Random()
#
# match = axl.Match((player1, player2), turns=200)
# match.play()
# # print(match.sparklines())
# print(match.final_score())

# players = selected_strategies()

# axl.all_strategies.append(KNN)

def one_match():
    player1 = KNN()
    player2 = axl.Random()
    match = axl.Match((player1, player2), 100)
    match.play()
    print(match.winner())
    print(match.final_score())
    print(match.sparklines())

def main():

    players = [s for s in axl.all_strategies if axl.obey_axelrod(s())
               and not s().classifier['long_run_time']]

    players.append(KNN)
    players = [s() for s in players]

    tournament = axl.Tournament(players=players, repetitions=20)
    results = tournament.play()

    plot = axl.Plot(results)
    plot.save_all_plots(prefix="knn/tournament")

if __name__ == "__main__":
    with tensorflow.Session() as sess:
        # one_match()
        main()
    del sess
