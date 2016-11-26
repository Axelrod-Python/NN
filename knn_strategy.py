from collections import defaultdict
from axelrod import Actions, Player
import numpy as np

import tensorflow

config = tensorflow.ConfigProto(
    device_count={'GPU': 0}
)
sess = tensorflow.Session(config=config)

# import theano
# theano.config.device = "cpu"
# theano.config.force_device = True

from keras.models import model_from_json

C, D = Actions.C, Actions.D

from features import mapping, zeros_and_ones, extract_features_single
from train import load_model

"""
Todo:
* decision function based on output
* run tournaments to collect more data and re-train
network
* RNN
"""


class KNN(Player):
    """ """

    name = '============================'
    classifier = {
        'memory_depth': float('inf'),
        'stochastic': False,
        'makes_use_of': set(),
        'inspects_source': False,
        'manipulates_source': False,
        'manipulates_state': False
    }

    def __init__(self):
        Player.__init__(self)
        self.model = load_model("/home/user/repos/axelrod/nn/model")
        # self.play_counts = defaultdict(int)
        self.init_args = ()

    def strategy(self, opponent):
        # Record context counts
        # if len(self.history) > 0:
        #     last_round = (self.history[-1], opponent.history[-1])
        #     self.play_counts[last_round] += 1
        # else:
        #     return C
        # # TFT for first 2 rounds
        # if len(self.history) == 0:
        #     return C
        # if len(self.history) == 1:
        #     if opponent.history[-1] == D:
        #         return D
        #     return C

        if len(self.history) < 4:
            if len(self.history) == 0:
                return C
            # React to the opponent's last move
            if opponent.history[-1] == D:
                return D
            return C

        if opponent.defections == 0:
            return C

        # features = compute_features(self, opponent)
        features = extract_features_single(self.history, opponent.history,
                                           self.cooperations,
                                           opponent.cooperations,
                                           len(self.history))
        X = np.array([features])
        coop_prob = self.model.predict_proba(X, verbose=False)[0]

        if coop_prob < 0.50:
            return D

        # Can we get away with a defection?
        # Update features for a round of D, C
        h1 = self.history + ['D']
        h2 = opponent.history + ['C']
        features = extract_features_single(h1, h2,
                                           self.cooperations,
                                           opponent.cooperations + 1,
                                           len(h1))
        X = np.array([features])
        next_coop_prob = self.model.predict_proba(X, verbose=False)
        if next_coop_prob > 0.90:
            return D
        else:
            return C

        # if coop_prob > 0.5:
        #     return C
        # else:
        #     return D

        # def reset(self):
        #     Player.reset(self)
        #     self.play_counts = defaultdict(int)
