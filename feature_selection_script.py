from api import State, util
import pickle
import os.path
from argparse import ArgumentParser
import time
import sys
from itertools import chain

# This package contains various machine learning algorithms
import sklearn
import sklearn.linear_model
from sklearn.neural_network import MLPClassifier
import joblib
from bots.rand import rand
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest, chi2

def features(state):
    # type: (State) -> tuple[float, ...]
    """
    Extract features from this state. Remember that every feature vector returned should have the same length.

    :param state: A state to be converted to a feature vector
    :return: A tuple of floats: a feature vector representing this state.
    """

    feature_set = []

    # Add player 1's points to feature set
    p1_points = state.get_points(1)

    # Add player 2's points to feature set
    p2_points = state.get_points(2)

    # Add player 1's pending points to feature set
    p1_pending_points = state.get_pending_points(1)

    # Add plauer 2's pending points to feature set
    p2_pending_points = state.get_pending_points(2)



    ################## You do not need to do anything below this line ########################

   

    # Append normalized points to feature_set
    total_points = p1_points + p2_points
    feature_set.append(p1_points/total_points if total_points > 0 else 0.)
    feature_set.append(p2_points/total_points if total_points > 0 else 0.)

    # Append normalized pending points to feature_set
    total_pending_points = p1_pending_points + p2_pending_points
    feature_set.append(p1_pending_points/total_pending_points if total_pending_points > 0 else 0.)
    feature_set.append(p2_pending_points/total_pending_points if total_pending_points > 0 else 0.)



    # Return feature set
    return feature_set


state = State.generate(phase=1)
print(state)
state_vectors = []

while not state.finished():

            # Give the state a signature if in phase 1, obscuring information that a player shouldn't see.
            given_state = state.clone(signature=state.whose_turn()) if state.get_phase() == 1 else state

            # Add the features representation of a state to the state_vectors array
            state_vectors.append(features(given_state))

            # Advance to the next state
            move = rand.Bot().get_move(given_state)
            state = state.next(move)
           





