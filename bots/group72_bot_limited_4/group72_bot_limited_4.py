#!/usr/bin/env python
"""
A basic machine learing bot but limited to a set of only 4 features. 
"""

from numpy import percentile
from api import State, util
import random, os
from itertools import chain
from api import Deck
import joblib

# Path of the model we will use. If you make a model
# with a different name, point this line to its path.
DEFAULT_MODEL = os.path.dirname(os.path.realpath(__file__)) + '/model.pkl'

class Bot:

    __randomize = True

    __model = None

    def __init__(self, randomize=True, model_file=DEFAULT_MODEL):

        print(model_file)
        self.__randomize = randomize

        # Load the model
        self.__model = joblib.load(model_file)

    def get_move(self, state):

        val, move = self.value(state)

        return move

    def value(self, state):
        """
        Return the value of this state and the associated move
        :param state:
        :return: val, move: the value of the state, and the best move.
        """

        best_value = float('-inf') if maximizing(state) else float('inf')
        best_move = None

        moves = state.moves()
     
        if self.__randomize:
            random.shuffle(moves)

        for move in moves:

            next_state = state.next(move)

            # IMPLEMENT: Add a function call so that 'value' will
            # contain the predicted value of 'next_state'
            # NOTE: This is different from the line in the minimax/alphabeta bot
            
            value = self.heuristic(next_state)
         
            if maximizing(state):
                if value > best_value:
                    best_value = value
                    best_move = move

                    
            else:
                if value < best_value:
                    best_value = value
                    best_move = move


        return best_value, best_move

    def heuristic(self, state):

        # Convert the state to a feature vector
        feature_vector = [features(state)]

        # These are the classes: ('won', 'lost')
        classes = list(self.__model.classes_)
       
        # Ask the model for a prediction
        # This returns a probability for each class
        prob = self.__model.predict_proba(feature_vector)[0]

        # Weigh the win/loss outcomes (-1 and 1) by their probabilities
        res = -1.0 * prob[classes.index('lost')] + 1.0 * prob[classes.index('won')]
 
        return res

def maximizing(state):
    """
    Whether we're the maximizing player (1) or the minimizing player (2).
    :param state:
    :return:
    """
    return state.whose_turn() == 1


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

    # Get trump suit
   

    # How many trump does a player hold? 
 
		#Get all trump suit moves available
    
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
