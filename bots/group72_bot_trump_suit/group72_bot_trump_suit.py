#!/usr/bin/env python
"""
The simplest version of our bot, just a copy of the ml bot but with 
additional feature - number of trump cards in the hand of the current player in the second
phase it also switches to minimax algorithm. 

"""
from api import State, Deck, util
import random, os
from itertools import chain
from  bots import alphabeta
import joblib

# Path of the model we will use. If you make a model
# with a different name, point this line to its path.
DEFAULT_MODEL = os.path.dirname(os.path.realpath(__file__)) + '/model.pkl'

class Bot:

    __randomize = True

    __model = None

    def __init__(self, randomize=True, model_file=DEFAULT_MODEL, depth = 6):

       
        self.__randomize = randomize

        # Load the model
        self.__model = joblib.load(model_file)

        self.__max_depth = depth

    def get_move(self, state):
        

        val, move = self.value(state)
        return move
        

    def value(self, state, depth = 0):
        """
        Return the value of this state and the associated move
        :param state:
        :return: val, move: the value of the state, and the best move.
        """
        if state.finished():
            winner, points = state.winner()
            return (points, None) if winner == 1 else (-points, None)

        if depth == self.__max_depth:
            return self.heuristic_minimax(state)
        best_value = float('-inf') if maximizing(state) else float('inf')
        best_move = None

        moves = state.moves()

        if self.__randomize:
            random.shuffle(moves)
        hand_in_phase_1 = state.hand()
        for move in moves:

            next_state = state.next(move)

            # IMPLEMENT: Add a function call so that 'value' will
            # contain the predicted value of 'next_state'
            # NOTE: This is different from the line in the minimax/alphabeta bot

            if state.get_phase() == 1: # [GROUP 72 - commment] if we are not in the second phase our bot "switches" to using minimax algorithm. 

                value = self.heuristic(next_state, hand_in_phase_1)
            else:
                value, _ = self.value(next_state)  # [GROUP 72 - commment] it is possible to limit the depth of the minimax search by adding 
               # additional parameters to the recursive method call like for example self.value(next_state, depth+1), and tweaking the maximum depth 
               # parameter in the class. In the case of this design we decided not to limit the depth in order to achieve the best performance, in spite of a
            # bigger computational needs. 
        

            if maximizing(state):
                if value > best_value:
                    best_value = value
                    best_move = move
                 
            else:
                if value < best_value:
                    best_value = value
                    best_move = move


        return best_value, best_move

    def heuristic(self, state, h):

        # Convert the state to a feature vector
        feature_vector = [features(state, h)]

        # These are the classes: ('won', 'lost')
        classes = list(self.__model.classes_)

        # Ask the model for a prediction
        # This returns a probability for each class
        prob = self.__model.predict_proba(feature_vector)[0]

        # Weigh the win/loss outcomes (-1 and 1) by their probabilities
        res = -1.0 * prob[classes.index('lost')] + 1.0 * prob[classes.index('won')]

        return res
    def heuristic_minimax(self, state):
        # type: (State) -> float
        """
        Estimate the value of this state: -1.0 is a certain win for player 2, 1.0 is a certain win for player 1

        :param state:
        :return: A heuristic evaluation for the given state (between -1.0 and 1.0)
        """
        return util.ratio_points(state, 1) * 2.0 - 1.0, None
def maximizing(state):
    """
    Whether we're the maximizing player (1) or the minimizing player (2).
    :param state:
    :return:
    """
    return state.whose_turn() == 1


def features(state, h):

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
    trump_suit = state.get_trump_suit()

    # Add phase to feature set
    phase = state.get_phase()

    # Add stock size to feature set
    stock_size = state.get_stock_size()

    # Add leader to feature set
    leader = state.leader()

    # Add whose turn it is to feature set
    whose_turn = state.whose_turn()

    # Add opponent's played card to feature set
    opponents_played_card = state.get_opponents_played_card()

    # How many trump does a player hold? 
 
		#Get all trump suit moves available
    
    ################## You do not need to do anything below this line ########################

    perspective = state.get_perspective()

    # Perform one-hot encoding on the perspective.
    # Learn more about one-hot here: https://machinelearningmastery.com/how-to-one-hot-encode-sequence-data-in-python/
    perspective = [card if card != 'U'   else [1, 0, 0, 0, 0, 0] for card in perspective]
    perspective = [card if card != 'S'   else [0, 1, 0, 0, 0, 0] for card in perspective]
    perspective = [card if card != 'P1H' else [0, 0, 1, 0, 0, 0] for card in perspective]
    perspective = [card if card != 'P2H' else [0, 0, 0, 1, 0, 0] for card in perspective]
    perspective = [card if card != 'P1W' else [0, 0, 0, 0, 1, 0] for card in perspective]
    perspective = [card if card != 'P2W' else [0, 0, 0, 0, 0, 1] for card in perspective]

    # Append one-hot encoded perspective to feature_set

    feature_set += list(chain(*perspective))

   
    # Append normalized points to feature_set
    total_points = p1_points + p2_points
    feature_set.append(p1_points/total_points if total_points > 0 else 0.)
    feature_set.append(p2_points/total_points if total_points > 0 else 0.)

    # Append normalized pending points to feature_set
    total_pending_points = p1_pending_points + p2_pending_points
    feature_set.append(p1_pending_points/total_pending_points if total_pending_points > 0 else 0.)
    feature_set.append(p2_pending_points/total_pending_points if total_pending_points > 0 else 0.)

    # Convert trump suit to id and add to feature set
    # You don't need to add anything to this part
    suits = ["C", "D", "H", "S"]
    trump_suit_onehot = [0, 0, 0, 0]
    trump_suit_onehot[suits.index(trump_suit)] = 1
    feature_set += trump_suit_onehot

    # Append one-hot encoded phase to feature set
    feature_set += [1, 0] if phase == 1 else [0, 1]

    # Append normalized stock size to feature set
    feature_set.append(stock_size/10)

    # Append one-hot encoded leader to feature set
    feature_set += [1, 0] if leader == 1 else [0, 1]

    # Append one-hot encoded whose_turn to feature set
    feature_set += [1, 0] if whose_turn == 1 else [0, 1]

    # Append one-hot encoded opponent's card to feature set
    opponents_played_card_onehot = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    opponents_played_card_onehot[opponents_played_card if opponents_played_card is not None else 20] = 1
    feature_set += opponents_played_card_onehot

    # Return feature set

    def get_ratio_of_trump(t_s, h):
    
        n = 0 
        for card in h:
            if  Deck.get_suit(card) == t_s:
                n += 1
        return n/len(h)
   
    feature_set.append(get_ratio_of_trump(trump_suit, h))
    return feature_set

