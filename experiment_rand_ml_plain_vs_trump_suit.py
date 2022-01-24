"""

This script shows an example of how to run a simple computational experiment. The research
question is as follows:

    What is the value of saving your trump cards for the late stages of the game?

As a first step towards answering this question, we will make two assumptions:

    1) Players have two options: play randomly, or play highest value non-trump card
    2) Players decide between these two entirely at random

Under these simplified circumstances, how often should players decide to save their trump cards?
This is a simple question to answer, we simply build rand bots for a range of parameters, and play a few games for each
combination. We plot the results in a heat map

"""
from os import stat
from shutil import move
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt
from api import State, util
from statsmodels.stats.proportion import binom_test

import random
from bots.group72_bot_plain_ml import group72_bot_plain_ml
from bots.group72_bot_trump_suit import group72_bot_trump_suit

# Define the bot:
# (we're not using it with the command line tools, so we can just put it here)
class Bot:

    # Probability of moving with non-trump cards
  

    def get_move(self, state):

        moves = state.moves()



            # IMPLEMENT: Make the best non-trump move you can. Use the best_non_trump_card method written below.
  
        #IMPLEMENT: Make a random move (but exclude the best non-trump move from above)

      
 
        random_choice = random.choice(moves)

        return random_choice


def empty(n):
    """
    :param n: Size of the matrix to return
    :return: n by n matrix (2D array) filled with 0s
    """
    return [[0 for i in range(n)] for j in range(n)]



# For experiments, it's good to have repeatability, so we set the seed of the random number generator to a known value.
# That way, if something interesting happens, we can always rerun the exact same experiment
seed = random.randint(1, 1000)
print('Using seed {}.'.format(seed))
random.seed(seed)

# Parameters of our experiment
STEPS = 10


# Make empty matrices to count how many times each player won for a given
# combination of parameters
won_by_1 = 0
won_by_2 = 0


# We will move through the parameters from 0 to 1 in STEPS steps, and play REPEATS games for each
# combination. If at combination (i, j) player 1 wins a game, we increment won_by_1[i][j]

for i in range(STEPS):
    for j in range(100):

            # Make the players
            player1 = group72_bot_plain_ml.Bot()
            player2 = group72_bot_trump_suit.Bot()

            state = State.generate()

            # play the game
            while not state.finished():
                player = player1 if state.whose_turn() == 1 else player2
                state = state.next(player.get_move(state))

            #TODO Maybe add points for state.winner()
            if state.finished():
                winner, points = state.winner()
                if winner == 1:
                    won_by_1 += 1
                else:
                    won_by_2 += 1










# Plot the data as a heatmap
p_value = binom_test(won_by_2, 1000, prop=0.5, alternative='larger') # Alternative could get three values : "two-sided", "larger", "smaller"


# Plot the data as a heatmap

names = [ str(won_by_1) + ' games group72_bot_plain_ml', str(won_by_2) +  ' games group72_bot_trump_suit_ml' + '\n p_value=' + str(p_value)]

values = [won_by_1, won_by_2]

plt.figure(figsize=(10, 10))

plt.bar(names, values)


# Always label your axes

plt.savefig('experiment'+ '_seed' + str(seed)+ '_group72_bot_plain_ml_vs_group72_bot_trump_suit_1000_games.pdf')
