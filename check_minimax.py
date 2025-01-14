"""
Check that the minmax bot and alpha beta bot return the same judgement, and that alphabeta bot is faster

"""

from api import State, util
import random, time

from bots.group72_bot_alphabeta import group72_bot_alphabeta
from bots.group72_bot_minimax import group72_bot_minimax

REPEATS = 20
DEPTH = 15

ab = group72_bot_alphabeta.Bot(randomize=False, depth=DEPTH)
mm = group72_bot_minimax.Bot(randomize=False, depth=DEPTH)

mm_time = 0
ab_time = 0

# Repeat
for r in range(REPEATS):
    
    # Repeat some more 
    for r2 in range(REPEATS):

        # Generate a starting state
        state = State.generate(phase=2)

        # Ask both bots their move
        # (and time their responses)

        start = time.time()
        mm_move = mm.get_move(state)
        mm_time += (time.time() - start)

        start = time.time()
        ab_move = ab.get_move(state)
        ab_time += (time.time() - start)


        if mm_move != ab_move:
            print('Difference of opinion! Minimax said: {}, alphabeta said: {}. State: {}'.format(mm_move, ab_move, state))
        else:
            print('Agreed.')

print('Done. time Minimax: {}, time Alphabeta: {}.'.format(mm_time/REPEATS, ab_time/REPEATS))
print('Alphabeta speedup: {} '.format(mm_time/ab_time))

