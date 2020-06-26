#!usr/bin/env python
"""
A command line program for multiple games between several bots.

For all the options run
python play.py -h
"""

from argparse import ArgumentParser, ArgumentTypeError
from api import State, util, engine
import random, time

def run_tournament(options):

    botnames = options.players.split(",")

    bots = []
    for botname in botnames:
        bots.append(util.load_player(botname))

    n = len(bots)
    wins = [0] * len(bots)
    matches = [(p1, p2) for p1 in range(n) for p2 in range(n) if p1 < p2]

    totalgames = (n*n - n)/2 * options.repeats
    playedgames = 0

    #setup rng
    random.seed(984976)

    gameRNG = random.Random(options.gameseed)



    print('Playing {} games:'.format(int(totalgames)))
    for a, b in matches:
        for r in range(options.repeats):

            if random.choice([True, False]):
                p = [a, b]
            else:
                p = [b, a]
            # Generate a state with a random seed
            gameID = gameRNG.random()
            state = State.generate(id = gameID, phase=int(options.phase))

            winner, score = engine.play(bots[p[0]], bots[p[1]], state, options.max_time*1000, verbose=options.verbose, fast=options.fast)

            if winner is not None:
                winner = p[winner - 1]
                wins[winner] += score

            playedgames += 1
            print('Played {} out of {:.0f} games ({:.0f}%): {} \r'.format(playedgames, totalgames, playedgames/float(totalgames) * 100, wins))

    print('Results:')
    for i in range(len(bots)):
        print('    bot {}: {} points'.format(bots[i], wins[i]))


if __name__ == "__main__":

    ## Parse the command line options
    parser = ArgumentParser()
    
    def oneOrTwo(arg_value):
        phase = int(arg_value)
        if not (phase == 1 or phase == 2):
            raise ArgumentTypeError("The pahse has to be 1 or 2")
        return phase

    parser.add_argument("-s", "--starting-phase",
                        dest="phase",
                        help="Which phase the game should start at.",
                        type=oneOrTwo,
                        default=1)

    parser.add_argument("-p", "--players",
                        dest="players",
                        help="Comma-separated list of player names (enclose with quotes).",
                        default="rand,bully,rdeep")

    parser.add_argument("-r", "--repeats",
                        dest="repeats",
                        help="How many matches to play for each pair of bots",
                        type=int, default=10)

    parser.add_argument("-t", "--max-time",
                        dest="max_time",
                        help="maximum amount of time allowed per turn in seconds (default: 5)",
                        type=int, default=5)

    parser.add_argument("-f", "--fast",
                        dest="fast",
                        action="store_true",
                        help="This option forgoes the engine's check of whether a bot is able to make a decision in the allotted time, so only use this option if you are sure that your bot is stable.")

    parser.add_argument("-v", "--verbose",
                        dest="verbose",
                        action="store_true",
                        help="Print verbose information")

    parser.add_argument("-g", "--gameseed",
                        dest="gameseed",
                        type=int,
                        help="The seed for the rng that creates the games",
                        default=4654687
                        )

    options = parser.parse_args()

    run_tournament(options)
