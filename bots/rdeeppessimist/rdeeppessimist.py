"""
RdeepBot - This bot looks ahead by following a random path down the game tree. That is,
 it assumes that all players have the same strategy as rand.py, and samples N random
 games following from a given move. It then ranks the moves by averaging the heuristics
 of the resulting states.
"""

# Import the API objects
from api import State, Deck, util
import random
from typing import List, Tuple

class Bot:

	# How many samples to take per move
	__num_samples = -1
	# How deep to sample
	__depth = -1

	def __init__(self, num_samples=1, depth=8):
		self.__num_samples = num_samples
		self.__depth = depth

	def get_move(self, state):

		# See if we're player 1 or 2
		player = state.whose_turn()

		# Get a list of all legal moves
		moves = state.moves()

		# Sometimes many moves have the same, highest score, and we'd like the bot to pick a random one.
		# Shuffling the list of moves ensures that.
		random.shuffle(moves)

		best_score = float("-inf")
		best_move = None

		scores = [0.0] * len(moves)

		for move in moves:
			for s in range(self.__num_samples):

				# If we are in an imperfect information state, make an assumption.

				sample_state = Bot.make_pessimistic_assumption(state) if state.get_phase() == 1 else state

				score = self.evaluate(sample_state.next(move), player)

				if score > best_score:
					best_score = score
					best_move = move

		return best_move # Return the best scoring move

	@staticmethod
	def make_pessimistic_assumption(state: State):
		"""
		This code was moved here from deck and state. Trying to iompletment make_assumption outside of these classes.
		
		Takes the current imperfect information state and makes a 
		random guess as to the states of the unknown cards.
		:return: A perfect information state object.
		"""
		#This implementation assumed the game is in the first phase
		assert state.get_phase() == 1
			
		currentplayer = state.whose_turn()
		current_player_term = "P1H" if currentplayer == 1 else "P2H"
		other_player_term = "P2H" if currentplayer == 1 else "P1H"
		trump_suit = state.get_trump_suit()
	
		#we need to make the updated deck, for this we need the representation of the state of the cards, the beliefs of the players about the cards, the content of the stock and the trump suit
		
		oldCurrentPlayerPerspective = state.get_perspective()
		
		## we find the current unknowns.
		unknowns = [index for index, card in enumerate(oldCurrentPlayerPerspective) if card == "U"]
		#unknowns = the total number of cards we do not know. Both in the hand of the other player and on the stock
		
		#the other player has 5 cards in had, so this is the number of cards that player also does not know.
		# Note that we might know some of these cards because of mariages.
		other_player_unknowns = 5 - oldCurrentPlayerPerspective.count(other_player_term)
		numberOfUnknownForStock = len(unknowns) - other_player_unknowns

		#now, we split the cards for the hand of the other player and the stock
		toOtherPlayerHand, toStock = Bot.splitcards(unknowns, other_player_unknowns, numberOfUnknownForStock)

		# we nnow create a new perspective, thereby also filling in the unknowns
		newGlobalPerspective = list(oldCurrentPlayerPerspective)
		for i in toOtherPlayerHand:
			newGlobalPerspective[i] = other_player_term

		trump_index = oldCurrentPlayerPerspective.index("S")
		#We place the rest of the unknowns on  the stock. Keeping the trump on the bottom.
		stock = [trump_index] + toStock
		#Note, we did not touch the 'S'  representing the trump card in the newPerspective at all, so it is still set
		assert newGlobalPerspective[trump_index] == 'S' #Better safe than sorry

		for i in toStock:
			newGlobalPerspective[i] = "S"
		#the whole state should be known by now		
		assert not 'U' in newGlobalPerspective

		# Now we get to a tricky part. we need to create a vector to indicate the knowledge of the other. This means, we have to create the perspective for the other player.
		# This we will do based on our own perspective on the cards, the one of the current_player. 
		# In the standard implementation of make_assumption the perspective of the other player is directly copied. However, see https://github.com/intelligent-systems-course/schnapsen/issues/15
		# We do make one mistake in the implementation here. In what we do, the oponent players will loose information regarding past marriages and trump exchanges.
		# in order to preservce this, we would need to either remeber these things happened, or have access to the opponent's perspective.
		current_player_perspective = oldCurrentPlayerPerspective # the current player hasn't obtained more knowledge compared to what was known before
		other_player_perspective = list(newGlobalPerspective) #we make a copy of the new state of the cards and will then blind out things
		# We do create the perspective by making the following replacements in the newly created state:
		# If a card is marked as being in the hand of the current player, the other player cannot know, so we make it unknown. Note, here the mistake is made that the opponent would not know any of our cards.
		other_player_perspective = ['U' if cardStatus == current_player_term else cardStatus for cardStatus in other_player_perspective]
		# If a card is marked as being on the stock, the other player cannot now, except for the trump card, which we set back
		other_player_perspective = ['U' if cardStatus == 'S' else cardStatus for cardStatus in other_player_perspective]
		other_player_perspective[trump_index] = 'S'
		
		#The deck does not care about current and other, only about 1 and 2
		if currentplayer == 1:
			player1_perspective = current_player_perspective
			player2_perspective = other_player_perspective
		else: ## currentplayer == 2
			player2_perspective = current_player_perspective
			player1_perspective = other_player_perspective


		print (oldCurrentPlayerPerspective)
		print (newGlobalPerspective)
		print (player1_perspective)
		print (player2_perspective)
		deck = Deck(newGlobalPerspective, stock, player1_perspective , player2_perspective, trump_suit)


		player1s_turn = currentplayer == 1
		p1_points = state.get_points(1)
		p2_points = state.get_points(2)
		p1_pending_points = state.get_pending_points(1)
		p2_pending_points = state.get_pending_points(2)


		#we need to make a new state
		newState = State(deck, player1s_turn, p1_points, p2_points, p1_pending_points, p2_pending_points)
		# print (state)
		# print (newState)
		return newState

	@staticmethod
	def splitcards (cards: List[int], neededForOtherPlayer: int, neededForStock: int) -> Tuple[List[int], List[int]]:
		assert len(cards) == neededForOtherPlayer + neededForStock
		####### here you would change the order in unknowns #####
		#For now just shuffling them with a fixed seed for testing
		rng = random.Random(467486564)
		rng.shuffle(cards)
		forOtherPlayer = cards[:neededForOtherPlayer]
		forStock = cards[neededForOtherPlayer:]
		### end changing of the order ######
		assert len(forOtherPlayer) == neededForOtherPlayer
		assert len(forStock) == neededForStock
		return forOtherPlayer, forStock

	def evaluate(self,
				 state,     # type: State
				 player     # type: int
			):
		# type: () -> float
		"""
		Evaluates the value of the given state for the given player
		:param state: The state to evaluate
		:param player: The player for whom to evaluate this state (1 or 2)
		:return: A float representing the value of this state for the given player. The higher the value, the better the
			state is for the player.
		"""

		score = 0.0

		for _ in range(self.__num_samples):

			st = state.clone()

			# Do some random moves
			for i in range(self.__depth):
				if st.finished():
					break

				st = st.next(random.choice(st.moves()))

			score += self.heuristic(st, player)

		return score/float(self.__num_samples)

	def heuristic(self, state, player):
		return util.ratio_points(state, player)