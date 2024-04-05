# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util,math
from game import Agent


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)

        "*** YOUR CODE HERE ***"
        ghost_distance = self.distance_to_nearest_ghost(successorGameState)
        food_distance = self.distance_to_nearest_food(successorGameState, currentGameState)
        return ghost_distance / (food_distance + 1)

    def distance_to_nearest_ghost(self, gameState):
        pacmanPos = gameState.getPacmanPosition()
        ghostsPos = gameState.getGhostPositions()
        distances = [math.dist(pacmanPos, ghostPos) for ghostPos in ghostsPos]
        return min(distances)

    def distance_to_nearest_food(self, nextState, currentState):
        pacmanPos = nextState.getPacmanPosition()
        foodsPos = currentState.getFood().asList()
        distances = [math.dist(pacmanPos, foodPos) for foodPos in foodsPos]
        return min(distances)


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()
        # return util.raiseNotDefined();
    

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.
        """
        action, score = self.minimax(0, 0, gameState)  # Get the action and score for Pacman (agent_index=0)
        return action  # Return the action to be taken as per the Minimax algorithm

    def minimax(self, curr_depth, agent_index, gameState):
        '''
        Returns the best score for an agent using the minimax algorithm. For the max player (agent_index=0), the best
        score is the maximum score among its successor states, and for the min player (agent_index!=0), the best
        score is the minimum score among its successor states. Recursion ends if there are no successor states
        available or curr_depth equals the max depth to be searched until.
        :param curr_depth: the current depth of the tree (int)
        :param agent_index: index of the current agent (int)
        :param gameState: the current state of the game (GameState)
        :return: action, score
        '''
        if agent_index >= gameState.getNumAgents():
            agent_index = 0
            curr_depth += 1
        
        if curr_depth == self.depth:
            return None, self.evaluationFunction(gameState)
        
        if agent_index == 0:  
            return self.max_value(curr_depth, agent_index, gameState)
        else:  
            return self.min_value(curr_depth, agent_index, gameState)

    def max_value(self, curr_depth, agent_index, gameState):
        best_score, best_action = None, None
        for action in gameState.getLegalActions(agent_index):  
            next_game_state = gameState.generateSuccessor(agent_index, action)
            _, score = self.minimax(curr_depth, agent_index + 1, next_game_state)
            if best_score is None or score > best_score:
                best_score = score
                best_action = action
        if best_score is None:
            return None, self.evaluationFunction(gameState)
        return best_action, best_score  

    def min_value(self, curr_depth, agent_index, gameState):
        best_score, best_action = None, None
        for action in gameState.getLegalActions(agent_index):  
            next_game_state = gameState.generateSuccessor(agent_index, action)
            _, score = self.minimax(curr_depth, agent_index + 1, next_game_state)
            if best_score is None or score < best_score:
                best_score = score
                best_action = action
        if best_score is None:
            return None, self.evaluationFunction(gameState)
        return best_action, best_score  

        # util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        action, _ = self.alpha_beta(0, 0, gameState, -math.inf, math.inf)  
        return action  

    def alpha_beta(self, curr_depth, agent_index, gameState, alpha, beta):
        '''
        Returns the best score for an agent using the alpha-beta algorithm. For max player (agent_index=0), the best
        score is the maximum score among its successor states and for the min player (agent_index!=0), the best
        score is the minimum score among its successor states. Recursion ends if there are no successor states
        available or curr_depth equals the max depth to be searched until. If alpha > beta, we can stop generating
        further successors and prune the search tree.
        :param curr_depth: the current depth of the tree (int)
        :param agent_index: index of the current agent (int)
        :param gameState: the current state of the game (GameState)
        :param alpha: the alpha value of the parent (float)
        :param beta: the beta value of the parent (float)
        :return: action, score
        '''
        
        if agent_index >= gameState.getNumAgents():
            agent_index = 0
            curr_depth += 1
        
        if curr_depth == self.depth:
            return None, self.evaluationFunction(gameState)
        
        if agent_index == 0:
            return self.max_value(curr_depth, agent_index, gameState, alpha, beta)
        else:
            return self.min_value(curr_depth, agent_index, gameState, alpha, beta)

    def max_value(self, curr_depth, agent_index, gameState, alpha, beta):
        best_score, best_action = None, None
        for action in gameState.getLegalActions(agent_index):
            next_game_state = gameState.generateSuccessor(agent_index, action)
            _, score = self.alpha_beta(curr_depth, agent_index + 1, next_game_state, alpha, beta)
            if best_score is None or score > best_score:
                best_score = score
                best_action = action
            alpha = max(alpha, score)
            if alpha > beta:
                break
        if best_score is None:
            return None, self.evaluationFunction(gameState)
        return best_action, best_score  

    def min_value(self, curr_depth, agent_index, gameState, alpha, beta):
        best_score, best_action = None, None
        for action in gameState.getLegalActions(agent_index):
            next_game_state = gameState.generateSuccessor(agent_index, action)
            _, score = self.alpha_beta(curr_depth, agent_index + 1, next_game_state, alpha, beta)
            if best_score is None or score < best_score:
                best_score = score
                best_action = action
            beta = min(beta, score)
            if beta < alpha:
                break
        if best_score is None:
            return None, self.evaluationFunction(gameState)
        return best_action, best_score  
 
        # util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        action, score = self.expectimax(0, 0, gameState)  
        return action  

    def expectimax(self, curr_depth, agent_index, gameState):
        '''
        Returns the best score for an agent using the expectimax algorithm. For max player (agent_index=0), the best
        score is the maximum score among its successor states and for the min player (agent_index!=0), the best
        score is the average of all its successor states. Recursion ends if there are no successor states
        available or curr_depth equals the max depth to be searched until.
        :param curr_depth: the current depth of the tree (int)
        :param agent_index: index of the current agent (int)
        :param gameState: the current state of the game (GameState)
        :return: action, score
        '''
        
        if agent_index >= gameState.getNumAgents():
            agent_index = 0
            curr_depth += 1

        if curr_depth == self.depth:
            return None, self.evaluationFunction(gameState)

        if agent_index == 0:
            return self.max_value(curr_depth, agent_index, gameState)
        else:
            return self.exp_value(curr_depth, agent_index, gameState)

    def max_value(self, curr_depth, agent_index, gameState):
        best_score, best_action = None, None
        for action in gameState.getLegalActions(agent_index):
            next_game_state = gameState.generateSuccessor(agent_index, action)
            _, score = self.expectimax(curr_depth, agent_index + 1, next_game_state)

            if best_score is None or score > best_score:
                best_score = score
                best_action = action
        if best_score is None:
            return None, self.evaluationFunction(gameState)
        return best_action, best_score

    def exp_value(self, curr_depth, agent_index, gameState):
        best_score, best_action = None, None
        ghostActions = gameState.getLegalActions(agent_index)
        if len(ghostActions) != 0:
            prob = 1.0 / len(ghostActions)
            for action in ghostActions:
                next_game_state = gameState.generateSuccessor(agent_index, action)
                _, score = self.expectimax(curr_depth, agent_index + 1, next_game_state)

                if best_score is None:
                    best_score = 0.0
                best_score += prob * score
                best_action = action
        if best_score is None:
            return None, self.evaluationFunction(gameState)
        return best_action, best_score


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    WIN_FACTOR = 50000
    LOST_FACTOR = -50000
    FOOD_COUNT_FACTOR = 1000000
    FOOD_DISTANCE_FACTOR = 1000
    CAPSULES_FACTOR = 10000

    food_distance = distance_to_nearest(currentGameState, currentGameState.getFood().asList())
    ghost_distance = distance_to_nearest(currentGameState, currentGameState.getGhostPositions())
    food_count = currentGameState.getNumFood()
    capsules_count = len(currentGameState.getCapsules())

    food_distance_value = 1 / food_distance * FOOD_DISTANCE_FACTOR
    ghost_value = ghost_distance
    food_count_value = 1 / (food_count + 1) * FOOD_COUNT_FACTOR
    capsules_count_value = 1 / (capsules_count + 1) * CAPSULES_FACTOR

    end_value = 0

    if currentGameState.isLose():
        end_value += LOST_FACTOR
    elif currentGameState.isWin():
        end_value += WIN_FACTOR

    return food_count_value + ghost_value + food_distance_value + capsules_count_value + end_value


def distance_to_nearest(game_state, positions):
    if len(positions) == 0:
        return math.inf
    pacman_pos = game_state.getPacmanPosition()
    distances = [manhattanDistance(pacman_pos, position) for position in positions]
    return min(distances)
    # return util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
