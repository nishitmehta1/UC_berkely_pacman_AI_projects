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
import random, util

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
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

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
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        food = currentGameState.getFood()
        here = list(successorGameState.getPacmanPosition())  #Retrieve Pacman Position
        list_food = food.asList() #Get the list of food items
        dist = float("-Inf")

        if action == 'Stop':
            return float("-Inf")

        for state in newGhostStates:
            if state.getPosition() == tuple(here) and (state.scaredTimer == 0):
                return float("-Inf")

        for i in list_food:
            temp_dist = -1 * (manhattanDistance(here, i))
            if (temp_dist > dist):
                dist = temp_dist

        return dist  #Return the distance from food
        # return successorGameState.getScore()

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

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        def find_minimum_value(gameState, depth, agentcounter):
          least_value = ["", float("inf")]
          enemy_action = gameState.getLegalActions(agentcounter)

          if not enemy_action:
              return self.evaluationFunction(gameState)

          for enem_action in enemy_action:
              current_state = gameState.generateSuccessor(agentcounter, enem_action)
              current = min_max(current_state, depth, agentcounter + 1)
              if type(current) is not list:
                  next_ = current
              else:
                  next_ = current[1]
              if next_ < least_value[1]:
                  least_value = [enem_action, next_]
          return least_value

        def find_max_value(gameState, depth, agentcounter):
          maximum = ["", -float("inf")]
          actions = gameState.getLegalActions(agentcounter)

          if not actions:
              return self.evaluationFunction(gameState)

          for action in actions:
              current_state = gameState.generateSuccessor(agentcounter, action)
              current = min_max(current_state, depth, agentcounter + 1)
              if type(current) is not list:
                  next_ = current
              else:
                  next_ = current[1]
              if next_ > maximum[1]:
                  maximum = [action, next_]
          return maximum

        def min_max(gameState, depth, agentcounter):
          if agentcounter >= gameState.getNumAgents():
              depth += 1
              agentcounter = 0

          if (depth == self.depth or gameState.isWin() or gameState.isLose()):
              return self.evaluationFunction(gameState)
          elif (agentcounter == 0):
              return find_max_value(gameState, depth, agentcounter)
          else:
              return find_minimum_value(gameState, depth, agentcounter)

        actionsList = min_max(gameState, 0, 0)
        return actionsList[0]
        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """
    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def min_value(gameState, depth, agentcounter, x, y):
          minimum1 = ["", float("inf")]
          enemy_actions = gameState.getLegalActions(agentcounter)

          if not enemy_actions:
              return self.evaluationFunction(gameState)

          for movement in enemy_actions:
              current_state = gameState.generateSuccessor(agentcounter, movement)
              current = min_max(current_state, depth, agentcounter + 1, x, y)

              if type(current) is not list:
                  next_value = current
              else:
                  next_value = current[1]

              if next_value < minimum1[1]:
                  minimum1 = [movement, next_value]
              if next_value < x:
                  return [movement, next_value]
              y = min(y, next_value)
          return minimum1

        def max_value(gameState, depth, agentcounter, a, b):
            maximum = ["", -float("inf")]
            movements = gameState.getLegalActions(agentcounter)

            if not movements:
                return self.evaluationFunction(gameState)

            for movement in movements:
                current_state = gameState.generateSuccessor(agentcounter, movement)
                current = min_max(current_state, depth, agentcounter + 1, a, b)

                if type(current) is not list:
                    next_value = current
                else:
                    next_value = current[1]

                if next_value > maximum[1]:
                    maximum = [movement, next_value]
                if next_value > b:
                    return [movement, next_value]
                a = max(a, next_value)
            return maximum

        def min_max(gameState, depth, agentcounter, a, b):
            if agentcounter >= gameState.getNumAgents():
                depth += 1
                agentcounter = 0

            if (depth == self.depth or gameState.isWin() or gameState.isLose()):
                return self.evaluationFunction(gameState)
            elif (agentcounter == 0):
                return max_value(gameState, depth, agentcounter, a, b)
            else:
                return min_value(gameState, depth, agentcounter, a, b)

        actionsList = min_max(gameState, 0, 0, -float("inf"), float("inf"))
        return actionsList[0]

        util.raiseNotDefined()

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
        def find_e(gameState, depth, agentcounter):
            enemy_actions = gameState.getLegalActions(agentcounter)
            prob = (1.000000 / len(enemy_actions))
            max_e = ["", 0]

            if not enemy_actions:
                return self.evaluationFunction(gameState)

            for action_e in enemy_actions:
                current_state = gameState.generateSuccessor(agentcounter, action_e)
                current = expect_max(current_state, depth, agentcounter + 1)
                if type(current) is list:
                    next_value = current[1]
                else:
                    next_value = current
                max_e[0] = action_e
                max_e[1] += next_value * prob
            return max_e

        def max_value(gameState, depth, agentcounter):
            actions = gameState.getLegalActions(agentcounter)
            max1 = ["", -float("inf")]

            if not actions:
                return self.evaluationFunction(gameState)

            for action in actions:
                current_state = gameState.generateSuccessor(agentcounter, action)
                current = expect_max(current_state, depth, agentcounter + 1)
                if type(current) is not list:
                    next_value = current
                else:
                    next_value = current[1]
                if next_value > max1[1]:
                    max1 = [action, next_value]
            return max1

        def expect_max(gameState, depth, agentcounter):
            if agentcounter >= gameState.getNumAgents():
                agentcounter = 0
                depth += 1

            if (depth == self.depth or gameState.isWin() or gameState.isLose()):
                return self.evaluationFunction(gameState)
            elif (agentcounter == 0):
                return max_value(gameState, depth, agentcounter)
            else:
                return find_e(gameState, depth, agentcounter)

        actionsList = expect_max(gameState, 0, 0)
        return actionsList[0]


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).
    """
    "*** YOUR CODE HERE ***"
    now_state = list(currentGameState.getPacmanPosition())
    food_position = currentGameState.getFood().asList()
    food_items = []

    for food in food_position:
        pacman_dist_food = manhattanDistance(now_state, food)
        food_items.append(-1 * pacman_dist_food)

    if not food_items:
        food_items.append(0)

    return currentGameState.getScore() + max(food_items)
    # util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction