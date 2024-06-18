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


import random

import util
from game import Agent, Directions
from util import manhattanDistance


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
        bestIndices = [
            index for index in range(len(scores)) if scores[index] == bestScore
        ]
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
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        # Initialize the evaluation score with the game score
        score = successorGameState.getScore()

        # Calculate the reciprocal of distance to the nearest food
        foodDistances = [manhattanDistance(newPos, food) for food in newFood.asList()]
        if foodDistances:
            score += max(1.0 / min(foodDistances), 0)

        # Calculate the reciprocal of distance to the nearest active ghost
        activeGhostDistances = [manhattanDistance(newPos, ghost.getPosition()) for ghost in newGhostStates if ghost.scaredTimer == 0]
        if activeGhostDistances:
            score -= max(2.0 / (min(activeGhostDistances) + 1e-10), 0)


        # Add the number of scared ghosts as a bonus to the score
        score += sum(newScaredTimes)

        # Consider the number of remaining capsules as a bonus
        capsulesLeft = len(currentGameState.getCapsules())
        score += capsulesLeft * 10

        return score
        

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

    def __init__(self, evalFn="scoreEvaluationFunction", depth="2"):
        self.index = 0  # Pacman is always agent index 0
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

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        def minimax(agent, depth, gameState):
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState)

            if agent == 0:  # Pacman is maximizing agent
                return max(minimax(1, depth, gameState.generateSuccessor(agent, newState)) for newState in gameState.getLegalActions(agent))
            else:  # Ghosts are minimizing agents
                nextAgent = agent + 1  
                if gameState.getNumAgents() == nextAgent:
                    nextAgent = 0
                if nextAgent == 0:
                     depth += 1
                return min(minimax(nextAgent, depth, gameState.generateSuccessor(agent, newState)) for newState in gameState.getLegalActions(agent))

        # Pacman starts first
        return max(gameState.getLegalActions(0), key=lambda x: minimax(1, 0, gameState.generateSuccessor(0, x)))
        util.raiseNotDefined()


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        def alphaBeta(agent, depth, gameState, alpha, beta):
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState)

            if agent == 0:  # Pacman is maximizing agent
                value = float("-inf")
                for action in gameState.getLegalActions(agent):
                    value = max(value, alphaBeta(1, depth, gameState.generateSuccessor(agent, action), alpha, beta))
                    if value > beta:
                        return value
                    alpha = max(alpha, value)
                return value
            else:  # Ghosts are minimizing agents
                value = float("inf")
                nextAgent = agent + 1  
                if gameState.getNumAgents() == nextAgent:
                    nextAgent = 0
                if nextAgent == 0:
                     depth += 1
                for action in gameState.getLegalActions(agent):
                    value = min(value, alphaBeta(nextAgent, depth, gameState.generateSuccessor(agent, action), alpha, beta))
                    if value < alpha:
                        return value
                    beta = min(beta, value)
                return value

        # Pacman starts first
        alpha = float("-inf")
        beta = float("inf")
        bestAction = Directions.STOP
        for action in gameState.getLegalActions(0):
            value = alphaBeta(1, 0, gameState.generateSuccessor(0, action), alpha, beta)
            if value > alpha:
                alpha = value
                bestAction = action
        return bestAction



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
        def expectimax(agent, depth, gameState):
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState)

            if agent == 0:  # Pacman is maximizing agent
                return max(expectimax(1, depth, gameState.generateSuccessor(agent, newState)) for newState in gameState.getLegalActions(agent))
            else:  # Ghosts are expected agents
                nextAgent = agent + 1  
                if gameState.getNumAgents() == nextAgent:
                    nextAgent = 0
                if nextAgent == 0:
                     depth += 1
                averageValue = sum(expectimax(nextAgent, depth, gameState.generateSuccessor(agent, newState)) for newState in gameState.getLegalActions(agent))
                return averageValue / len(gameState.getLegalActions(agent))

        # Pacman starts first
        return max(gameState.getLegalActions(0), key=lambda x: expectimax(1, 0, gameState.generateSuccessor(0, x)))
        util.raiseNotDefined()


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).
    DESCRIPTION: <write something here so we know what you did>
    """
    # Constants
    LOSE_SCORE = -9999999999999
    WIN_SCORE = 9999999999999

    # Check for terminal states
    if currentGameState.isLose():
        return LOSE_SCORE
    if currentGameState.isWin():
        return WIN_SCORE

    # Extract game state info
    pacmanPos = currentGameState.getPacmanPosition()
    foodPositions = currentGameState.getFood().asList()
    ghostStates = currentGameState.getGhostStates()

    # Compute distances to ghosts and food
    ghostDistances = [manhattanDistance(pacmanPos, ghostState.getPosition()) for ghostState in ghostStates]
    foodDistances = [manhattanDistance(pacmanPos, foodPos) for foodPos in foodPositions]

    # Compute scared times
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]

    # Compute features
    nearestGhostDistance = min(ghostDistances) if ghostDistances else 0
    nearestFoodDistance = min(foodDistances) if foodDistances else 0
    averageScaredTime = sum(scaredTimes) / len(scaredTimes) if scaredTimes else 0

    # Compute bonus score
    bonusScore = 0
    if nearestGhostDistance < 3:
        bonusScore -= 1200 / nearestGhostDistance
        if nearestGhostDistance == 1:
            bonusScore -= 5000
    if nearestFoodDistance > 0:
        bonusScore += 5000 / nearestFoodDistance

    # Compute final score
    result = currentGameState.getScore() + bonusScore - 7 * nearestFoodDistance - averageScaredTime - 5000 * len(foodPositions)
    return result

# Abbreviation
better = betterEvaluationFunction