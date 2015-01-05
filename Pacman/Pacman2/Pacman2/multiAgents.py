# multiAgents.py
# --------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from util import manhattanDistance
from game import Directions
import random, util
import sys

from game import Agent

#
# Utility functions.
#

def IsAdjacent(Position1, Position2):
  x = 0
  y = 1

  Adj1 = (Position1[x] - 1, Position1[y])
  Adj2 = (Position1[x] + 1, Position1[y])
  Adj3 = (Position1[x], Position1[y] - 1)
  Adj4 = (Position1[x], Position1[y] + 1)

  if (   Adj1 == Position2
      or Adj2 == Position2
      or Adj3 == Position2
      or Adj4 == Position2):

    return True;

  return False;

class ReflexAgent(Agent):
  """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
  """
  PastStates = {}
  OldStatePenalty = -.5

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
    
    #
    # Save chosen state (for now a state is pacman position).
    #

    ChosenState = gameState.generatePacmanSuccessor(legalMoves[chosenIndex])
    Position = ChosenState.getPacmanPosition()
    if Position in self.PastStates:
      self.PastStates[Position] += self.OldStatePenalty
    else:
      self.PastStates[Position] = self.OldStatePenalty

    return legalMoves[chosenIndex]

  def evaluationFunction(self, currentGameState, action):
    """
    Design a better evaluation function here.

    The evaluation function takes in the current and proposed successor
    GameStates (pacman.py) and returns a number, where higher numbers are better.

    The code below extracts some useful information from the state, like the
    remaining food (oldFood) and Pacman position after moving (newPos).
    newScaredTimes holds the number of moves that each ghost will remain
    scared because of Pacman having eaten a power pellet.

    Print out these variables to see what you're getting, then combine them
    to create a masterful evaluation function.
    """
    # Useful information you can extract from a GameState (pacman.py)
    successorGameState = currentGameState.generatePacmanSuccessor(action)
    newPos = successorGameState.getPacmanPosition()
    oldFood = currentGameState.getFood()
    newGhostStates = successorGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    "*** YOUR CODE HERE ***"
    score = successorGameState.getScore()

    #
    # Avoid ghosts.
    #
    
    GhostPenalty = -11
    for Ghost in newGhostStates:
      if IsAdjacent(newPos, Ghost.getPosition()):
        score += GhostPenalty

    #
    # Prefer to explore new states.
    #

    if newPos in self.PastStates:
      score += self.PastStates[newPos]
      
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

  def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
    self.index = 0 # Pacman is always agent index 0
    self.evaluationFunction = util.lookup(evalFn, globals())
    self.depth = int(depth)
    self.numAgents = 0

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

      Directions.STOP:
        The stop direction, which is always legal

      gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

      gameState.getNumAgents():
        Returns the total number of agents in the game
    """
    "*** YOUR CODE HERE ***"
    
    #
    # min max starting with Pacman (Max) actions.
    #

    self.numAgents = gameState.getNumAgents()
    return self.miniMax(gameState)
 
  def miniMax(self, gameState):
    
    #
    # Assume agentIndex is 0 when miniMax is called.
    #

    agentIndex = 0
    actions = gameState.getLegalActions(agentIndex)
    if Directions.STOP in actions:
      actions.remove(Directions.STOP)
    depth = 0
    maxCost = - sys.maxint
    maxActionDict = {}
    for action in actions:
      successor = gameState.generateSuccessor(agentIndex, action)
      tempMax = self.minValue(successor, agentIndex + 1, depth)
      if tempMax > maxCost:
        maxCost = tempMax
        maxActionDict[maxCost] = action

    #print "BestAction " + str(maxActionDict) + " max: " + str(maxCost)
    return maxActionDict[maxCost]

  def minValue(self, gameState, agentIndex, depth):
    if self.terminalState(gameState, depth):
      return self.evaluationFunction(gameState)

    minCost = sys.maxint
    preMinCost = minCost
    actions = gameState.getLegalActions(agentIndex)

    #
    # Does pacman or a ghost have next move.
    #

    if agentIndex == (self.numAgents - 1):
      depth += 1

    for action in actions:
      successor = gameState.generateSuccessor(agentIndex, action)   
      if agentIndex == (self.numAgents - 1):
        minCost = min(minCost, self.maxValue(successor, self.nextAgentIndex(agentIndex), depth))
      else:
        minCost = min(minCost, self.minValue(successor, self.nextAgentIndex(agentIndex), depth))

      if minCost < preMinCost:
        preMinCost = minCost
        minAction = action

    #print "Min agent(" +str(agentIndex) +")- depth: " + str(depth) + " Min: " + str(minCost) + " Action: " + str(minAction)
    return minCost

  def maxValue(self, gameState, agentIndex, depth):
    if self.terminalState(gameState, depth):
      return self.evaluationFunction(gameState)

    actions = gameState.getLegalActions(agentIndex)
    if Directions.STOP in actions:
      actions.remove(Directions.STOP)

    maxCost = -sys.maxint
    preMaxCost = maxCost
    for action in actions:
      successor = gameState.generateSuccessor(agentIndex, action)
      maxCost = max(maxCost, self.minValue(successor, agentIndex + 1, depth))
      if preMaxCost < maxCost:
        preMaxCost = maxCost
        maxAction = action

    #print "Max agent(" +str(agentIndex) +")depth: " + str(depth) + " Max: " + str(maxCost) + " Action: " + str(maxAction)
    return maxCost

  def nextAgentIndex(self, index):
    if index + 1 == self.numAgents:
      return 0
    return index + 1

  def terminalState(self, gameState, depth):
    if gameState.isWin() or gameState.isLose() or depth >= self.depth:
      return True

    return False

class AlphaBetaAgent(MultiAgentSearchAgent):
  """
    Your minimax agent with alpha-beta pruning (question 3)
  """

  def getAction(self, gameState):
    """
      Returns the minimax action using self.depth and self.evaluationFunction
    """
    "*** YOUR CODE HERE ***"
    self.numAgents = gameState.getNumAgents()
    return self.alphabeta(gameState)
 
  def alphabeta(self, gameState):
    
    #
    # Assume agentIndex is 0 when miniMax is called.
    #

    alpha = -sys.maxint
    beta = sys.maxint
    agentIndex = 0
    actions = gameState.getLegalActions(agentIndex)
    if Directions.STOP in actions:
      actions.remove(Directions.STOP)
    depth = 0
    maxCost = - sys.maxint
    maxActionDict = {}
    for action in actions:
      successor = gameState.generateSuccessor(agentIndex, action)
      tempMax = self.minValue(successor, agentIndex + 1, depth, alpha, beta)
      if tempMax > maxCost:
        maxCost = tempMax
        maxActionDict[maxCost] = action
      if maxCost >= beta:
        break
      alpha = max(alpha, maxCost)

    #print "BestAction " + str(maxActionDict) + " max: " + str(maxCost)
    return maxActionDict[maxCost]

  def minValue(self, gameState, agentIndex, depth, alpha, beta):
    if self.terminalState(gameState, depth):
      return self.evaluationFunction(gameState)

    minCost = sys.maxint
    preMinCost = minCost
    actions = gameState.getLegalActions(agentIndex)

    #
    # Does pacman or a ghost have next move.
    #

    if agentIndex == (self.numAgents - 1):
      depth += 1

    for action in actions:
      successor = gameState.generateSuccessor(agentIndex, action)   
      if agentIndex == (self.numAgents - 1):
        minCost = min(minCost, self.maxValue(successor, self.nextAgentIndex(agentIndex), depth, alpha, beta))
      else:
        minCost = min(minCost, self.minValue(successor, self.nextAgentIndex(agentIndex), depth, alpha, beta))

      if minCost < preMinCost:
        preMinCost = minCost
        minAction = action
      if minCost < alpha:
        break
      beta = min(beta, minCost)

    #print "Min agent(" +str(agentIndex) +")- depth: " + str(depth) + " Min: " + str(minCost) + " Action: " + str(minAction)
    return minCost

  def maxValue(self, gameState, agentIndex, depth, alpha, beta):
    if self.terminalState(gameState, depth):
      return self.evaluationFunction(gameState)

    actions = gameState.getLegalActions(agentIndex)
    if Directions.STOP in actions:
      actions.remove(Directions.STOP)

    maxCost = -sys.maxint
    preMaxCost = maxCost
    for action in actions:
      successor = gameState.generateSuccessor(agentIndex, action)
      maxCost = max(maxCost, self.minValue(successor, agentIndex + 1, depth, alpha, beta))
      if preMaxCost < maxCost:
        preMaxCost = maxCost
        maxAction = action
      if maxCost > beta:
        break
      alpha = max(maxCost, alpha)

    #print "Max agent(" +str(agentIndex) +")depth: " + str(depth) + " Max: " + str(maxCost) + " Action: " + str(maxAction)
    return maxCost

  def nextAgentIndex(self, index):
    if index + 1 == self.numAgents:
      return 0
    return index + 1

  def terminalState(self, gameState, depth):
    if gameState.isWin() or gameState.isLose() or depth >= self.depth:
      return True

    return False
class ExpectimaxAgent(MinimaxAgent):
  """
    Your expectimax agent (question 4)
  """
  #def result(self, gameState, agentIndex, action):
  #  return gameState.generateSuccessor(agentIndex, action)

  #def utility(self, gameState):
  #  return self.evaluationFunction(gameState)

  #def expectiMax(gameState, agentIndex, depth):
     

  #def getAction(self, gameState):
   #util.raiseNotDefined()

  def minValue(self, gameState, agentIndex, depth):
    if self.terminalState(gameState, depth):
      return self.evaluationFunction(gameState)

    v = 0
    actions = gameState.getLegalActions(agentIndex)

    #
    # Does pacman or a ghost have next move.
    #

    if agentIndex == (self.numAgents - 1):
      depth += 1

    for action in actions:
      successor = gameState.generateSuccessor(agentIndex, action)   
      if agentIndex == (self.numAgents - 1):
        v += self.maxValue(successor, self.nextAgentIndex(agentIndex), depth)
      else:
        v += self.minValue(successor, self.nextAgentIndex(agentIndex), depth)


    #print "Min agent(" +str(agentIndex) +")- depth: " + str(depth) + " Min: " + str(minCost) + " Action: " + str(minAction)
    return v/len(gameState.getLegalActions(agentIndex))

def getNeighbors(position):
    neighbors = []
    neighbors.append((position[0]+1, position[1]))
    neighbors.append((position[0]-1, position[1]))
    neighbors.append((position[0], position[1]-1))
    neighbors.append((position[0], position[1]+1))
    return neighbors

pastStates = {}
pastStatePenalty = -.3
   
def betterEvaluationFunction(currentGameState):
  """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
  """
  "*** YOUR CODE HERE ***"

  score = 0

  #
  # Grab important game states.
  #

  currentPosition = currentGameState.getPacmanPosition()
  currentFood = currentGameState.getFood()
  currentGhostStates = currentGameState.getGhostStates()
  currentGhostPositions = [ghostState.getPosition() for ghostState in currentGhostStates]
  currentScaredTimes = [ghostState.scaredTimer for ghostState in currentGhostStates]
  
  #
  # Get ghost distances from pacman.
  #

  ghostDistances = [manhattanDistance(ghostPosition, currentPosition) for ghostPosition in currentGhostPositions]

  #
  # Get ghost scared times.
  #

  for time in currentScaredTimes:
      score += time          

  #
  # Find food distance from pacman and empty spaces next to food (don't want to leave single food).
  #

  foodList = currentFood.asList()
  wallList = currentGameState.getWalls().asList()   
  foodNeighborsEmpty = 0
  foodDistances = []
  for food in foodList:
      neighbors = getNeighbors(food)
      for foodNeighbor in neighbors:
          if foodNeighbor not in foodList and foodNeighbor not in wallList:
              foodNeighborsEmpty += 1

      foodDistances += [manhattanDistance(currentPosition,food)]
     
  inverseFoodDistance = 0
  if foodDistances:
      inverseFoodDistance = 1.0 / min(foodDistances)

  #
  # Don't leave food behind (because we'll have to spend time going back to get it)
  #

  score += currentGameState.getScore()
  score -= foodNeighborsEmpty * 8

  #
  # Stay away from ghosts, closest ghost matters the most, get closest food.
  #

  score += (min(ghostDistances) * inverseFoodDistance * 5)
 

  #print "st:",currentScaredTimes, " min(foodDistance):", min(foodDistances), "inverse ", inverseFoodDistance, " foodNeighborsEmpty:", foodNeighborsEmpty, " ghostdist:", ghostDistances
  #print "score: ", score
  
  return score

# Abbreviation
better = betterEvaluationFunction

class ContestAgent(MultiAgentSearchAgent):
  """
    Your agent for the mini-contest
  """

  def getAction(self, gameState):
    """
      Returns an action.  You can use any method you want and search to any depth you want.
      Just remember that the mini-contest is timed, so you have to trade off speed and computation.

      Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
      just make a beeline straight towards Pacman (or away from him if they're scared!)
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

