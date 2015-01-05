# search.py
# ---------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

"""
In search.py, you will implement generic search algorithms which are called 
by Pacman agents (in searchAgents.py).
"""

import util
import sys
from util import Queue
from util import Stack
from util import PriorityQueue
from util import PriorityQueueWithFunction
from game import Directions

DebugEnabled = 0

def DbgPrint(str):
  if DebugEnabled:
    print str

class PacmanSolver:
  """
  This class is a universal solver. It uses a priority queue with a
  priority function when initialized for managing the fringe.
  """

  def GraphSearch(self, Problem, PriorityFunction):
    """
    Given a prolbem object from the pacman game this function returns a path
    to the goal state using the priority queue with function to manage the 
    fringe states.

    Each node is defined as follows:

    Node: (state, cost so far)

    """
    Fringe = PriorityQueue()
    Parent = {}
    Node = (Problem.getStartState(), [])
    Parent[Node[0]] = ""
    while not Problem.isGoalState(Node[0]):
      for Successor in Problem.getSuccessors(Node[0]):
        if Successor[0] not in Parent.keys():
          SuccessorNode = (Successor[0], Node[1] + [Successor[1]])
          Fringe.push(SuccessorNode, PriorityFunction(Problem, SuccessorNode))
          Parent[SuccessorNode[0]] = Node
      if Fringe.isEmpty():
        break
      DbgPrint("\tFringe: " + str(Fringe))
      Node = Fringe.pop()
      DbgPrint("\tPoped node: " + str(Node))
    if not Problem.isGoalState(Node[0]):
      raise "No Path found"

    return Node[1]

"""
Below is a list of priority queue funtions that implement different search paths
approaches
"""
DepthFirstSearchCount = sys.maxint
def DepthFirstSearch(Problem, Node):
  """
  This function returns monotonically decreasing values for each item added to a
  priority queue to simulate using a stack data structure, resulting in a depth
  first search.
  """

  global DepthFirstSearchCount
  Temp = DepthFirstSearchCount
  DepthFirstSearchCount -= 1
  return Temp

BreadthFirstSearchCount = 0
def BreadthFirstSearch(Problem, Node):
  """
  This function returns a monotonically increasing value for each item, resulting
  in a breadth first search cost structure.
  """
  global BreadthFirstSearchCount
  Temp = BreadthFirstSearchCount
  BreadthFirstSearchCount += 1
  return Temp


def UniformCostSearch(Problem, Node):
  """
  This function returns the sum of the cost of the path so far, resulting in
  uniform cost search.
  """
  return Problem.getCostOfActions(Node[1])

AStartHeuristic = 0
def AStarSearch(Problem, Node):
  """
  This funciton returns the cost so far plus the estimate from the huristic,
  resulting in A* search.
  """
  return Problem.getCostOfActions(Node[1]) + AStartHeuristic(Node[0], Problem)

class SearchProblem:
  """
  This class outlines the structure of a search problem, but doesn't implement
  any of the methods (in object-oriented terminology: an abstract class).
  
  You do not need to change anything in this class, ever.
  """
  
  def getStartState(self):
     """
     Returns the start state for the search problem 
     """
     util.raiseNotDefined()
    
  def isGoalState(self, state):
     """
       state: Search state
    
     Returns True if and only if the state is a valid goal state
     """
     util.raiseNotDefined()

  def getSuccessors(self, state):
     """
       state: Search state
     
     For a given state, this should return a list of triples, 
     (successor, action, stepCost), where 'successor' is a 
     successor to the current state, 'action' is the action
     required to get there, and 'stepCost' is the incremental 
     cost of expanding to that successor
     """
     util.raiseNotDefined()

  def getCostOfActions(self, actions):
     """
      actions: A list of actions to take
 
     This method returns the total cost of a particular sequence of actions.  The sequence must
     be composed of legal moves
     """
     util.raiseNotDefined()
           

def tinyMazeSearch(problem):
  """
  Returns a sequence of moves that solves tinyMaze.  For any other
  maze, the sequence of moves will be incorrect, so only use this for tinyMaze
  """
  from game import Directions
  s = Directions.SOUTH
  w = Directions.WEST
  path = [s,s,w,s,w,w,s,w]
  return path

def depthFirstSearch(problem):
  """
  Search the deepest nodes in the search tree first [p 85].
  
  Your search algorithm needs to return a list of actions that reaches
  the goal.  Make sure to implement a graph search algorithm [Fig. 3.7].
  
  To get started, you might want to try some of these simple commands to
  understand the search problem that is being passed in:
  
  print "Start:", problem.getStartState()
  print "Is the start a goal?", problem.isGoalState(problem.getStartState())
  print "Start's successors:", problem.getSuccessors(problem.getStartState())
  """
  "*** YOUR CODE HERE ***"
  Solver = PacmanSolver()
  Path = Solver.GraphSearch(problem, DepthFirstSearch)
  return Path 

def breadthFirstSearch(problem):
  "Search the shallowest nodes in the search tree first. [p 81]"
  "*** YOUR CODE HERE ***"

  Solver = PacmanSolver()
  Path = Solver.GraphSearch(problem, BreadthFirstSearch)
  return Path 

def uniformCostSearch(problem):
  "Search the node of least total cost first. "
  "*** YOUR CODE HERE ***"

  Solver = PacmanSolver()
  Path = Solver.GraphSearch(problem, UniformCostSearch)
  return Path 


def nullHeuristic(state, problem=None):
  """
  A heuristic function estimates the cost from the current state to the nearest
  goal in the provided SearchProblem.  This heuristic is trivial.
  """
  return 0

def aStarSearch(problem, heuristic=nullHeuristic):
  "Search the node that has the lowest combined cost and heuristic first."
  "*** YOUR CODE HERE ***"
  DbgPrint("\nAstart start\n")
  Solver = PacmanSolver()
  global AStartHeuristic
  AStartHeuristic = heuristic
  Path = Solver.GraphSearch(problem, AStarSearch)
  DbgPrint("\nAstar stop\n")
  return Path
  
    
  
# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch