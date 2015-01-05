# valueIterationAgents.py
# -----------------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import mdp, util
from operator import itemgetter
from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
  """
      * Please read learningAgents.py before reading this.*

      A ValueIterationAgent takes a Markov decision process
      (see mdp.py) on initialization and runs value iteration
      for a given number of iterations using the supplied
      discount factor.
  """
  def __init__(self, mdp, discount = 0.9, iterations = 100):
    """
      Your value iteration agent should take an mdp on
      construction, run the indicated number of iterations
      and then act according to the resulting policy.
    
      Some useful mdp methods you will use:
          mdp.getStates()
          mdp.getPossibleActions(state)
          mdp.getTransitionStatesAndProbs(state, action)
          mdp.getReward(state, action, nextState)
    """
    self.mdp = mdp
    self.discount = discount
    self.iterations = iterations
    self.values = util.Counter() # A Counter is a dict with default 0
    states = mdp.getStates()
    self.policy = util.Counter()
    valuesNew = util.Counter((element, 0) for element in states)
    self.values = util.Counter((element, 0) for element in states)
    for i in range(0, iterations):
      for state in self.values:
        actions = mdp.getPossibleActions(state)
        actionExpectedValues = []
        for action in actions:
          probabilities = mdp.getTransitionStatesAndProbs(state, action)
          actionExpectedValueSum = 0
          for probability in probabilities:
            reward = mdp.getReward(state, action, (0,0))
            actionExpectedValueSum += probability[1] * (reward + discount * self.values[probability[0]])

          actionExpectedValues.append(actionExpectedValueSum)

        if actionExpectedValues:
            maxActionIndex, maxActionValue = max(enumerate(actionExpectedValues), key=itemgetter(1))
            maxAction = actions[maxActionIndex]
            valuesNew[state] =  maxActionValue
            self.policy[state] = maxAction
        else:
            valuesNew[state] = self.values[state]

      self.values = valuesNew.copy()
      valuesNew = {}

    self.policy['TERMINAL_STATE'] = 'south'
    
  def getValue(self, state):
    """
      Return the value of the state (computed in __init__).
    """
    return self.values[state]


  def getQValue(self, state, action):
    """
      The q-value of the state action pair
      (after the indicated number of value iteration
      passes).  Note that value iteration does not
      necessarily create this quantity and you may have
      to derive it on the fly.
    """
    "*** YOUR CODE HERE ***"
    probabilities = self.mdp.getTransitionStatesAndProbs(state, action)
    actionExpectedValueSum = 0
    for probability in probabilities:
        actionExpectedValueSum += probability[1] * (self.values[probability[0]])

    return actionExpectedValueSum
    

  def getPolicy(self, state):
    """
      The policy is the best action in the given state
      according to the values computed by value iteration.
      You may break ties any way you see fit.  Note that if
      there are no legal actions, which is the case at the
      terminal state, you should return None.
    """
    "*** YOUR CODE HERE ***"
    return self.policy[state]

  def getAction(self, state):
    "Returns the policy at the state (no exploration)."
    return self.getPolicy(state)
  
