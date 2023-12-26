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
  def __init__(self):
    self.lastPositions = []
    self.dc = None

  def getAction(self, gameState):
    """
    getAction chooses among the best options according to the evaluation function.

    getAction takes a GameState and returns some Directions.X for some X in the set {North, South, West, East, Stop}
    ------------------------------------------------------------------------------
    Description of GameState and helper functions:

    A GameState specifies the full game state, including the food, capsules,
    agent configurations and score changes. In this function, the |gameState| argument 
    is an object of GameState class. Following are a few of the helper methods that you 
    can use to query a GameState object to gather information about the present state 
    of Pac-Man, the ghosts and the maze.
    
    gameState.getLegalActions(): 
        Returns the legal actions for the agent specified. Returns Pac-Man's legal moves by default.

    gameState.generateSuccessor(agentIndex, action): 
        Returns the successor state after the specified agent takes the action. 
        Pac-Man is always agent 0.

    gameState.getPacmanState():
        Returns an AgentState object for pacman (in game.py)
        state.configuration.pos gives the current position
        state.direction gives the travel vector

    gameState.getGhostStates():
        Returns list of AgentState objects for the ghosts

    gameState.getNumAgents():
        Returns the total number of agents in the game

    gameState.getScore():
        Returns the score corresponding to the current state of the game
        It corresponds to Utility(s)

    
    The GameState class is defined in pacman.py and you might want to look into that for 
    other helper methods, though you don't need to.
    """
    # Collect legal moves and successor states
    legalMoves = gameState.getLegalActions()

    # Choose one of the best actions
    scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
    bestScore = max(scores)
    bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
    chosenIndex = random.choice(bestIndices) # Pick randomly among the best

    return legalMoves[chosenIndex]

  def evaluationFunction(self, currentGameState, action):
    """
    The evaluation function takes in the current and proposed successor
    GameStates (pacman.py) and returns a number, where higher numbers are better.

    The code below extracts some useful information from the state, like the
    remaining food (oldFood) and Pacman position after moving (newPos).
    newScaredTimes holds the number of moves that each ghost will remain
    scared because of Pacman having eaten a power pellet.
    """
    # Useful information you can extract from a GameState (pacman.py)
    successorGameState = currentGameState.generatePacmanSuccessor(action)
    newPos = successorGameState.getPacmanPosition()
    oldFood = currentGameState.getFood()
    newGhostStates = successorGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    return successorGameState.getScore()

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

######################################################################################
# Problem 1a: implementing minimax

class MinimaxAgent(MultiAgentSearchAgent):
  """
    Your minimax agent (problem 1)
  """

  def getValue(self, gameState, depth, index):
    # depth 도달 시 evaluate
    if depth == 0:
      return self.evaluationFunction(gameState)
    # 더이상 이동할 곳이 없을 때, 이겼을 때, 졌을 때 score 반환 
    actions = gameState.getLegalActions(index)
    if actions is None or gameState.isWin() or gameState.isLose():
      return gameState.getScore()
    # 이외에 재귀적으로 Value 찾기. 단, agent index에 따라 max, min 바꿔줘야 함.
    # actions 중 가장 높은 Value 주는 action
    numOfAgents = gameState.getNumAgents()
    scores = [self.getValue(gameState.generateSuccessor(index, action), 
                        depth if index < numOfAgents-1 else depth-1, 
                        (index+1)%numOfAgents)
                        for action in actions]
    bestScore = max(scores) if index==0 else min(scores)
    return bestScore

  def getAction(self, gameState):
    """
      Returns the minimax action from the current gameState using self.depth
      and self.evaluationFunction. Terminal states can be found by one of the following: 
      pacman won, pacman lost or there are no legal moves. 

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

      gameState.getScore():
        Returns the score corresponding to the current state of the game
        It corresponds to Utility(s)
    
      gameState.isWin():
        Returns True if it's a winning state
    
      gameState.isLose():
        Returns True if it's a losing state

      self.depth:
        The depth to which search should continue
    """
    # 현재 상태에서 가능한 actions에 대해 minimax를 반환하는 V들 중 가장 높은 값을 주는 action을 취함 
    actions = gameState.getLegalActions()
    numOfAgents = gameState.getNumAgents()
    scores = [self.getValue(gameState.generateSuccessor(self.index, action), self.depth, (self.index+1)%numOfAgents) for action in actions]
    bestScore = max(scores)
    # print(bestScore)
    bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
    chosenIndex = random.choice(bestIndices) # Pick randomly among the best
    return actions[chosenIndex]

  
  def getQ(self, gameState, action):
    """
      Returns the minimax Q-Value from the current gameState and given action
      using self.depth and self.evaluationFunction.
      Terminal states can be found by one of the following: 
      pacman won, pacman lost or there are no legal moves.
    """
    # BEGIN_YOUR_ANSWER
    numOfAgents = gameState.getNumAgents()
    return self.getValue(gameState.generateSuccessor(self.index, action), 
                        self.depth if self.index < numOfAgents-1 else self.depth-1, 
                        (self.index+1)%numOfAgents)
    # END_YOUR_ANSWER

######################################################################################
# Problem 2a: implementing expectimax

class ExpectimaxAgent(MultiAgentSearchAgent):
  """
    Your expectimax agent (problem 2)
  """
  # Minimax와 동일하지만, ghost가 uniformly random한 action을 취하는 것만 다름 -> random하게 취하므로 ghost의 value는 그 기댓값이 됨
  def getValue(self, gameState, depth, index):
    if depth == 0:
      return self.evaluationFunction(gameState)
    actions = gameState.getLegalActions(index)
    if actions is None or gameState.isWin() or gameState.isLose():
      return gameState.getScore()
    numOfAgents = gameState.getNumAgents()
    scores = [self.getValue(gameState.generateSuccessor(index, action), 
                        depth if index < numOfAgents-1 else depth-1, 
                        (index+1)%numOfAgents)
                        for action in actions]
    bestScore = max(scores) if index==0 else sum(scores)/len(scores) # Minimax와 유일하게 다른 부분
    return bestScore
  
  def getAction(self, gameState):
    """
      Returns the expectimax action using self.depth and self.evaluationFunction

      All ghosts should be modeled as choosing uniformly at random from their
      legal moves.
    """

    # BEGIN_YOUR_ANSWER
    
    actions = gameState.getLegalActions()
    numOfAgents = gameState.getNumAgents()
    scores = [self.getValue(gameState.generateSuccessor(self.index, action), self.depth, (self.index+1)%numOfAgents) for action in actions]
    bestScore = max(scores)
    # print(bestScore)
    bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
    chosenIndex = random.choice(bestIndices) # Pick randomly among the best
    return actions[chosenIndex]
    # END_YOUR_ANSWER
  
  def getQ(self, gameState, action):
    """
      Returns the expectimax Q-Value using self.depth and self.evaluationFunction.
    """
    # BEGIN_YOUR_ANSWER
    numOfAgents = gameState.getNumAgents()
    return self.getValue(gameState.generateSuccessor(self.index, action), 
                        self.depth if self.index < numOfAgents-1 else self.depth-1, 
                        (self.index+1)%numOfAgents)
    # END_YOUR_ANSWER

######################################################################################
# Problem 3a: implementing biased-expectimax

class BiasedExpectimaxAgent(MultiAgentSearchAgent):
  """
    Your biased-expectimax agent (problem 3)
  """
  # ExpectiMax와 동일하지만, ghost가 biased한 action을 취하는 것만 다름. STOP할 확률이 다른 action의 확률보다 50% 높음 
  def getValue(self, gameState, depth, index):
    if depth == 0:
      return self.evaluationFunction(gameState)
    actions = gameState.getLegalActions(index)
    if actions is None or gameState.isWin() or gameState.isLose():
      return gameState.getScore()
    numOfAgents = gameState.getNumAgents()
    scores = [(self.getValue(gameState.generateSuccessor(index, action), 
                        depth if index < numOfAgents-1 else depth-1, 
                        (index+1)%numOfAgents))
                        for action in actions]
    numOfScores = len(scores)
    # for score in scores:
    #   print(score)
    bestScore = max(scores) if index==0 else sum([scores[i]/(2*numOfScores) if actions[i]!="STOP"
                                                                        else scores[i]*(0.5+1/(2*numOfScores))
                                                                        for i in range(len(actions))]) # STOP에 biased됨을 반영
    return bestScore
  
  def getAction(self, gameState):
    """
      Returns the biased-expectimax action using self.depth and self.evaluationFunction

      All ghosts should be modeled as choosing stop-biasedly from their
      legal moves.
    """

    # BEGIN_YOUR_ANSWER
    actions = gameState.getLegalActions()
    numOfAgents = gameState.getNumAgents()
    scores = [self.getValue(gameState.generateSuccessor(self.index, action), self.depth, (self.index+1)%numOfAgents) for action in actions]
    bestScore = max(scores)
    # print(bestScore)
    bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
    chosenIndex = random.choice(bestIndices) # Pick randomly among the best
    return actions[chosenIndex]
    # END_YOUR_ANSWER
  
  def getQ(self, gameState, action):
    """
      Returns the biased-expectimax Q-Value using self.depth and self.evaluationFunction.
    """
    # BEGIN_YOUR_ANSWER
    numOfAgents = gameState.getNumAgents()
    return self.getValue(gameState.generateSuccessor(self.index, action), 
                        self.depth if self.index < numOfAgents-1 else self.depth-1, 
                        (self.index+1)%numOfAgents)
    # END_YOUR_ANSWER

######################################################################################
# Problem 4a: implementing expectiminimax

class ExpectiminimaxAgent(MultiAgentSearchAgent):
  """
    Your expectiminimax agent (problem 4)
  """
  # ghost의 index에 따라 min, random 선택 
  def getValue(self, gameState, depth, index):
    if depth == 0:
      return self.evaluationFunction(gameState)
    actions = gameState.getLegalActions(index)
    if actions is None or gameState.isWin() or gameState.isLose():
      return gameState.getScore()
    numOfAgents = gameState.getNumAgents()
    scores = [(self.getValue(gameState.generateSuccessor(index, action), 
                        depth if index < numOfAgents-1 else depth-1, 
                        (index+1)%numOfAgents))
                        for action in actions]
    
    bestScore = -1
    if index == 0:
      bestScore = max(scores)
    elif index%2 == 1:
      bestScore = min(scores)
    else:
      bestScore = sum(scores)/len(scores)

    return bestScore

  def getAction(self, gameState):
    """
      Returns the expectiminimax action using self.depth and self.evaluationFunction

      The even-numbered ghost should be modeled as choosing uniformly at random from their
      legal moves.
    """

    # BEGIN_YOUR_ANSWER    
    actions = gameState.getLegalActions()
    numOfAgents = gameState.getNumAgents()
    scores = [self.getValue(gameState.generateSuccessor(self.index, action), self.depth, (self.index+1)%numOfAgents) for action in actions]
    bestScore = max(scores)
    # print(bestScore)
    bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
    chosenIndex = random.choice(bestIndices) # Pick randomly among the best
    return actions[chosenIndex]
    # END_YOUR_ANSWER
  
  def getQ(self, gameState, action):
    """
      Returns the expectiminimax Q-Value using self.depth and self.evaluationFunction.
    """
    # BEGIN_YOUR_ANSWER
    numOfAgents = gameState.getNumAgents()
    return self.getValue(gameState.generateSuccessor(self.index, action), 
                        self.depth if self.index < numOfAgents-1 else self.depth-1, 
                        (self.index+1)%numOfAgents)
    # END_YOUR_ANSWER

######################################################################################
# Problem 5a: implementing alpha-beta

class AlphaBetaAgent(MultiAgentSearchAgent):
  """
    Your expectiminimax agent with alpha-beta pruning (problem 5)
  """
  def getValue(self, gameState, depth, index, alpha, beta):
    if depth == 0:
      return self.evaluationFunction(gameState)
    actions = gameState.getLegalActions(index)
    if actions is None or gameState.isWin() or gameState.isLose():
      return gameState.getScore()
      
    numOfAgents = gameState.getNumAgents()
    bestScore = float('-inf') if index==0 else float('inf')
    if index==0:
      for action in actions:
        bestScore = max(bestScore, self.getValue(gameState.generateSuccessor(index, action), 
                                                    depth if index < numOfAgents-1 else depth-1, 
                                                    (index+1)%numOfAgents,
                                                    alpha, beta))
        alpha = max(alpha, bestScore)
        if beta < alpha:
          break
    elif index%2 == 1:
      for action in actions:
        bestScore = min(bestScore, self.getValue(gameState.generateSuccessor(index, action), 
                                                    depth if index < numOfAgents-1 else depth-1, 
                                                    (index+1)%numOfAgents,
                                                    alpha, beta))
        beta = min(beta, bestScore)
        if beta < alpha:
          break
    else:
      scores = [(self.getValue(gameState.generateSuccessor(index, action), 
                    depth if index < numOfAgents-1 else depth-1, 
                    (index+1)%numOfAgents, 
                    alpha, beta))
                    for action in actions]
      bestScore = sum(scores)/len(scores)

    return bestScore

  def getAction(self, gameState):
    """
      Returns the expectiminimax action using self.depth and self.evaluationFunction
    """

    # BEGIN_YOUR_ANSWER
    # Alpha-Beta pruning 해야함
    actions = gameState.getLegalActions()
    numOfAgents = gameState.getNumAgents()
    scores = [self.getValue(gameState.generateSuccessor(self.index, action), self.depth, (self.index+1)%numOfAgents, float('-inf'), float('inf')) 
              for action in actions]
    bestScore = max(scores)
    # print(bestScore)
    bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
    chosenIndex = random.choice(bestIndices) # Pick randomly among the best
    return actions[chosenIndex]
    # END_YOUR_ANSWER
  
  def getQ(self, gameState, action):
    """
      Returns the expectiminimax Q-Value using self.depth and self.evaluationFunction.
    """
    # BEGIN_YOUR_ANSWER
    numOfAgents = gameState.getNumAgents()
    return self.getValue(gameState.generateSuccessor(self.index, action), 
                        self.depth if self.index < numOfAgents-1 else self.depth-1, 
                        (self.index+1)%numOfAgents, 
                        float('-inf'), float('inf'))
    # END_YOUR_ANSWER

######################################################################################
# Problem 6a: creating a better evaluation function

def betterEvaluationFunction(currentGameState):
  """
  Your extreme, unstoppable evaluation function (problem 6).
  """

  # BEGIN_YOUR_ANSWER
  raise NotImplementedError  # remove this line before writing code
  # END_YOUR_ANSWER

def choiceAgent():
  """
    Choose the pacman agent model you want for problem 6.
    You can choose among the agents above or design your own agent model.
    You should return the name of class of pacman agent.
    (e.g. 'MinimaxAgent', 'BiasedExpectimaxAgent', 'MyOwnAgent', ...)
  """
  # BEGIN_YOUR_ANSWER
  raise NotImplementedError  # remove this line before writing code
  # END_YOUR_ANSWER

# Abbreviation
better = betterEvaluationFunction
