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
import random, util , sys

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
        oldFood = currentGameState.getFood().asList()
        newFood = successorGameState.getFood().asList()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        "*** YOUR CODE HERE ***"
        
        max = 0
        min = 10000
        for i in oldFood:
            if manhattanDistance(i,newPos) < min:
                min = manhattanDistance(i,newPos)
                min_node = i
        """
        for i in oldFood:
            if not(i==min_node): 
                if manhattanDistance(i,min_node) > max:
                    max = manhattanDistance(i,min_node)
        min += max
        """
        min_ghost = 10000
        for i in newGhostStates:
                if manhattanDistance(i.getPosition(),newPos) < min_ghost:
                    min_ghost = manhattanDistance(i.getPosition(),newPos)
        """if newScaredTimes[0] == 0:
            if min_ghost<=3:
                if successorGameState.getScore()>0:
                    return -(successorGameState.getScore())*(min_ghost+min+10000)
                else:
                    return (successorGameState.getScore())*(min_ghost+min+10000)"""
        if not(newScaredTimes[0] == 0) and not(newScaredTimes[0] == 1):
            if not(min==0) :
                return 1/min
            else:
                return 1
        else:
            if min_ghost < 3:
                return (min_ghost*4) - (min*2) - (len(newFood)*3)
            else:
                return (min_ghost) - (min*15) - (len(newFood)*15)
        
            
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
        self.depth_reached = 0


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
        return self.minimax(gameState, self.index, 0)[1]
    
    def minimax(self, gameState, index, depth):
        num_agents = gameState.getNumAgents()

        if depth == self.depth and index % num_agents == 0:
            return self.evaluationFunction(gameState), None

        if index % num_agents == 0:
            return self.max_value(gameState, index % num_agents, depth)

        return self.min_value(gameState, index % num_agents, depth)

    def min_value(self, gameState, index, depth):
        successor_states = [(gameState.generateSuccessor(index, action), action) for action in gameState.getLegalActions(index)]

        if len(successor_states) == 0:
            return self.evaluationFunction(gameState), None

        x = index + 1
        State_values = []
        for successor, action in successor_states:
            next_val_action = self.minimax(successor , x, depth)
            State_values.append((next_val_action[0],action))
        value , action_to_successor = min(State_values)        

        return value, action_to_successor

    def max_value(self, gameState, index, depth):
        successor_states = [(gameState.generateSuccessor(index, action), action) for action in gameState.getLegalActions(index)]

        if len(successor_states) == 0:
            return self.evaluationFunction(gameState), None
        x = index + 1
        State_values = []
        for successor, action in successor_states:
            next_val_action = self.minimax(successor, x, depth+1)
            State_values.append((next_val_action[0],action))
        value , action_to_successor = max(State_values)        

        return value, action_to_successor


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        return self.alpha_beta_minimax(gameState, self.index, 0 , -1000000 , 1000000)[1]
    
    def alpha_beta_minimax(self, gameState, index, depth , alpha , beta):
        num_agents = gameState.getNumAgents()

        if depth == self.depth and index % num_agents == 0:
            return self.evaluationFunction(gameState), None

        if index % num_agents == 0:
            return self.alpha_beta_max(gameState, index % num_agents, depth , alpha ,beta)

        return self.alpha_beta_min(gameState, index % num_agents, depth , alpha ,beta)


    def alpha_beta_min(self, gamestate, index, depth, alpha, beta):
        legal_actions = gamestate.getLegalActions(index)
        if len(legal_actions) == 0:
            return self.evaluationFunction(gamestate), None
        min = 100000
        new_action = None
        for action in legal_actions:
            successor_game_state = gamestate.generateSuccessor(index, action)
            new_val_action = self.alpha_beta_minimax(successor_game_state, index+1, depth , alpha, beta)
            """
            if next_val_action[0] <= beta:
                return next_val_action[0],action
            State_values.append((next_val_action[0],action))
            value , action_to_successor = min(State_values)
            if value > alpha:
            """
            if new_val_action[0] < min:
                min = new_val_action[0]
                action_to_successor = action
            if min < alpha:
                return min, action_to_successor
            if beta > min :
                beta = min
        return min, action_to_successor
    

    def alpha_beta_max(self, gamestate, index, depth, alpha, beta):
        legal_actions = gamestate.getLegalActions(index)
        if len(legal_actions) == 0:
            return self.evaluationFunction(gamestate), None
        max = -100000
        new_action = None
        for action in legal_actions:
            successor_game_state = gamestate.generateSuccessor(index, action)
            new_val_action = self.alpha_beta_minimax(successor_game_state, index+1, depth+1 , alpha, beta)
            """
            if next_val_action[0] >= alpha:
                return next_val_action[0],action
            State_values.append((next_val_action[0],action))
            value , action_to_successor = max(State_values)        
            if value < beta:
            """
            if new_val_action[0] > max:
                max = new_val_action[0]
                action_to_successor = action
            if max > beta:
                return max, action_to_successor
            if alpha < max :
                alpha = max
        return max, action_to_successor


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        return self.expecti_minimax(gameState, self.index, 0 , -1000000 , 1000000)[1]
    
    def expecti_minimax(self, gameState, index, depth , alpha , beta):
        num_agents = gameState.getNumAgents()

        if depth == self.depth and index % num_agents == 0:
            return self.evaluationFunction(gameState), None

        if index % num_agents == 0:
            return self.expecti_max(gameState, index % num_agents, depth , alpha ,beta)

        return self.expecti_mean(gameState, index % num_agents, depth , alpha ,beta)


    def expecti_mean(self, gamestate, index, depth, alpha, beta):
        legal_actions = gamestate.getLegalActions(index)
        if len(legal_actions) == 0:
            return self.evaluationFunction(gamestate), None
        sum = 0
        new_action = None
        for action in legal_actions:
            successor_game_state = gamestate.generateSuccessor(index, action)
            new_val_action = self.expecti_minimax(successor_game_state, index+1, depth , alpha, beta)
            sum += new_val_action[0]
        
                
        return sum/len(legal_actions),None
    

    def expecti_max(self, gamestate, index, depth, alpha, beta):
        legal_actions = gamestate.getLegalActions(index)
        if len(legal_actions) == 0:
            return self.evaluationFunction(gamestate), None
        max = -100000
        new_action = None
        for action in legal_actions:
            successor_game_state = gamestate.generateSuccessor(index, action)
            new_val_action = self.expecti_minimax(successor_game_state, index+1, depth+1 , alpha, beta)
            if new_val_action[0] > max:
                max = new_val_action[0]
                action_to_successor = action
            if max > beta:
                return max, action_to_successor
            if alpha < max :
                alpha = max
        return max, action_to_successor

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: Features used are distance to closest food, distance to closest ghosts, Ghosts Scared Time, Score of the state , number of food remaining.
    The weights are added according to optimise the agent.
    The closer the food the lesser the value to make the value of the state more it has to be reciprocated , same is done to distance to ghosts.
    Adding Ghosts Scared Time will make the agent prefer food rather than thinking of ghosts while the ghosts scare time is running.
    Score of the state lets us know that weather the present state has a food which has been eaten or it.
    Number of food remaining will help to prefer the states which has food in it. 
    By using this all features and providing them with necessary weights(i.e., for example when ghost is near to pacman ,pacman has to prefer surviving rather than food so min_ghost is given higher weight than min_food) this evaluation function is made
    """
    "*** YOUR CODE HERE ***"
    newPos = currentGameState.getPacmanPosition()
    oldFood = currentGameState.getFood().asList()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    min = 10000
    max = 0
    for i in oldFood:
        if manhattanDistance(i,newPos) < min:
            min = manhattanDistance(i,newPos)
            min_node = i
        if manhattanDistance(i,newPos) > max:
            max = manhattanDistance(i,newPos)
            max_node = i
    min_ghost = 10000
    for i in newGhostStates:
        if manhattanDistance(i.getPosition(),newPos) < min_ghost:
                min_ghost = manhattanDistance(i.getPosition(),newPos)
    if min_ghost == 0 or min ==0 or max ==0:
        if min_ghost<= 2:
            return (11/((min_ghost+1)*3)) + (2/(min+1)) - len(oldFood*7) + (newScaredTimes[0]*0.6) + currentGameState.getScore()

        return (2/(min_ghost+1)) + (5/(min+1)) - len(oldFood*7) + (newScaredTimes[0]*2) + currentGameState.getScore()
   
    if min_ghost<= 2:
        return (11/(min_ghost*3)) + (2/(min)) - len(oldFood*7) + (newScaredTimes[0]*0.6) + currentGameState.getScore()

    return (2/(min_ghost)) + (5/(min)) - len(oldFood*7) + (newScaredTimes[0]*2) + currentGameState.getScore()
    
# Abbreviation
better = betterEvaluationFunction
