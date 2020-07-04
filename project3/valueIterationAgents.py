# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

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
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        
        Policy_States = self.mdp.getStates()
        Vk1= util.Counter()
        for i in range(0,self.iterations):
            for j in Policy_States:
                if self.mdp.isTerminal(j):
                    continue
                Vk1[j] = self.getQValue(j,self.getAction(j))
            for x in Policy_States:
                self.values[x] = Vk1[x]
                    
                
                
    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        newState_prob = self.mdp.getTransitionStatesAndProbs(state, action)
        sum = 0
        for i in newState_prob:
            sum += i[1]*(self.mdp.getReward(state, action, i[0])+(self.discount*self.getValue(i[0])))
        return sum            
        
        util.raiseNotDefined()

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        
        State_actions = self.mdp.getPossibleActions(state)
        max_Action=util.Counter()
        for k in State_actions:
            max_Action[k] = self.getQValue(state,k)
        return max_Action.argMax()
        
        util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        Policy_States = self.mdp.getStates()
        Vk1= util.Counter()
        for i in range(0,self.iterations):
            if not Policy_States:
                Policy_States = self.mdp.getStates()
            j = Policy_States[0]
            #print(Policy_States)
            if self.mdp.isTerminal(j):
                Policy_States.pop(0)
                continue
            self.values[j] = self.getQValue(j,self.getAction(j))
            Policy_States.pop(0)

        


class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        Policy_States = self.mdp.getStates()
        Predecessors = util.Counter()
        VK1 = util.Counter()
        
        for i in Policy_States:
            for j in self.mdp.getPossibleActions(i):
                for k in self.mdp.getTransitionStatesAndProbs(i,j):
                    if not Predecessors[k[0]]:
                        Predecessors[k[0]] = set()
                    Predecessors[k[0]].add(i)
        print(Predecessors)

        Priority_Queue = util.PriorityQueue()            
        for j in Policy_States:
            if self.mdp.isTerminal(j):
                #print("1    ",j)
                continue
            VK1[j] = self.getQValue(j,self.getAction(j))
            Priority_Queue.push(j,-abs(VK1[j]-self.values[j]))
        for i in range(0,self.iterations):
            if Priority_Queue.isEmpty():
                break
            
            Updation_state = Priority_Queue.pop()
           
            self.values[Updation_state]=self.getQValue(Updation_state,self.getAction(Updation_state))
            for j in Predecessors[Updation_state]:
                if self.mdp.isTerminal(j):
                    continue
                #print(j,self.getAction(j))
                VK1[j] = self.getQValue(j,self.getAction(j))
                diff = abs(VK1[j]- self.values[j])
                if diff > self.theta:
                    Priority_Queue.update(j,-diff)
                
        

