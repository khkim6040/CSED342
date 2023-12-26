'''
Licensing Information: Please do not distribute or publish solutions to this
project. You are free to use and extend Driverless Car for educational
purposes. The Driverless Car project was developed at Stanford, primarily by
Chris Piech (piech@cs.stanford.edu). It was inspired by the Pacman projects.
'''
from engine.const import Const
import util, math, random, collections

# Class: ConditionalProb
class ConditionalProb:
    def __init__(self):
        self.t = 0
        self.condProb = {} # type is dictionary. it is referred to as Belief in the later problems.

    def setEnv(self, initial, transition, emission, states):
        self.initial = initial
        self.transition = transition
        self.emission = emission
        self.states = states

############################################################
# Problem 1: Simplification
# setBinaryEnv():
# - delta: [δ] is the parameter governing the distribution
#   of the initial car's position
# - epsilon: [ε] is the parameter governing the conditional distribution
#   of the next car's position given the previos car's position
# - eta: [η] is the parameter governing the conditional distribution
#   of the sensor's measurement given the current car's position
# - states: possible values of $c_t$ and $d_t$
#   This corresponds to coordinates in later problems.
# - c_curr, c_prev and d_curr: $c_t$, $c_{t-1}$ and $d_t$, respectively.
# 
# normalize(): normalize self.CondProb to suit the probability distribution.
#
# observe():
# - d: observation $d_t$ at time [self.t]  
# - Update a conditional probability [self.condProb] of $c_t$ 
#   given observation $d_0, d_1, ..., d_t-1$ and $d_t$
#
# observeSeq(): 
# - [d_list]: $d_s$, ..., $d_t$
# - Estimate a conditional probability [self.condProb] of $c_t$ 
#   given observation $d_0, d_1, ..., d_t$
#
# Notes:
# - initial, transition, emission, and normalize functions and
#   if statements in observe functions are just guides.
# - If you really want, you can remove the given functions.
############################################################

    def setBinaryEnv(self, delta, epsilon, eta):        
        states = set(range(2))
    # BEGIN_YOUR_ANSWER (our solution is 22 lines of code, but don't worry if you deviate from this)
        # 문제 주어진 조건으로 ㄱ
        # 가능한 초기 위치 확률 initial -> c1==0 이면 delta, c1==1이면 1-delta
        def initial(c1): 
            return delta if c1==0 else 1-delta
        # ct-1 에서 ct로 갈 떄 확률 분포
        def transition(c_curr, c_prev): 
            return epsilon if c_curr!=c_prev else 1-epsilon
        # sensor가 time t에 ct를 제대로 detect할 확률 
        def emission(d_curr, c_curr): 
            return eta if d_curr!=c_curr else 1-eta
        self.setEnv(initial, transition, emission, states)

    def normalize(self):
        prob_sum = sum(self.condProb.values())
        for k in self.condProb.keys():
            self.condProb[k] /= prob_sum

    def observe(self, d):
        # 구하고자 하는것: Dt까지 주어졌을 떄의 Ct의 확률 분포(여기서는 0, 1을 가질 확률)를 알아내는 것 => Conditional Prob. P(Ct=ct|D1:t)
        # P(Ct=ct|D1:t) = p(dt|ct)p(ct|d1:t−1) = p(dt|ct)SUM(p(ct|ct−1)p(ct−1|d1:t−1)) <- ct-1의 가능한 값들에 대한 sum.
        # 따라서 Time t에서 condProb는 결국 emission*sum(transition*condProb(t-1))로 구할 수 있다. 이것 또한 확률분포이므로 가능한 Ct=ct 에 대해 모두 구해줘야 함
        states = set(range(2))
        if self.t == 0:
            self.condProb = {state: self.emission(d, state)*self.initial(state) for state in states}
            
        else:
            self.condProb = {state: self.emission(d, state)*
                             sum([self.transition(state, prev_state)*self.condProb[prev_state] for prev_state in states]) 
                             for state in states}

        self.t= self.t + 1
        self.normalize()
        # print(self.condProb)
   # END_YOUR_ANSWER

    def observeSeq(self, d_list):
        for d in d_list:
            self.observe(d)

    def getCondProb(self): return self.condProb



# Class: ExactInference
# ---------------------
# Maintain and update a belief distribution over the probability of a car
# being in a tile using exact updates (correct, but slow times).
class ExactInference:
    
    # Function: Init
    # --------------
    # Constructer that initializes an ExactInference object which has
    # numRows x numCols number of tiles.
    def __init__(self, numRows, numCols):
        self.skipElapse = False ### ONLY USED BY GRADER.PY in case problem 3 has not been completed
        # util.Belief is a class (constructor) that represents the belief for a single
        # inference state of a single car (see util.py).
        self.belief = util.Belief(numRows, numCols)
        self.transProb = util.loadTransProb()
   
     
    ############################################################
    # Problem 2: 
    # Function: Observe (update the probablities based on an observation)
    # -----------------
    # Takes |self.belief| and updates it based on the distance observation
    # $d_t$ and your position $a_t$.
    #
    # - agentX: x location of your car (not the one you are tracking)
    # - agentY: y location of your car (not the one you are tracking)
    # - observedDist: true distance plus a mean-zero Gaussian with standard 
    #                 deviation Const.SONAR_STD
    # 
    # Notes:
    # - Convert row and col indices into locations using util.rowToY and util.colToX.
    # - util.pdf: computes the probability density function for a Gaussian
    # - Don't forget to normalize self.belief!
    ############################################################

    def observe(self, agentX, agentY, observedDist):
        # BEGIN_YOUR_ANSWER (our solution is 9 lines of code, but don't worry if you deviate from this)
        # 각 타일의 위치를 ct라고 가정했을 때 관측값을 보고 거기에 얼마만큼의 확률로 차가 있는지 확인.
        # 각 tile과 at의 거리를 mean으로 하는 Gaussian 분포에 관측값 dt를 넣었을 때 PDF = emission
        # 이를 기존 belief에 곱해줌으로써 update
        for row in range(self.belief.getNumRows()):
            for col in range(self.belief.getNumCols()):
                x = util.colToX(col)
                y = util.rowToY(row)
                ct = ((x-agentX)**2 + (y-agentY)**2)**0.5
                emissionProb = util.pdf(ct, Const.SONAR_STD, observedDist)
                self.belief.setProb(row, col, self.belief.getProb(row, col)*emissionProb)
        self.belief.normalize()
        # END_YOUR_ANSWER

    ############################################################
    # Problem 3: 
    # Function: Elapse Time (propose a new belief distribution based on a learned transition model)
    # ---------------------
    # Takes |self.belief| and updates it based on the passing of one time step.
    # Notes:
    # - Use the transition probabilities in self.transProb, which gives all
    #   ((oldTile, newTile), transProb) key-val pairs that you must consider.
    # - Other ((oldTile, newTile), transProb) pairs not in self.transProb have
    #   zero probabilities and do not need to be considered. 
    # - util.Belief is a class (constructor) that represents the belief for a single
    #   inference state of a single car (see util.py).
    # - Be sure to update beliefs in self.belief ONLY based on the current self.belief distribution. 
    #   Do NOT invoke any other updated belief values while modifying self.belief.
    # - Use addProb and getProb to manipulate beliefs to add/get probabilities from a belief (see util.py).
    # - Don't forget to normalize self.belief!
    ############################################################
    def elapseTime(self):
        if self.skipElapse: return ### ONLY FOR THE GRADER TO USE IN Problem 2
        # BEGIN_YOUR_ANSWER (our solution is 8 lines of code, but don't worry if you deviate from this)
        # transProb 안에 있는 모든 타일들에 대해 업데이트 해야함
        # newTile, oldTile의 위치가 겹칠 수 있으므로 self.belief로 하면 get, set 과정에서 이중 업데이트 될 수 있음(current distribution만으로 업데이트 안됨)
        # temp로 복사 후 temp를 업데이트 후 붙여넣기
        temp = util.Belief(self.belief.getNumRows(), self.belief.getNumCols(), 0.0)
        for (oldTile, newTile), transProb in self.transProb.items():
            oldRow, oldCol = oldTile
            newRow, newCol = newTile
            temp.addProb(newRow, newCol, self.belief.getProb(oldRow, oldCol)*transProb)
        
        self.belief = temp
        self.belief.normalize()
        # END_YOUR_ANSWER
      
    # Function: Get Belief
    # ---------------------
    # Returns your belief of the probability that the car is in each tile. Your
    # belief probabilities should sum to 1.    
    def getBelief(self):
        return self.belief

        
# Class: Particle Filter
# ----------------------
# Maintain and update a belief distribution over the probability of a car
# being in a tile using a set of particles.
class ParticleFilter:
    
    NUM_PARTICLES = 200
    
    # Function: Init
    # --------------
    # Constructer that initializes an ParticleFilter object which has
    # numRows x numCols number of tiles.
    def __init__(self, numRows, numCols):
        self.belief = util.Belief(numRows, numCols)

        # Load the transition probabilities and store them in a dict of defaultdict
        # self.transProbDict[oldTile][newTile] = probability of transitioning from oldTile to newTile
        self.transProb = util.loadTransProb()
        self.transProbDict = dict()
        for (oldTile, newTile) in self.transProb:
            if not oldTile in self.transProbDict:
                self.transProbDict[oldTile] = collections.defaultdict(int)
            self.transProbDict[oldTile][newTile] = self.transProb[(oldTile, newTile)]
            
        # Initialize the particles randomly
        self.particles = collections.defaultdict(int)
        potentialParticles = list(self.transProbDict.keys())
        for i in range(self.NUM_PARTICLES):
            particleIndex = int(random.random() * len(potentialParticles))
            self.particles[potentialParticles[particleIndex]] += 1
            
        self.updateBelief()

    # Function: Update Belief
    # ---------------------
    # Updates |self.belief| with the probability that the car is in each tile
    # based on |self.particles|, which is a defaultdict from particle to
    # probability (which should sum to 1).
    def updateBelief(self):
        newBelief = util.Belief(self.belief.getNumRows(), self.belief.getNumCols(), 0)
        for tile in self.particles:
            newBelief.setProb(tile[0], tile[1], self.particles[tile])
        newBelief.normalize()
        self.belief = newBelief
    
    ############################################################
    # Problem 4 (part a): 
    # Function: Observe:
    # -----------------
    # Takes |self.particles| and updates them based on the distance observation
    # $d_t$ and your position $a_t$. 
    # This algorithm takes two steps:
    # 1. Reweight the particles based on the observation.
    #    Concept: We had an old distribution of particles, we want to update these
    #             these particle distributions with the given observed distance by
    #             the emission probability. 
    #             Think of the particle distribution as the unnormalized posterior 
    #             probability where many tiles would have 0 probability.
    #             Tiles with 0 probabilities (no particles), we do not need to update. 
    #             This makes particle filtering runtime to be O(|particles|).
    #             In comparison, exact inference (problem 2 + 3), most tiles would
    #             would have non-zero probabilities (though can be very small). 
    # 2. Resample the particles.
    #    Concept: Now we have the reweighted (unnormalized) distribution, we can now 
    #             resample the particles and update where each particle should be at.
    #
    # - agentX: x location of your car (not the one you are tracking)
    # - agentY: y location of your car (not the one you are tracking)
    # - observedDist: true distance plus a mean-zero Gaussian with standard deviation Const.SONAR_STD
    #
    # Notes:
    # - Create |self.NUM_PARTICLES| new particles during resampling.
    # - To pass the grader, you must call util.weightedRandomChoice() once per new particle.
    ############################################################
    def observe(self, agentX, agentY, observedDist):
        # BEGIN_YOUR_ANSWER (our solution is 12 lines of code, but don't worry if you deviate from this)
        # 1. Reweighting
        weights = {}
        for tile in self.particles:
            y = util.rowToY(tile[0])
            x = util.colToX(tile[1])
            ct = ((x-agentX)**2 + (y-agentY)**2)**0.5
            emissionProb = util.pdf(ct, Const.SONAR_STD, observedDist)
            weights[tile] = self.particles[tile]*emissionProb
        # 2. Resampling
        temp = collections.defaultdict(int)
        for _ in range(self.NUM_PARTICLES):
            tile = util.weightedRandomChoice(weights)
            temp[tile] += 1
        
        self.particles = temp
        # END_YOUR_ANSWER
        self.updateBelief()
    
    ############################################################
    # Problem 4 (part b): 
    # Function: Elapse Time (propose a new belief distribution based on a learned transition model)
    # ---------------------
    # Read |self.particles| (defaultdict) corresonding to time $t$ and writes
    # |self.particles| corresponding to time $t+1$.
    # This algorithm takes one step
    # 1. Proposal based on the particle distribution at current time $t$:
    #    Concept: We have particle distribution at current time $t$, we want to
    #             propose the particle distribution at time $t+1$. We would like
    #             to sample again to see where each particle would end up using
    #             the transition model.
    #
    # Notes:
    # - transition probabilities is now using |self.transProbDict|
    # - Use util.weightedRandomChoice() to sample a new particle.
    # - To pass the grader, you must loop over the particles using
    #       for tile in self.particles
    #   and call util.weightedRandomChoice() $once per particle$ on the tile.
    ############################################################
    def elapseTime(self):
        # BEGIN_YOUR_ANSWER (our solution is 7 lines of code, but don't worry if you deviate from this)
        # t+1의 particle
        temp = collections.defaultdict(int)
        # {타일: 횟수}로 되어있는 particles에서 모든 타일 뽑고 그 횟수만큼 transProb을 통해 유력한 다음 타일 뽑음 -> 새로운 particle proposed
        for oldTile, val in self.particles.items():
            for _ in range(val):
                # 1. Proposal
                newTile = util.weightedRandomChoice(self.transProbDict[oldTile])
                temp[newTile] += 1
        # t+1 particles
        self.particles = temp
        # normalize
        self.updateBelief()
        # END_YOUR_ANSWER
        
    # Function: Get Belief
    # ---------------------
    # Returns your belief of the probability that the car is in each tile. Your
    # belief probabilities should sum to 1.    
    def getBelief(self):
        return self.belief
