import util, math, random
from collections import defaultdict
from util import ValueIteration


############################################################
# Problem 2a: BlackjackMDP


class BlackjackMDP(util.MDP):
    def __init__(self, cardValues, multiplicity, threshold, peekCost):
        """
        cardValues: array of card values for each card type
        multiplicity: number of each card type
        threshold: maximum total before going bust
        peekCost: how much it costs to peek at the next card
        """
        super().__init__()

        self.cardValues = cardValues
        self.multiplicity = multiplicity
        self.threshold = threshold
        self.peekCost = peekCost

    # Return the start state.
    # Look at this function to learn about the state representation.
    # The first element of the tuple is the sum of the cards in the player's
    # hand.
    # The second element is the index (not the value) of the next card, if the player peeked in the
    # last action.  If they didn't peek, this will be None.
    # The final element is the current deck.
    def startState(self):
        return (0, None, (self.multiplicity,) * len(self.cardValues))  # total, next card (if any), multiplicity for each card

    # Return set of actions possible from |state|.
    # You do not need to modify this function.
    # All logic for dealing with end states should be done in succAndProbReward
    def actions(self, state):
        return ['Take', 'Peek', 'Quit']

    # Return a list of (newState, prob, reward) tuples corresponding to edges
    # coming out of |state|.  Indicate a terminal state (after quitting or
    # busting) by setting the deck to None. 
    # When the probability is 0 for a particular transition, don't include that 
    # in the list returned by succAndProbReward.
    def succAndProbReward(self, state, action):
        # action 무엇인지 확인하고 그에 따라 결과 리스트에 다음 상태 넣은 후 결과 리스트 리턴
        # 먼저 카드 패 다 떨어졌는지 확인 후 처리
        # Take 
            # 만약 이전에 Peek했다면 Peek 한 index만 넣음, Peek 안했다면 모든 남은 카드 동일한 확률로 넣음
            # Peek -> 한 장만 선택. 
            # Peek 아니라면 -> 남은 모든 종류 카트 동일한 확률로 선택 
            # takeCard(cardIndex) 에서 cardIndex의 카드를 뽑음. 뽑은 카드 value에 따라 burst, 정상적 진행 나뉘고 다음 상태 생성 후 result에 append함. 
        # Peek 
            # nextCardIndexIfPeeked 가 None인 경우에만 진행. 아니라면 빈 리스트 리턴.
            # 현재 덱의 모든 종류 카드 동일한 확률로 result에 append. -index만큼 reward에 추가
        # Quit 
            # 현재 갖고있는 카드만큼 reward로. state는 (갖고있는 카드, None, None)
        
        
        totalCardValueInHand, nextCardIndexIfPeeked, deckCardCounts = state
        result = []
        
        def takeCard(cardIndex, isPeeked):
            amountOfCards = sum(deckCardCounts)
            prob = 1 if isPeeked else deckCardCounts[cardIndex]/amountOfCards
            card = self.cardValues[cardIndex]
            totalValue = totalCardValueInHand+card

            # burst 상황
            if totalValue > self.threshold:
                result.append(((totalValue, None, None), prob, 0))
            else:
                newDeck = list(deckCardCounts)
                newDeck[cardIndex] -= 1
                # 카드 다 떨어짐
                if(amountOfCards == 1):
                    result.append(((totalValue, None, None), prob, totalValue))  
                # 카드 남아있음 
                else:
                    result.append(((totalValue, None, tuple(newDeck)), prob, 0))


        if deckCardCounts is None:
            return []

        if action == 'Take':
            if nextCardIndexIfPeeked is not None:
                takeCard(nextCardIndexIfPeeked, True)
            else:
                for i in range(len(deckCardCounts)):
                    # 해당 index에 카드 하나도 없다면(=0) 스킵
                    if deckCardCounts[i] == 0:
                        continue
                    takeCard(i, False)

        elif action == 'Peek':
            if nextCardIndexIfPeeked is not None:
                return []
            else:
                for i in range(len(deckCardCounts)):
                    if deckCardCounts[i] == 0:
                        continue
                    prob = deckCardCounts[i]/sum(deckCardCounts)
                    result.append(((totalCardValueInHand, i, deckCardCounts), prob, -self.peekCost))

        elif action == 'Quit':
            result.append(((0, None, None), 1, totalCardValueInHand))
        
        return result

    def discount(self):
        return 1


############################################################
# Problem 3a: Q learning

# Performs Q-learning.  Read util.RLAlgorithm for more information.
# actions: a function that takes a state and returns a list of actions.
# discount: a number between 0 and 1, which determines the discount factor
# featureExtractor: a function that takes a state and action and returns a list of (feature name, feature value) pairs.
# explorationProb: the epsilon value indicating how frequently the policy
# returns a random action
class Qlearning(util.RLAlgorithm):
    def __init__(self, actions, discount, featureExtractor, explorationProb=0.2):
        self.actions = actions
        self.discount = discount
        self.featureExtractor = featureExtractor
        self.explorationProb = explorationProb
        self.weights = defaultdict(float)
        self.numIters = 0

    # Return the Q function associated with the weights and features
    def getQ(self, state, action):
        score = 0
        for f, v in self.featureExtractor(state, action):
            score += self.weights[f] * v
        return score

    # This algorithm will produce an action given a state.
    # Here we use the epsilon-greedy algorithm: with probability
    # |explorationProb|, take a random action.
    def getAction(self, state):
        self.numIters += 1
        if random.random() < self.explorationProb:
            return random.choice(self.actions(state))
        else:
            return max((self.getQ(state, action), action) for action in self.actions(state))[1]

    # Call this function to get the step size to update the weights.
    def getStepSize(self):
        return 1.0 / math.sqrt(self.numIters)

    # We will call this function with episode=[..., state, action,
    # reward, newState], which you should use to update
    # |self.weights|. You should update |self.weights| using
    # self.getStepSize(); use self.getQ() to compute the current
    # estimate of the parameters. Also, you should assume that
    # V_opt(newState)=0 when isLast(newState) is True
    def incorporateFeedback(self, episode, isLast):
        state, action, reward, newState = episode[-4:]

        if isLast(state):
            return

        def decrement(weight, other):
            result = weight
            for k, v in other:
                result[k] -= v
            return result
        
        def multiplyConstant(feature, c):
            result = []
            for k, v in feature:
                result.append((k, v*c))
            return result
            
        # w = w - eta(Q_opt(s, a) - (r + discount*V_opt(s')))*Ø(s, a)
        Q_opt = self.getQ(state, action)
        V_opt = max(self.getQ(newState, newAction) for newAction in self.actions(newState)) if not isLast(newState) else 0
        self.weights = decrement(self.weights, multiplyConstant(self.featureExtractor(state, action), self.getStepSize()*(Q_opt - (reward + self.discount*V_opt))))



############################################################
# Problem 3b: Q SARSA

class SARSA(Qlearning):
    # We will call this function with episode=[..., state, action,
    # reward, newState, newAction, newReward, newNewState], which you
    # should use to update |self.weights|. You should
    # update |self.weights| using self.getStepSize(); use self.getQ()
    # to compute the current estimate of the parameters. Also, you
    # should assume that Q_pi(newState, newAction)=0 when when
    # isLast(newState) is True
    def incorporateFeedback(self, episode, isLast):
        assert (len(episode) - 1) % 3 == 0
        if len(episode) >= 7:
            state, action, reward, newState, newAction = episode[-7: -2]
        else:
            return
        
        def decrement(weight, other):
            result = weight
            for k, v in other:
                result[k] -= v
            return result
        
        def multiplyConstant(feature, c):
            result = []
            for k, v in feature:
                result.append((k, v*c))
            return result
            
        Q_pi = self.getQ(state, action)
        Q_pi_further = self.getQ(newState, newAction) if not isLast(newState) else 0
        self.weights = decrement(self.weights, multiplyConstant(self.featureExtractor(state, action), self.getStepSize()*(Q_pi - (reward + self.discount*Q_pi_further))))

# Return a singleton list containing indicator feature (if exist featurevalue = 1)
# for the (state, action) pair.
def identityFeatureExtractor(state, action):
    featureKey = (state, action)
    featureValue = 1
    return [(featureKey, featureValue)]

############################################################
# Problem 3c: features for Q-learning.

# You should return a list of (feature key, feature value) pairs
# (see identityFeatureExtractor() above for an example).
# Implement the following features:
# - indicator on the total and the action (1 feature).
# - indicator on the presence/absence of each card type and the action (1 feature).
#       Example: if the deck is (3, 4, 0, 2), then your indicator on the presence of each card type is (1,1,0,1)
#       Only add this feature if the deck != None
# - indicator on the number of cards for each card type and the action (len(counts) features).
#       Example: if the deck is (3, 4, 0, 2), you should have four features (one for each card type).
#       And the first feature key will be (0, 3, action)
#       Only add these features if the deck != None

def blackjackFeatureExtractor(state, action):
    total, nextCard, counts = state
    features = []

    feature_1 = (total, action)
    features.append(feature_1)

    if counts is not None:
        feature_2 = list(1 if x!=0 else 0 for x in counts)
        features.append((tuple(feature_2), action))

        for i in range(len(counts)):
            feature_3 = (i, counts[i], action)
            features.append(feature_3)

    return list((feature, 1) for feature in features)
