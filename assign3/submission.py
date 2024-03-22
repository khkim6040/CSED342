import shell
import util
import wordsegUtil



############################################################
# Problem 1: Word Segmentation

# Problem 1a: Solve the word segmentation problem under a unigram model

class WordSegmentationProblem(util.SearchProblem):
    def __init__(self, query, unigramCost):
        self.query = query
        self.unigramCost = unigramCost

    def startState(self):
        # BEGIN_YOUR_ANSWER (our solution is 1 lines of code, but don't worry if you deviate from this)
        return self.query
        # END_YOUR_ANSWER

    def isEnd(self, state):
        # BEGIN_YOUR_ANSWER (our solution is 1 lines of code, but don't worry if you deviate from this)
        return len(state) == 0
        # END_YOUR_ANSWER

    def succAndCost(self, state):
        # BEGIN_YOUR_ANSWER (our solution is 5 lines of code, but don't worry if you deviate from this)
        # split one by one from starting index
        result = []
        for i in range(1, len(state)+1):
            action = state[:i]
            new_state = state[i:]
            cost = self.unigramCost(action)
            result.append((action, new_state, cost))
        return result
        # END_YOUR_ANSWER

def segmentWords(query, unigramCost):
    if len(query) == 0:
        return ''

    ucs = util.UniformCostSearch()
    ucs.solve(WordSegmentationProblem(query, unigramCost))

    # BEGIN_YOUR_ANSWER (our solution is 1 lines of code, but don't worry if you deviate from this)
    return ' '.join(ucs.actions)
    # END_YOUR_ANSWER

# Problem 1b: Solve the k-word segmentation problem under a unigram model

class KWordSegmentationProblem(util.SearchProblem):
    def __init__(self, k, query, unigramCost):
        self.k = k
        self.query = query
        self.unigramCost = unigramCost

    def startState(self):
        # BEGIN_YOUR_ANSWER (our solution is 1 lines of code, but don't worry if you deviate from this)
        # (받은 string, split 횟수)
        return (self.query, 0)
        # END_YOUR_ANSWER

    def isEnd(self, state):
        # BEGIN_YOUR_ANSWER (our solution is 1 lines of code, but don't worry if you deviate from this)
        return len(state[0]) == 0 and state[1] == self.k
        # END_YOUR_ANSWER

    def succAndCost(self, state):
        # BEGIN_YOUR_ANSWER (our solution is 7 lines of code, but don't worry if you deviate from this)
        curStr, splitCount = state
        result = []
        if splitCount == self.k-1:
            # If only one split is left, split the remaining string and calculate the cost
            action = curStr
            newState = ("", splitCount+1)
            cost = self.unigramCost(action)
            result.append((action, newState, cost))
            return result
        
        for i in range(1, len(curStr)+1):
            # If more than one split is left, split the string at the current position and continue with the rest of the string
            action = curStr[:i]
            cost = self.unigramCost(action)
            newState = (curStr[i:], splitCount+1)
            result.append((action, newState, cost))
            
        return result
        # END_YOUR_ANSWER

def segmentKWords(k, query, unigramCost):
    if len(query) == 0:
        return ''

    # BEGIN_YOUR_ANSWER (our solution is 3 lines of code, but don't worry if you deviate from this)
    ucs = util.UniformCostSearch()
    ucs.solve(KWordSegmentationProblem(k, query, unigramCost))
    return ' '.join(ucs.actions)
    # END_YOUR_ANSWER

############################################################
# Problem 2: Vowel Insertion

# Problem 2a: Solve the vowel insertion problem under a bigram cost

class VowelInsertionProblem(util.SearchProblem):
    def __init__(self, queryWords, bigramCost, possibleFills):
        self.queryWords = queryWords
        self.bigramCost = bigramCost
        self.possibleFills = possibleFills

    def startState(self):
        # BEGIN_YOUR_ANSWER (our solution is 1 lines of code, but don't worry if you deviate from this)
        # bigram을 사용하려면 이전 단어 필요함. state = (다음 index, 이전 단어)로 구성
        return (0, wordsegUtil.SENTENCE_BEGIN)
        # END_YOUR_ANSWER

    def isEnd(self, state):
        # BEGIN_YOUR_ANSWER (our solution is 1 lines of code, but don't worry if you deviate from this)
        return state[0] == len(self.queryWords)
        # END_YOUR_ANSWER

    def succAndCost(self, state):
        # BEGIN_YOUR_ANSWER (our solution is 9 lines of code, but don't worry if you deviate from this)
        curIndex, preWord = state
        curWord = self.queryWords[curIndex]
        possibleWords = self.possibleFills(curWord)
        result = []
        # if curWord is not in possibleFills(), curWord will be put in possibleWords itself
        if not possibleWords:
            possibleWords = {curWord}
        for word in possibleWords:
            cost = self.bigramCost(preWord, word)
            result.append((word, (curIndex+1, word), cost))
        return result
        # END_YOUR_ANSWER

def insertVowels(queryWords, bigramCost, possibleFills):
    # BEGIN_YOUR_ANSWER (our solution is 3 lines of code, but don't worry if you deviate from this)
    ucs = util.UniformCostSearch()
    ucs.solve(VowelInsertionProblem(queryWords, bigramCost, possibleFills))
    return ' '.join(ucs.actions)
    # END_YOUR_ANSWER

# Problem 2b: Solve the limited vowel insertion problem under a bigram cost

class LimitedVowelInsertionProblem(util.SearchProblem):
    def __init__(self, impossibleVowels, queryWords, bigramCost, possibleFills):
        self.impossibleVowels = impossibleVowels
        self.queryWords = queryWords
        self.bigramCost = bigramCost
        self.possibleFills = possibleFills

    def startState(self):
        # BEGIN_YOUR_ANSWER (our solution is 1 lines of code, but don't worry if you deviate from this)
        return (0, wordsegUtil.SENTENCE_BEGIN)
        # END_YOUR_ANSWER

    def isEnd(self, state):
        # BEGIN_YOUR_ANSWER (our solution is 1 lines of code, but don't worry if you deviate from this)
        return state[0] == len(self.queryWords)
        # END_YOUR_ANSWER

    def succAndCost(self, state):
        # BEGIN_YOUR_ANSWER (our solution is 10 lines of code, but don't worry if you deviate from this)
        curIndex, preWord = state
        curWord = self.queryWords[curIndex]
        possibleWords = self.possibleFills(curWord)

        for vowel in self.impossibleVowels:
            possibleWords = set.intersection(possibleWords, {word for word in possibleWords if vowel not in word})
        if not possibleWords:
            possibleWords = {curWord}
            
        result = []
        for word in possibleWords:
            cost = self.bigramCost(preWord, word)
            result.append((word, (curIndex+1, word), cost))
        return result
        # END_YOUR_ANSWER

def insertLimitedVowels(impossibleVowels, queryWords, bigramCost, possibleFills):
    # BEGIN_YOUR_ANSWER (our solution is 3 lines of code, but don't worry if you deviate from this)
    ucs = util.UniformCostSearch()
    ucs.solve(LimitedVowelInsertionProblem(impossibleVowels, queryWords, bigramCost, possibleFills))
    return ' '.join(ucs.actions)
    # END_YOUR_ANSWER

############################################################
# Problem 3: Putting It Together

# Problem 3a: Solve the joint segmentation-and-insertion problem

class JointSegmentationInsertionProblem(util.SearchProblem):
    def __init__(self, query, bigramCost, possibleFills):
        self.query = query
        self.bigramCost = bigramCost
        self.possibleFills = possibleFills

    def startState(self):
        # BEGIN_YOUR_ANSWER (our solution is 1 lines of code, but don't worry if you deviate from this)
        return (0, wordsegUtil.SENTENCE_BEGIN)
        # END_YOUR_ANSWER

    def isEnd(self, state):
        # BEGIN_YOUR_ANSWER (our solution is 1 lines of code, but don't worry if you deviate from this)
        return state[0] == len(self.query)
        # END_YOUR_ANSWER

    def succAndCost(self, state):
        # BEGIN_YOUR_ANSWER (our solution is 8 lines of code, but don't worry if you deviate from this)
        # query의 index 하나씩 늘려가면서 가능한 단어들 모두 cost구해서 result에 추가
        curIndex, preWord = state
        result = []
        for indexTo in range(curIndex+1, len(self.query)+1):
            curWord = self.query[curIndex:indexTo]
            possibleWords = self.possibleFills(curWord)
            for word in possibleWords:
                cost = self.bigramCost(preWord, word)
                result.append((word, (indexTo, word), cost))
        
        return result
        # END_YOUR_ANSWER

def segmentAndInsert(query, bigramCost, possibleFills):
    if len(query) == 0:
        return ''

    # BEGIN_YOUR_ANSWER (our solution is 3 lines of code, but don't worry if you deviate from this)
    ucs = util.UniformCostSearch()
    ucs.solve(JointSegmentationInsertionProblem(query, bigramCost, possibleFills))
    return ' '.join(ucs.actions)
    # END_YOUR_ANSWER

############################################################
# Problem 4: A* search

# Problem 4a: Define an admissible but not consistent heuristic function

class SimpleProblem(util.SearchProblem):
    def __init__(self):
        # BEGIN_YOUR_ANSWER (our solution is 4 lines of code, but don't worry if you deviate from this)
        self.start = 0
        self.end = 4
        self.path = [[(1, 2), (2, 1)], [(3, 3)], [(3, 1)], [(4, 100)]]
        # END_YOUR_ANSWER

    def startState(self):
        # BEGIN_YOUR_ANSWER (our solution is 1 lines of code, but don't worry if you deviate from this)
        return self.start
        # END_YOUR_ANSWER

    def isEnd(self, state):
        # BEGIN_YOUR_ANSWER (our solution is 1 lines of code, but don't worry if you deviate from this)
        return state == self.end
        # END_YOUR_ANSWER

    def succAndCost(self, state):
        # BEGIN_YOUR_ANSWER (our solution is 3 lines of code, but don't worry if you deviate from this)
        result = []
        for action, cost in self.path[state]:
            result.append(("Move {0} to {1}".format(state, action), action, cost))
        return result
        # END_YOUR_ANSWER

def admissibleButInconsistentHeuristic(state):
    # BEGIN_YOUR_ANSWER (our solution is 2 lines of code, but don't worry if you deviate from this)
    heuristicCost = {0:1, 1:1, 2:100, 3:1, 4:0}
    return heuristicCost[state]
    # END_YOUR_ANSWER

# Problem 4b: Apply a heuristic function to the joint segmentation-and-insertion problem

def makeWordCost(bigramCost, wordPairs):
    """
    :param bigramCost: learned bigram cost from a training corpus
    :param wordPairs: all word pairs in the training corpus
    :returns: wordCost, which is a function from word to cost
    """
    # BEGIN_YOUR_ANSWER (our solution is 4 lines of code, but don't worry if you deviate from this)
    wordCostDict = {} # key: word, value: minimum cost
    for w1, w2 in wordPairs:
        wordCostDict[w2] = min(bigramCost(w1, w2), wordCostDict[w2]) if w2 in wordCostDict.keys() else bigramCost(w1, w2)
        
    def wordCost(word):
        return wordCostDict[word] if word in wordCostDict.keys() else bigramCost(wordsegUtil.SENTENCE_UNK, word)
    
    return wordCost
    # END_YOUR_ANSWER

class RelaxedProblem(util.SearchProblem):
    def __init__(self, query, wordCost, possibleFills):
        self.query = query
        self.wordCost = wordCost
        self.possibleFills = possibleFills

    def startState(self):
        # BEGIN_YOUR_ANSWER (our solution is 1 lines of code, but don't worry if you deviate from this)
        return 0
        # END_YOUR_ANSWER

    def isEnd(self, state):
        # BEGIN_YOUR_ANSWER (our solution is 1 lines of code, but don't worry if you deviate from this)
        return state == len(self.query)
        # END_YOUR_ANSWER

    def succAndCost(self, state):
        # BEGIN_YOUR_ANSWER (our solution is 5 lines of code, but don't worry if you deviate from this)
        result = []
        # index 1씩 증가
        for indexTo in range(state+1, len(self.query)+1):
            # 해당 범위에서 가능한 모든 단어 취하기
            for word in self.possibleFills(self.query[state:indexTo]):
                result.append((word, indexTo, self.wordCost(word)))
        return result
        # END_YOUR_ANSWER

def makeHeuristic(query, wordCost, possibleFills):
    # BEGIN_YOUR_ANSWER (our solution is 2 lines of code, but don't worry if you deviate from this)
    problem = RelaxedProblem(query, wordCost, possibleFills)
    dpHelper = util.DynamicProgramming(problem)
    def heuristic(state):
        return dpHelper(state[0])
    
    return heuristic
    # END_YOUR_ANSWER

def fastSegmentAndInsert(query, bigramCost, wordCost, possibleFills):
    if len(query) == 0:
        return ''

    # BEGIN_YOUR_ANSWER (our solution is 4 lines of code, but don't worry if you deviate from this)
    ucs = util.UniformCostSearch()
    ucs.solve(JointSegmentationInsertionProblem(query, bigramCost, possibleFills), makeHeuristic(query, wordCost, possibleFills))
    return ' '.join(ucs.actions)
    # END_YOUR_ANSWER
    

############################################################

if __name__ == '__main__':
    shell.main()
