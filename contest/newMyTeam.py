# baselineTeam.py
# ---------------
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

# negin :D
# baselineTeam.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from captureAgents import CaptureAgent
import random, time, util, sys
from game import Directions
from util import nearestPoint


#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first='TimidAgent', second='DefensiveReflexAgent'):
    """
  This function should return a list of two agents that will form the
  team, initialized using firstIndex and secondIndex as their agent
  index numbers.  isRed is True if the red team is being created, and
  will be False if the blue team is being created.

  As a potentially helpful development aid, this function can take
  additional string-valued keyword arguments ("first" and "second" are
  such arguments in the case of this function), which will come from
  the --redOpts and --blueOpts command-line arguments to capture.py.
  For the nightly contest, however, your team will be created without
  any extra arguments, so you should make sure that the default
  behavior is what you want for the nightly contest.
  """
    return [eval(first)(firstIndex), eval(second)(secondIndex)]
    # return [eval(first)(firstIndex)]


##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
    """
  A base class for reflex agents that chooses score-maximizing actions
  """

    def registerInitialState(self, gameState):


        CaptureAgent.registerInitialState(self, gameState)
        self.inferenceMods = {i: ExactInference(i, self.index, gameState) for i in self.getOpponents(gameState)}

    def chooseAction(self, gameState):
        """
    Picks among the actions with the highest Q(s,a).
    """

        actions = gameState.getLegalActions(self.index)

        # print gameState
        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()
        values = [self.evaluate(gameState, a) for a in actions]
        # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

        maxValue = max(values)
        bestActions = [a for a, v in zip(actions, values) if v == maxValue]

        foodLeft = len(self.getFood(gameState).asList())

        if foodLeft <= 2:
            bestDist = 9999
            for action in actions:
                successor = self.getSuccessor(gameState, action)
                pos2 = successor.getAgentPosition(self.index)
                dist = self.getMazeDistance(self.start, pos2)
                if dist < bestDist:
                    bestAction = action
                    bestDist = dist
            return bestAction

        return random.choice(bestActions)

    def getSuccessor(self, gameState, action):

        """
    Finds the next successor which is a grid position (location tuple).
    """
        successor = gameState.generateSuccessor(self.index, action)
        pos = successor.getAgentState(self.index).getPosition()

        if pos != nearestPoint(pos):
            # Only half a grid position was covered
            return successor.generateSuccessor(self.index, action)
        else:
            return successor

    def evaluate(self, gameState, action):
        """
    Computes a linear combination of features and feature weights
    """
        features = self.getFeatures(gameState, action)
        weights = self.getWeights(gameState, action)
        return features * weights

    def getFeatures(self, gameState, action):
        """
    Returns a counter of features for the state
    """
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        features['successorScore'] = self.getScore(successor)
        return features

    def getWeights(self, gameState, action):
        """
    Normally, weights do not depend on the gamestate.  They can be either
    a counter or a dictionary.
    """
        return {'successorScore': 1.0}

    def getMostDenseArea(self, gameState):
        ourFood = self.getFoodYouAreDefending(gameState).asList()
        distance = [self.getMazeDistance(gameState.getAgentPosition(self.index), a) for a in ourFood]
        nearestFood = ourFood[0]
        nearestDstance = distance[0]

        for i in range(len(distance)):
            if distance[i] < nearestDstance:
                nearestFood = ourFood[i]
                nearestDstance = distance[i]
        return nearestFood


class DefensiveReflexAgent(ReflexCaptureAgent):
    lastSuccess = 0
    flag = 1
    flag2 = 0
    currentFoods = []
    s = []

    """
    A reflex agent that keeps its side Pacman-free. Again,
    this is to give you an idea of what a defensive agent
    could be like.  It is not the best or only way to make
    such an agent.
    """

    def getFeatures(self, gameState, action):
        for i in self.inferenceMods:
            self.inferenceMods[i].step(gameState)
        self.start = self.getMostDenseArea(gameState)

        features = util.Counter()
        successor = self.getSuccessor(gameState, action)

        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()
        self.s = (gameState.data.layout.width/2, gameState.data.layout.height/2)
        # print self.s
        # Computes whether we're on defense (1) or offense (0)
        features['onDefense'] = 1
        if myState.isPacman: features['onDefense'] = 0

        features['Boundries'] = self.getMazeDistance(myPos, self.s)

        if (self.flag2 == 0):
            self.flag2 = 1
            self.currentFoods = self.getFoodYouAreDefending(gameState).asList()
        # Computes distance to invaders we can see
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]

        features['numInvaders'] = len(invaders)
        if len(invaders) == 0:
            prob = [self.inferenceMods[a].getMostLikelyPosition() for a in
                       self.inferenceMods]
            enemy_distance = [self.getMazeDistance(myPos, self.inferenceMods[a].getMostLikelyPosition()) for a in
                             self.inferenceMods]
            nearest_enemy = enemy_distance[0]
            for i in range(len(enemy_distance)):
                if enemy_distance[i] < nearest_enemy:
                    nearest_enemy = enemy_distance[i]
            # print nearest_enemy
            # print prob[0]
            features["nearestEnemy"] = nearest_enemy

        if len(invaders) > 0:
            features["nearestEnemy"] = 0
            dists = [9999,9999]
            c_dist = [9999,9999]
            pos = [a.getPosition() for a in invaders]
            pro_pos = [self.inferenceMods[a].getMostLikelyPosition() for a in
                       self.inferenceMods]

            if len(pos) != 0:
                for i in range(len(pos)):
                    dists[i] = self.getMazeDistance(myPos, pos[i])
                    # c_dist[i] = self.getMazeDistance(pos[i], self.getCapsulesYouAreDefending(gameState)[0])
                # nearestPos = pos[0]
                nearestDst = dists[0]
                # nearcap = c_dist[0]

                for i in range(len(dists)):
                    if dists[i] < nearestDst:
                        # nearestPos = pos[i]
                        nearestDst = dists[i]
                    # if c_dist[i] < nearcap:
                    #     # nearestPos = pos[i]
                    #     nearcap = c_dist[i]

            else:

                for i in range(len(pro_pos)):
                    dists[i] = self.getMazeDistance(myPos, pro_pos[i])
                    # c_dist[i] = self.getMazeDistance(pro_pos[i], self.getCapsulesYouAreDefending(gameState))

                nearestDst = dists[0]
                # nearcap = c_dist[0]

                for i in range(len(dists)):
                    if dists[i] < nearestDst:
                        # nearestPos = pos[i]
                        nearestDst = dists[i]
                    # if c_dist[i] < nearcap:
                    #     # nearestPos = pos[i]
                    #     nearcap = c_dist[i]

            # print nearcap
            # features['nearcapsule'] = 0
            # if nearcap < 6:
            #     features['nearcapsule'] = 1000
            #     print "WARNING"

            features['invaderPDistance'] = nearestDst

        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        return features

    def getWeights(self, gameState, action):
        return {'numInvaders': -1000, 'onDefense': 100, 'invaderPDistance': -20,
                 'stop': -100, 'Boundries': -10, 'reverse': -2, 'nearestEnemy' : -15}


class TimidAgent(CaptureAgent):
    """
  A Dummy agent to serve as an example of the necessary agent structure.
  You should look at baselineTeam.py for more details about how to
  create an agent as this is the bare minimum.
  """

    def __init__(self, index, timeForComputing=.1):
        CaptureAgent.__init__(self, index, timeForComputing=.1)
        self.escapepath = []
        self.eaten = 0
        self.height = 0
        self.width = 0
        self.plan = [[], []]

    def registerInitialState(self, gameState):
        """
    This method handles the initial setup of the
    agent to populate useful fields (such as what team
    we're on).

    A distanceCalculator instance caches the maze distances
    between each pair of positions, so your agents can use:
    self.distancer.getDistance(p1, p2)

    IMPORTANT: This method may run for at most 15 seconds.
    """
        self.eaten = 0


        self.height = len(gameState.getWalls()[0])
        for w in gameState.getWalls().asList():
            if w[1] == 0:
                self.width += 1

        print self.height
        print self.width


        '''
    Make sure you do not delete the following line. If you would like to
    use Manhattan distances instead of maze distances in order to save
    on initialization time, please take a look at
    CaptureAgent.registerInitialState in captureAgents.py.
    '''
        CaptureAgent.registerInitialState(self, gameState)

        '''
    Your initialization code goes here, if you need any.
    '''

    def chooseAction(self, gameState):
        """
    Picks among actions randomly.
    """
        start = time.time()

        mypos = gameState.getAgentPosition(self.index)

        # see if the position has food or not in last observasion
        if self.getPreviousObservation() is not None:
            if self.getPreviousObservation().hasFood(mypos[0], mypos[1]):
                self.eaten += 1

        nearestFood = self.nearestFood(gameState)
        nearestEnemy = self.getNearestEnemy(gameState)
        if self.index%2 == 1:
            opZone = int(self.width / 2) - 1
            oppositeZone = int(self.width / 2) - 2
            myEscapePoint = int(self.width / 2) + 3
            myHomePoint = int(self.width / 2)
        else:
            opZone = int(self.width / 2)
            oppositeZone = int(self.width / 2) + 1
            myEscapePoint = int(self.width / 2) - 4
            myHomePoint = int(self.width / 2) - 1

        if not gameState.getAgentState(self.index).isPacman:
            self.eaten = 0
            while len(self.plan[0]) == 0:
                y = random.choice(range(0, self.height, 1))
                if not gameState.hasWall(opZone, y):
                    # plan = []
                    self.plan = [[oppositeZone, y],
                                 self.bfs(gameState, self.width, self.height, nearestEnemy,
                                          [opZone, y])]
                    # print self.plan[1]
            if len(self.plan[1]) == 0:
                if not len(self.plan[0]) == 0:
                    self.plan[1] = self.bfs(gameState, self.width, self.height, nearestEnemy, self.plan[0])
            self.escapepath = self.plan[1]
        else:
            print "CHANGE!!!!!!!"
            self.plan = [[], []]
            if len(nearestEnemy) > 0:
                if nearestEnemy[1] < 4 : #and len(self.escapepath) == 0:
                    self.escapepath = self.bfs(gameState, self.width, self.height, nearestEnemy, [myEscapePoint])
                    print "RUN!!!"
            else:
                self.escapepath = []
                if self.eaten == 5:
                    self.escapepath = self.bfs(gameState, self.width, self.height, nearestEnemy, [myHomePoint])
                    print "ENOUGH FOOD GO HOME"

        # self.debugDraw(self.escapepath, [1.0, 1.0, 1.0], True)
        # print self.escapepath

        actions = gameState.getLegalActions(self.index)
        values = [self.evaluate(gameState, nearestFood, nearestEnemy, self.escapepath, a) for a in actions]
        maxValue = max(values)
        # print values
        bestActions = [a for a, v in zip(actions, values) if v == maxValue]

        action = random.choice(bestActions)
        return action

    def escapePath(self, game_state, width, height, enemy):

        stack = util.Stack()
        myState = game_state.getAgentState(self.index)
        myPos = myState.getPosition()

        visited = []
        if len(enemy) > 0:
            visited = [[enemy[0][0], enemy[0][1]]]
        stack.push([myPos[0], myPos[1]])
        path = []

        while not stack.isEmpty():

            myPos = [int(myPos[0]), int(myPos[1] + 0.5)]
            psize = len(visited)
            loop = []

            right = [myPos[0] - 1, myPos[1]]
            if right[0] >= 0 and right not in visited and not game_state.hasWall(right[0], right[1]):
                stack.push(right)
                visited.append(right)
            if right in visited: loop.append(right)

            up = [myPos[0], myPos[1] + 1]
            if up[1] < height and up not in visited and not game_state.hasWall(up[0], up[1]):
                stack.push(up)
                visited.append(up)
            if up in visited:  loop.append(up)

            down = [myPos[0], myPos[1] - 1]
            if down[1] >= 0 and down not in visited and not game_state.hasWall(down[0], down[1]):
                stack.push(down)
                visited.append(down)
            if down in visited: loop.append(down)

            left = [myPos[0] + 1, myPos[1]]
            if left[0] < width and left not in visited and not game_state.hasWall(left[0], left[1]):
                stack.push(left)
                visited.append(left)
            if left in visited: loop.append(left)

            if len(loop) > 0:
                for i in reversed(path):
                    if abs(i[0] - myPos[0]) + abs(i[1] - myPos[1]) > 1:
                        path.remove(i)
                    else:
                        break

            if myPos[0] == 1 and myPos[1] == 2:
                # self.debugDraw(path, [1.0, 1.0, 1.0], True)
                # print path
                return path

            myPos = stack.pop()

            for i in reversed(path):
                if abs(i[0] - myPos[0]) + abs(i[1] - myPos[1]) > 1:
                    path.remove(i)
                else:
                    break
            path.append(myPos)
        return path

    def nearestFood(self, gameState):

        food = self.getFood(gameState).asList()
        distance = [self.getMazeDistance(gameState.getAgentPosition(self.index), a) for a in food]

        # get back to own half
        if len(food) < 3:
            previous = self.getFoodYouAreDefending(gameState).asList()[0]
            return [previous, self.getMazeDistance(gameState.getAgentPosition(self.index), previous)]
        nearestFood = food[0]
        nearestDstance = distance[0]

        for i in range(len(distance)):
            if distance[i] < nearestDstance:
                nearestFood = food[i]
                nearestDstance = distance[i]

        return [nearestFood, nearestDstance]

    def getNearestEnemy(self, gameState):

        myState = gameState.getAgentState(self.index)
        myPos = myState.getPosition()

        # Computes distance to invaders we can see
        enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        invaders = [a for a in enemies if not a.isPacman and a.getPosition() != None]
        dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
        scare = 0
        if len(invaders) == 0:
            return []
        else:
            nearestEnemy = invaders[0].getPosition()
            isPacman = invaders[0].isPacman
            nearestDstance = dists[0]
            for i in range(len(dists)):
                if dists[i] < nearestDstance:
                    nearestEnemy = invaders[i].getPosition()
                    nearestDstance = dists[i]
                    scare = invaders[i].scaredTimer
        # print scare
        # self.debugDraw(nearestEnemy, [1.0, 0.5, 0.5], True)
        return [nearestEnemy, nearestDstance, scare, isPacman]

    def evaluate(self, gameState, nearestFood, nearestEnemy, escapepath, action):

        score = 0
        scorelist = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        next = gameState.generateSuccessor(self.index, action)
        nextpos = next.getAgentPosition(self.index)
        nextscore = next.getScore()

        if nextscore > gameState.getScore():
            score += 2
            scorelist[0] = 2

        # if len(nearestEnemy) > 0 > nearestEnemy[2] and nearestEnemy[1] < 3:
        #     score -= 2 * (self.getMazeDistance(next.getAgentPosition(self.index), nearestEnemy[0]) - nearestEnemy[1])
        #     scorelist[1] -= 2 * (
        #         self.getMazeDistance(next.getAgentPosition(self.index), nearestEnemy[0]) - nearestEnemy[1])

        # if len(nearestEnemy) == 0 or nearestEnemy[1:
        if self.getMazeDistance(next.getAgentPosition(self.index), nearestFood[0]) < nearestFood[1]:
            score += 1
            scorelist[2] = 1

        pre = self.getPreviousObservation()
        if pre != None:
            if self.getPreviousObservation().getAgentPosition(self.index) == nextpos:
                score -= 5
                scorelist[3] = -5

        if len(nearestEnemy) > 0 and nearestEnemy[1] < 4:
            if next.getAgentState(self.index).isPacman:
                score += (self.getMazeDistance(next.getAgentPosition(self.index), nearestEnemy[0]) - nearestEnemy[1])
                scorelist[4] = (
                    self.getMazeDistance(next.getAgentPosition(self.index), nearestEnemy[0]) - nearestEnemy[1])
                nextActions = next.getLegalActions(self.index)
                if len(nextActions) == 2:
                    score -= 100
                    scorelist[5] = -100
        else:
            score += 2
            scorelist[6] = 2

        if len(escapepath) > 0:
            if [nextpos[0], nextpos[1]] in escapepath:
                if not (len(nearestEnemy) > 0 > nearestEnemy[2] and nearestEnemy[1] < 3):
                    score += 10
                    scorelist[7] = 10
        if action == Directions.STOP:
            score = -10
            scorelist[8] = -10
        return score


    def bfs(self, game_state, width, height, enemy, point):

        queue = util.Queue()
        myState = game_state.getAgentState(self.index)
        myPos = myState.getPosition()

        visited = []
        if len(enemy) > 0:
            visited = [[enemy[0][0], enemy[0][1]]]
        queue.push([int(myPos[0]), int(myPos[1])])

        # i = 0
        path = []
        while not queue.isEmpty():

            # print i
            myPos = [int(myPos[0]), int(myPos[1] + 0.5)]
            psize = len(visited)
            loop = []

            right = [myPos[0] - 1, myPos[1]]
            if right[0] >= 0 and right not in visited and not game_state.hasWall(right[0], right[1]):
                queue.push(right)
                visited.append(right)
            if right in visited: loop.append(right)

            up = [myPos[0], myPos[1] + 1]
            if up[1] < height and up not in visited and not game_state.hasWall(up[0], up[1]):
                queue.push(up)
                visited.append(up)
            if up in visited:  loop.append(up)

            down = [myPos[0], myPos[1] - 1]
            if down[1] >= 0 and down not in visited and not game_state.hasWall(down[0], down[1]):
                queue.push(down)
                visited.append(down)
            if down in visited: loop.append(down)

            left = [myPos[0] + 1, myPos[1]]
            if left[0] < width and left not in visited and not game_state.hasWall(left[0], left[1]):
                queue.push(left)
                visited.append(left)

            if left in visited: loop.append(left)

            # if len(loop) > 0:
            #     for i in reversed(path):
            #         if abs(i[0] - myPos[0]) + abs(i[1] - myPos[1]) > 1:
            #             path.remove(i)
            #         else:
            #             break
            myPos = queue.pop()
            path.append(myPos)

            # print path
            if len(point) == 1:
                if myPos[0] == point[0]:
                    a = myPos
                    f = []
                    for i in reversed(path):
                        if abs(a[0] - i[0]) + abs(a[1] - i[1]) <= 1:
                            f.append(i)
                            a = i
                            # self.debugDraw(f, [1.0, 1.0, 1.0], True)
                    return f
            else:

                if myPos[0] == point[0] and myPos[1] == point[1]:
                    a = myPos
                    f = []
                    for i in reversed(path):
                        # print len(path)
                        if abs(a[0] - i[0]) + abs(a[1] - i[1]) <= 1:
                            f.append(i)
                            a = i
                            # print len(f)
                            # self.debugDraw(f, [1.0, 1.0, 1.0], True)
                    return f
        return []

class ExactInference:
  def __init__(self, targetIndex,myIndex,gameState):

    #Init beliefs
    self.beliefs = util.Counter()
    width, height = gameState.data.layout.width, gameState.data.layout.height
    for i in range(width):
      for j in range(height):
        self.beliefs[(i,j)]=0.0 if gameState.hasWall(i,j) else 1.0
    self.beliefs.normalize()

    #Initialize targetIndex
    self.targetIndex=targetIndex

    #Initialize my index
    self.index=myIndex

  def getMostLikelyPosition(self):
    return self.beliefs.argMax()

  def step(self,gameState):
    self.elapseTime(gameState)
    self.observe(gameState)

  def observe(self,gameState):
    absPos = gameState.getAgentPosition(self.targetIndex)
    noisyDistance = gameState.getAgentDistances()[self.targetIndex]
    if absPos:
      for pos in self.beliefs:
        self.beliefs[pos]=1.0 if pos == absPos else 0.0
    else:
      for pos in self.beliefs:
        dist = util.manhattanDistance(pos,gameState.getAgentPosition(self.index))
        self.beliefs[pos]*=gameState.getDistanceProb(dist,noisyDistance)
      self.beliefs.normalize()

  def elapseTime(self,gameState):
    newBeliefs = util.Counter()

    for pos in self.beliefs.keys():
      if self.beliefs[pos]>0:
        possiblePositions={}
        x,y=pos
        for dx,dy in ((-1,0),(0,0),(1,0),(0,-1),(0,1)):
          if not gameState.hasWall(x+dx,y+dy):
            possiblePositions[(x+dx,y+dy)]=1
        prob=1.0/len(possiblePositions)
        for possiblePosition in possiblePositions:
          newBeliefs[possiblePosition]+=prob*self.beliefs[pos]
    newBeliefs.normalize()
    self.beliefs=newBeliefs
    if self.beliefs.totalCount()<=0.0:

      width, height = gameState.data.layout.width, gameState.data.layout.height
      for i in range(width):
        for j in range(height):
          self.beliefs[(i,j)]=0.0 if gameState.hasWall(i,j) else 1.0
      self.beliefs.normalize()
