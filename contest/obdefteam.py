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
               first='whiteAgent', second='whiteAgent'):
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



class whiteAgent(CaptureAgent):

    def __init__(self, index, timeForComputing=.1):
        CaptureAgent.__init__(self, index, timeForComputing=.1)
        self.escapepath = []
        self.eaten = 0
        self.height = 0
        self.width = 0
        self.entry = None
        self.capsule = []
        self.isPower = False
        self.defence = False
        self.flag2 = 0
        self.currentFoods = []
        self.s = []
        self.target = []
        self.mateTarget = []


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
        self.capsule = self.getCapsules(gameState)
        self.defence = True
        self.update = None
        if self.index%2 == 0:
            self.teamMate = 2 - self.index
        else:
            self.teamMate = 4 - self.index
        print "I am ", self.index
        print "He is", self.teamMate
        print self.capsule
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
        self.s = self.getDefencePosition(gameState)
        self.lastupdate = self.getPacmanPosition(gameState)

        nearestOpponent = self.getNearestOpponent(gameState)
        mypos = gameState.getAgentPosition(self.index)
        nearestPacman = self.getNearestPacman(gameState)
        # print self.index,nearestPacman

        nearestFood = self.nearestFood(gameState)
        nearestEnemy = self.getNearestEnemy(gameState)
        nearestEscape = self.getNearestEscape(gameState)
        nearestCapsule = self.getNearestCapsule(gameState)
        self.isPower = self.getIsPower(gameState)
        if self.isAnyPacman(gameState):
            if len(nearestPacman)>0 and len(nearestPacman[0])>0:
                self.escapepath = self.defAstar(gameState,self.mhtheuristic,nearestPacman[0][0])
            elif len(self.lastupdate)>0:
                dis = 9999
                pos = self.lastupdate[0]
                for i in self.lastupdate:
                    if self.getMazeDistance(i, mypos)<=dis:
                        dis = self.getMazeDistance(i,mypos)
                        pos = i
                        self.update = i
                self.escapepath = self.defAstar(gameState, self.mhtheuristic, pos)
            elif not self.update is None :
                self.escapepath = self.defAstar(gameState, self.mhtheuristic, self.update)
        else:
            self.escapepath = self.defAstar(gameState, self.mhtheuristic, self.s)

        actions = gameState.getLegalActions(self.index)
        values = [self.evaluate(gameState, a,self.escapepath) for a in
                  actions]
        maxValue = max(values)
        bestActions = [a for a, v in zip(actions, values) if v == maxValue]

        action = random.choice(bestActions)
        return action



    def evaluate(self, gameState,action, escapepath):

        score = 0
        next = gameState.generateSuccessor(self.index, action)
        nextState = next.getAgentState(self.index)
        nextpos = next.getAgentPosition(self.index)
        nextscore = next.getScore()

        if not nextState.isPacman: score += 100


        enemies = [next.getAgentState(i) for i in self.getOpponents(next)]
        invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]

        score -= len(invaders) * 1000

        if len(escapepath) > 0:
            if [nextpos[0], nextpos[1]] in escapepath:
                score += 10

        if action == Directions.STOP: score -= 100
        rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
        if action == rev: score -= 2

        return score


    def mhtheuristic(self, myPos, goal):
        return self.getMazeDistance((myPos[0],myPos[1]),goal)

    def defAstar(self,gameState, heuristic, goal):
        myState = gameState.getAgentState(self.index)
        myPos = myState.getPosition()
        expended = util.PriorityQueue()
        expended.push(([int(myPos[0]),int(myPos[1])], [], 0), 0)
        visited = []
        while not expended.isEmpty():
            node = expended.pop()
            # print node
            curr_pos = node[0]
            curr_action = node[1]
            curr_cost = node[2]
            if curr_pos[0] == goal[0] and curr_pos[1] == goal[1]:
                a = curr_pos
                result = []
                for i in reversed(curr_action):
                    if abs(a[0] - i[0]) + abs(a[1] - i[1]) <= 1:
                        result.append(i)
                        a = i
                return result

            if curr_pos not in visited:
                visited.append(curr_pos)
            else:
                continue
            new_cost = curr_cost+1
            right = [curr_pos[0] - 1, curr_pos[1]]
            if right[0] >= 0 and right not in visited and not gameState.hasWall(right[0], right[1]):
                expended.push((right, curr_action+[right], new_cost), new_cost + heuristic(right, goal))

            up = [curr_pos[0], curr_pos[1] + 1]
            if up[1] < self.height and up not in visited and not gameState.hasWall(up[0], up[1]):
                expended.push((up, curr_action+[up], new_cost), new_cost + heuristic(up, goal))

            down = [curr_pos[0], curr_pos[1] - 1]
            if down[1] >= 0 and down not in visited and not gameState.hasWall(down[0], down[1]):
                expended.push((down, curr_action+[down], new_cost), new_cost + heuristic(down, goal))

            left = [curr_pos[0] + 1, curr_pos[1]]
            if left[0] < self.width and left not in visited and not gameState.hasWall(left[0], left[1]):
                expended.push((left, curr_action+[left], new_cost), new_cost + heuristic(left, goal))

        return []

    ###################################################################################################################
    ##############
    #Help function
    ##############
    ###################################################################################################################

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
        return [nearestEnemy, nearestDstance, scare, isPacman]


    def getNearestOpponent(self, gameState):
        myState = gameState.getAgentState(self.index)
        myPos = myState.getPosition()

        # Computes distance to invaders we can see
        enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        dists = [self.getMazeDistance(myPos, a.getPosition()) for a in enemies if a.getPosition() != None]
        scare = 0
        result = []
        if len(dists) == 0:
            return []
        else:
            isPacman = enemies[0].isPacman
            nearestDstance = dists[0]
            for i in range(len(enemies)):
                if not (enemies[i].getPosition() is None):
                    if self.getMazeDistance(myPos, enemies[i].getPosition()) <= nearestDstance:
                        nearestEnemy = enemies[i].getPosition()
                        nearestDstance = self.getMazeDistance(myPos, enemies[i].getPosition())
                        scare = enemies[i].scaredTimer
                        result = [nearestEnemy, nearestDstance, scare, isPacman]
            return result

    def getNearestPacman(self, gameState):
        myState = gameState.getAgentState(self.index)
        myPos = myState.getPosition()
        result = []
        # Computes distance to invaders we can see
        enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        dists = [self.getMazeDistance(myPos, a.getPosition()) for a in enemies if a.getPosition() != None and a.isPacman]
        if len(dists) == 0:
            return []
        elif len(dists) == 1:
            nearestDstance = dists[0]
            for i in range(len(enemies)):
                if not (enemies[i].getPosition() is None):
                    if self.getMazeDistance(myPos, enemies[i].getPosition()) == nearestDstance:
                        nearestEnemy = enemies[i].getPosition()
                        nearestDstance = self.getMazeDistance(myPos, enemies[i].getPosition())
                        result = [[nearestEnemy], [nearestDstance]]
        else:
            pos = [a.getPosition() for a in enemies]
            result = [pos,dists]
        return result


    def getNearestEscape(self, gameState):
        if self.index%2==1:
            w = int(self.width/2)
        else:
            w = int(self.width/2)-1
        myPos = gameState.getAgentPosition(self.index)
        mindis = 9999
        min_h = 0
        for i in range(self.height):
            if not gameState.hasWall(w, i):
                dis = self.getMazeDistance(myPos,(w,i))
                if dis<mindis:
                    mindis = dis
                    min_h = i
        return (w,min_h)

    def getSafestEntry(self, gameState):
        if self.index%2==1:
            w = int(self.width/2)-1
        else:
            w = int(self.width/2)
        while True:
            h = random.choice(range(0, self.height, 1))
            if not gameState.hasWall(w, h):
                return (w, h)


    def getIsPower(self, gameState):
        enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        for a in enemies:
            if not a.isPacman:
                if a.scaredTimer > 0:
                    return [True, a.scaredTimer]
        return [False, 0]

    def getNearestCapsule(self, gameState):
        myPos = gameState.getAgentPosition(self.index)
        nearest = []
        min_dis = 9999
        for i in self.capsule:
            if self.getMazeDistance(myPos,i)<min_dis:
                nearest = i
                min_dis = self.getMazeDistance(myPos,i)
        return nearest

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

    def getDefencePosition(self,gameState):
        if self.index%2==1:
            w = int(gameState.data.layout.width/2)
        else:
            w = int(gameState.data.layout.width/2)-1
        i = 0
        while not i<0:
            if self.index<self.teamMate:
                h = gameState.data.layout.height / 2 - i
            else:
                h = gameState.data.layout.height / 2 + i
            if not gameState.hasWall(w, h):
                return (w, h)
            else:
                i += 1



    def isAnyPacman(self,gameState):
        enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        for a in enemies:
            if a.isPacman:
                return True
        return False

    def getPacmanPosition(self, gameState):
        foods = self.getFoodYouAreDefending(gameState).asList()
        if self.getPreviousObservation() is not None:
            preFoods = self.getFoodYouAreDefending(self.getPreviousObservation()).asList()
        else:
            preFoods = foods
        eatenFoods = list(set(preFoods) - set(foods))
        if len(eatenFoods) > 2:
            return []
        else:
            return eatenFoods