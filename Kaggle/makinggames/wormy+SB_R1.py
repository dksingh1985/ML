# Wormy (a Nibbles clone)
# By Al Sweigart al@inventwithpython.com
# http://inventwithpython.com/pygame
# Released under a "Simplified BSD" license

import random, pygame, sys
import os
from pygame.locals import *
import numpy as np
import pandas as pd
from tensorflow.keras.models import *
from tensorflow.keras.layers import Dropout,Activation,Dense
from tensorflow.keras.optimizers import *
from tensorflow.keras.backend import argmax

sys.setrecursionlimit(900)



FPS = 20
WINDOWWIDTH = 200
WINDOWHEIGHT = 200
CELLSIZE = 5

assert WINDOWWIDTH % CELLSIZE == 0, "Window width must be a multiple of cell size."
assert WINDOWHEIGHT % CELLSIZE == 0, "Window height must be a multiple of cell size."
CELLWIDTH = int(WINDOWWIDTH / CELLSIZE)
CELLHEIGHT = int(WINDOWHEIGHT / CELLSIZE)

MODEL_SAVE_PATH = "D:\Data\worm\worm.hd5f"
MEM_LOC = "D:\Data\worm\mem-"

#             R    G    B
WHITE     = (255, 255, 255)
BLACK     = (  0,   0,   0)
RED       = (255,   0,   0)
GREEN     = (  0, 255,   0)
BLUE      = (200, 200,  255)
DARKGREEN = (  0, 155,   0)
DARKGRAY  = ( 40,  40,  40)
BGCOLOR = BLACK

UP = 'up'
DOWN = 'down'
LEFT = 'left'
RIGHT = 'right'

HEAD = 0 # syntactic sugar: index of the worm's head

'''
if (os.path.exists(MEM_LOC + "IN.csv")):
    df_train_in = pd.read_csv(MEM_LOC + "IN.csv", header=None)
    #df_train_out = pd.read_csv(MEM_LOC + "OUT.csv", header=None)
else:
    df_train_in = pd.DataFrame()
    #df_train_out = pd.DataFrame()
'''

def main():
    global FPSCLOCK, DISPLAYSURF, BASICFONT, MODEL

    pygame.init()
    FPSCLOCK = pygame.time.Clock()
    DISPLAYSURF = pygame.display.set_mode((WINDOWWIDTH, WINDOWHEIGHT))
    BASICFONT = pygame.font.Font('freesansbold.ttf', 18)
    pygame.display.set_caption('Wormy')

    MODEL = createAIModel()
        
    #showStartScreen()
    while True:
        #Run game logic
        print("Start new game--------------------------->")
        runGame()
        input("Next")
        #showGameOverScreen()


def runGame():
    # Set a random start point.
    startx = random.randint(5, CELLWIDTH - 6)
    starty = random.randint(5, CELLHEIGHT - 6)
    wormCoords = [{'x': startx,     'y': starty},
                  {'x': startx - 1, 'y': starty},
                  {'x': startx - 2, 'y': starty}]
    
    start_wormCoords = wormCoords.copy()
    
    direction = RIGHT

    TRAVEL = 0

    df_train_in = pd.DataFrame()
    df_train_out = pd.DataFrame()
    if (os.path.exists(MEM_LOC + "IN.csv")):
        df_train_in = pd.read_csv(MEM_LOC + "IN.csv", header=None)
        df_train_out = pd.read_csv(MEM_LOC + "OUT.csv", header=None)
            
    # Start the apple in a random place.
    apple = getRandomLocation()
    reward_matrix = calculateRewardMatrix(start_wormCoords, apple)
    moves = findBestPath1(start_wormCoords, apple) #findBestPath(start_wormCoords[0],reward_matrix)

    while True: # main game loop
        REWARD = 0
        
        state = getState(wormCoords,apple)
        action = np.zeros( shape=(1,4))
                
        for event in pygame.event.get(): # event handling loop
            if event.type == QUIT:
                MODEL.save(MODEL_SAVE_PATH, overwrite=True)
                terminate()
            elif event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    MODEL.save(MODEL_SAVE_PATH, overwrite=True)
                    terminate()
            
        #print("Model_IN ->", len(model_in[0]))


        #action = MODEL.predict(state)

        #t_direction = np.argmax(action, axis=1)

        if (len(moves)>0):
            wormCoords,direction = moveWorm(wormCoords,moves[0])
            print("Direction:", direction)
            moves = moves[1:]
        
        
       # check if worm has eaten an apply
        if wormCoords[HEAD]['x'] == apple['x'] and wormCoords[HEAD]['y'] == apple['y']:
            # don't remove worm's tail segment
            apple = getRandomLocation() # set a new apple somewhere
            start_wormCoords = wormCoords.copy()
            reward_matrix = calculateRewardMatrix(start_wormCoords, apple)
            moves = findBestPath1(start_wormCoords, apple)#findBestPath(start_wormCoords[0],reward_matrix)
            REWARD = 1
        else:
             wormCoords = np.delete(wormCoords,-1) # remove worm's tail segment
        
        # check if the worm has hit itself or the edge
        if wormCoords[HEAD]['x'] < 0 or wormCoords[HEAD]['x'] > CELLWIDTH-1 or wormCoords[HEAD]['y'] < 0 or wormCoords[HEAD]['y'] > CELLHEIGHT-1:
            print("Hit wall")
            REWARD = -1
            #return # game over
        for wormBody in wormCoords[1:]:
            if wormBody['x'] == wormCoords[HEAD]['x'] and wormBody['y'] == wormCoords[HEAD]['y']:
                REWARD = -1
                print("Hit itself")
                #return # game over

        if (REWARD == -2):
            print("Traning Model")
            print("------------------------------")
            TRAIN_IN = np.zeros(shape=(1,2*(CELLWIDTH * CELLHEIGHT)))
            TRAIN_OUT = np.zeros( shape=(1,4))
                
            for i in range(moves.shape[0]-1):
                TRAIN_IN = np.vstack([TRAIN_IN,getState(start_wormCoords,apple)[0]])
                TRAIN_OUT = np.vstack([TRAIN_OUT,getAction(moves[i])[0]])
                start_wormCoords,temp = moveWorm(wormCoords,moves[i])
                del start_wormCoords[-1]
                
                
            #print("TRAIN_OUT:",TRAIN_OUT)
            MODEL.fit(TRAIN_IN[1:,:], TRAIN_OUT[1:,:], batch_size=10, epochs=10, shuffle=True)

            '''
            df_train_in = df_train_in.append(pd.DataFrame(TRAIN_IN))
            df_train_out = df_train_out.append(pd.DataFrame(TRAIN_OUT))

            df_train_in.to_csv(MEM_LOC + "IN.csv",index = False, header=False)
            df_train_out.to_csv(MEM_LOC + "OUT.csv",index = False , header=False)
            '''
            
            MODEL.save(MODEL_SAVE_PATH, overwrite=True)

            #return # game over


        #moves = findBestPath(start_wormCoords[0],reward_matrix)
        
        DISPLAYSURF.fill(BGCOLOR)
        drawGrid()
        drawPath(wormCoords[0],moves)
        drawWorm(wormCoords)
        drawApple(apple)
        #drawScore(len(wormCoords) - 3)
        pygame.display.update()
        FPSCLOCK.tick(FPS)

        #input("Next")
       
        if (REWARD == -1):
            print("Reward : Negative")
            print("wormCoords:",wormCoords)
            TRAVEL = 0
            return # game over
        
        elif (REWARD == 0):
            TRAVEL += 1
            print("Reward : Neutral:", TRAVEL)
            if (TRAVEL > 2000):
                return
            
        elif (REWARD == 1):
            TRAVEL = 0
            print("Reward : Positive")
                       
            
def getState(wormCoords,apple):
    state = np.zeros( shape=(1,2*(CELLWIDTH * CELLHEIGHT)))
    #print("apple", apple)

    state[0][(CELLWIDTH * CELLHEIGHT) + (apple['x'] * CELLWIDTH) + apple['y'] - 1 ] = 1

    #print("wormCoords:", wormCoords)
        
    for i in range(len(wormCoords)):
        indxW = (wormCoords[i]['x'] * CELLWIDTH)+ wormCoords[i]['y']
        #print("Worm Index[",i,"]:",indxW)
        state[0][indxW] = 1
        
    return state

def getAction(move):
    action = np.zeros( shape=(1,4))
    action[0][int(move)] = 1
    return action

def moveWorm(wormCoords,t_direction):
    newHead={}
    #print("Fn: moveWorm")
    #print("wormCoords:",wormCoords)
    #print("t_direction:",t_direction)
    if (t_direction == 0):
        direction = LEFT
        newHead = {'x': wormCoords[HEAD]['x'] - 1, 'y': wormCoords[HEAD]['y']}
    elif (t_direction == 1):
        direction = UP
        newHead = {'x': wormCoords[HEAD]['x'], 'y': wormCoords[HEAD]['y'] - 1}
    elif (t_direction == 2):
        direction = RIGHT
        newHead = {'x': wormCoords[HEAD]['x'] + 1, 'y': wormCoords[HEAD]['y']}
    elif (t_direction == 3):
        direction = DOWN
        newHead = {'x': wormCoords[HEAD]['x'], 'y': wormCoords[HEAD]['y'] + 1}

    wormCoords = np.insert(wormCoords, 0, newHead)
    
    return wormCoords,direction

def checkIfWormHit(wCoords):
    if wCoords[HEAD]['x'] < 0 or wCoords[HEAD]['x'] > CELLWIDTH-1 or wCoords[HEAD]['y'] < 0 or wCoords[HEAD]['y'] > CELLHEIGHT-1:
        return True
    for wormBody in wCoords[1:]:
        if wormBody['x'] == wCoords[HEAD]['x'] and wormBody['y'] == wCoords[HEAD]['y']:
            return True
    return False

def drawPressKeyMsg():
    pressKeySurf = BASICFONT.render('Press a key to play.', True, DARKGRAY)
    pressKeyRect = pressKeySurf.get_rect()
    pressKeyRect.topleft = (WINDOWWIDTH - 200, WINDOWHEIGHT - 30)
    DISPLAYSURF.blit(pressKeySurf, pressKeyRect)


def checkForKeyPress():
    if len(pygame.event.get(QUIT)) > 0:
        terminate()

    keyUpEvents = pygame.event.get(KEYUP)
    if len(keyUpEvents) == 0:
        return None
    if keyUpEvents[0].key == K_ESCAPE:
        terminate()
    return keyUpEvents[0].key


def showStartScreen():
    titleFont = pygame.font.Font('freesansbold.ttf', 100)
    titleSurf1 = titleFont.render('Wormy!', True, WHITE, DARKGREEN)
    titleSurf2 = titleFont.render('Wormy!', True, GREEN)

    degrees1 = 0
    degrees2 = 0
    while True:
        DISPLAYSURF.fill(BGCOLOR)
        rotatedSurf1 = pygame.transform.rotate(titleSurf1, degrees1)
        rotatedRect1 = rotatedSurf1.get_rect()
        rotatedRect1.center = (WINDOWWIDTH / 2, WINDOWHEIGHT / 2)
        DISPLAYSURF.blit(rotatedSurf1, rotatedRect1)

        rotatedSurf2 = pygame.transform.rotate(titleSurf2, degrees2)
        rotatedRect2 = rotatedSurf2.get_rect()
        rotatedRect2.center = (WINDOWWIDTH / 2, WINDOWHEIGHT / 2)
        DISPLAYSURF.blit(rotatedSurf2, rotatedRect2)

        drawPressKeyMsg()

        if checkForKeyPress():
            pygame.event.get() # clear event queue
            return
        pygame.display.update()
        FPSCLOCK.tick(FPS)
        degrees1 += 3 # rotate by 3 degrees each frame
        degrees2 += 7 # rotate by 7 degrees each frame


def terminate():
    pygame.quit()
    sys.exit()


def getRandomLocation():
    return {'x': random.randint(0, CELLWIDTH - 1), 'y': random.randint(0, CELLHEIGHT - 1)}


def showGameOverScreen():
    gameOverFont = pygame.font.Font('freesansbold.ttf', 50)
    gameSurf = gameOverFont.render('Game', True, WHITE)
    overSurf = gameOverFont.render('Over', True, WHITE)
    gameRect = gameSurf.get_rect()
    overRect = overSurf.get_rect()
    gameRect.midtop = (WINDOWWIDTH / 2, 10)
    overRect.midtop = (WINDOWWIDTH / 2, gameRect.height + 10 + 25)

    DISPLAYSURF.blit(gameSurf, gameRect)
    DISPLAYSURF.blit(overSurf, overRect)
    drawPressKeyMsg()
    pygame.display.update()
    pygame.time.wait(500)
    checkForKeyPress() # clear out any key presses in the event queue

    while True:
        if checkForKeyPress():
            pygame.event.get() # clear event queue
            return

def drawScore(score):
    scoreSurf = BASICFONT.render('Score: %s' % (score), True, WHITE)
    scoreRect = scoreSurf.get_rect()
    scoreRect.topleft = (WINDOWWIDTH - 120, 10)
    DISPLAYSURF.blit(scoreSurf, scoreRect)
    pygame.display.set_caption(str(score) + '-Wormy')


def drawWorm(wormCoords):
    for coord in wormCoords:
        x = coord['x'] * CELLSIZE
        y = coord['y'] * CELLSIZE
        wormSegmentRect = pygame.Rect(x, y, CELLSIZE, CELLSIZE)
        pygame.draw.rect(DISPLAYSURF, DARKGREEN, wormSegmentRect)
        wormInnerSegmentRect = pygame.Rect(x + 4, y + 4, CELLSIZE - 8, CELLSIZE - 8)
        pygame.draw.rect(DISPLAYSURF, GREEN, wormInnerSegmentRect)

def drawPath(whCoords,moves):
    xh = whCoords['x']
    yh = whCoords['y']
    
    for move in moves:
        if (move == 0):
            xh -= 1
        elif (move == 1):
            yh -= 1
        elif (move == 2):
            xh += 1
        elif (move == 3):
            yh += 1
        x = xh * CELLSIZE
        y = yh * CELLSIZE
        wormSegmentRect = pygame.Rect(x, y, CELLSIZE, CELLSIZE)
        pygame.draw.rect(DISPLAYSURF, BLUE, wormSegmentRect)
        
def drawApple(coord):
    x = coord['x'] * CELLSIZE
    y = coord['y'] * CELLSIZE
    appleRect = pygame.Rect(x, y, CELLSIZE, CELLSIZE)
    pygame.draw.rect(DISPLAYSURF, RED, appleRect)


def drawGrid():
    for x in range(0, WINDOWWIDTH, CELLSIZE): # draw vertical lines
        pygame.draw.line(DISPLAYSURF, DARKGRAY, (x, 0), (x, WINDOWHEIGHT))
    for y in range(0, WINDOWHEIGHT, CELLSIZE): # draw horizontal lines
        pygame.draw.line(DISPLAYSURF, DARKGRAY, (0, y), (WINDOWWIDTH, y))

def createAIModel():
    
    if (os.path.exists(MODEL_SAVE_PATH)):
        print("Restoring model from disk.")
        model = load_model(MODEL_SAVE_PATH)
        model.summary()
        return model
    else:
        print("Creating new model.")
        INPUT_DIM = 2*(CELLWIDTH * CELLHEIGHT)
        model = Sequential()
        model.add(Dense(4096,activation='sigmoid', kernel_initializer="truncated_normal", input_dim=INPUT_DIM))
        model.add(Dense(2048,activation='sigmoid', kernel_initializer="truncated_normal"))
        model.add(Dense(1024,activation='sigmoid', kernel_initializer="truncated_normal"))
        model.add(Dense(512,activation='sigmoid', kernel_initializer="truncated_normal"))
        #model.add(Dense(128,activation='sigmoid', kernel_initializer="truncated_normal"))
        #model.add(Dense(64,activation='sigmoid', kernel_initializer="truncated_normal"))
        #model.add(Dense(32,activation='sigmoid', kernel_initializer="truncated_normal"))
        #model.add(Dense(16,activation='sigmoid', kernel_initializer="truncated_normal"))
        #model.add(Dense(8,activation='sigmoid', kernel_initializer="truncated_normal"))
        model.add(Dense(4,activation='sigmoid', kernel_initializer="truncated_normal"))
        model.compile(loss="mse", optimizer=Adam(lr = 1e-5), metrics=["accuracy"])
        model.summary()
        return model

def calculateRewardMatrix(wCoords,aCoord):
    reward_matrix = np.zeros(shape=(CELLWIDTH,CELLHEIGHT))
    reward_decay = 0.99

    a_x = aCoord['x']
    a_y = aCoord['y']
    
    for i in range(len(wCoords)):
        reward_matrix[wCoords[i]['x']][wCoords[i]['y']] = -1    

    reward_matrix[a_x][a_y] = 1

    for i in range(a_x,-1,-1):
        for j in range(a_y,-1,-1):
            if (reward_matrix[i][j] >= 0):
                if (i<(CELLWIDTH-1)):
                    if (reward_matrix[i][j] < (reward_decay * reward_matrix[i+1][j])):
                        reward_matrix[i][j] = (reward_decay * reward_matrix[i+1][j])
                if(j<(CELLHEIGHT-1)):
                    if (reward_matrix[i][j] < (reward_decay * reward_matrix[i][j+1])):
                        reward_matrix[i][j] = (reward_decay * reward_matrix[i][j+1])
                
        for j in range(a_y,CELLHEIGHT):
            if (reward_matrix[i][j] >= 0):
                if (i<(CELLWIDTH-1)):
                    if (reward_matrix[i][j] < (reward_decay * reward_matrix[i+1][j])):
                        reward_matrix[i][j] = (reward_decay * reward_matrix[i+1][j])

                if(j>0):
                    if (reward_matrix[i][j] < (reward_decay * reward_matrix[i][j-1])):
                        reward_matrix[i][j] = (reward_decay * reward_matrix[i][j-1])
                
            

    for i in range(a_x,CELLWIDTH):
        for j in range(a_y,-1,-1):
            if (reward_matrix[i][j] >= 0):
                if(i>0):
                    if (reward_matrix[i][j] < (reward_decay * reward_matrix[i-1][j])):
                        reward_matrix[i][j] = (reward_decay * reward_matrix[i-1][j])
                if(j<(CELLHEIGHT-1)):
                    if (reward_matrix[i][j] < (reward_decay * reward_matrix[i][j+1])):
                        reward_matrix[i][j] = (reward_decay * reward_matrix[i][j+1])
                    
        for j in range(a_y,CELLHEIGHT):
            if (reward_matrix[i][j] >= 0):
                if(i>0):
                    if (reward_matrix[i][j] < (reward_decay * reward_matrix[i-1][j])):
                        reward_matrix[i][j] = (reward_decay * reward_matrix[i-1][j])
                if(j>0):
                    if (reward_matrix[i][j] < (reward_decay * reward_matrix[i][j-1])):
                        reward_matrix[i][j] = (reward_decay * reward_matrix[i][j-1])
                
    return reward_matrix

def findBestPath(wHeadCoord,reward_matrix):
    moves = np.zeros(shape=(1))

    x = wHeadCoord['x']
    y = wHeadCoord['y']
    
    while (reward_matrix[x][y] != 1):
        move = findBestMove(x,y,reward_matrix)
        #print("Path value:",reward_matrix[x][y] , "Move:", move)
        #print("X:",x,"| Y:",y , "| Move:", move)
        
        if (move == 0):
            x -= 1
        elif (move == 1):
            y -= 1
        elif (move == 2):
            x += 1
        elif (move == 3):
            y += 1
        elif (move == -1):
            break

        moves = np.append(moves,move)
            
    return moves[1:]

def findBestMove(x,y,reward_matrix):
    move = -1
    if (x>0):
        if (reward_matrix[x][y] < reward_matrix[x-1][y]):
            move = 0
    if (y>0):
        if (reward_matrix[x][y] < reward_matrix[x][y-1]):
            move = 1
    if (x<CELLWIDTH-1):
        if (reward_matrix[x][y] < reward_matrix[x+1][y]):
            move = 2
    if (y<CELLHEIGHT-1):
        if(reward_matrix[x][y] < reward_matrix[x][y+1]):
            move = 3
    return move

def findBestPath1(wCoords,aCoord):
    print("findBestPath1 aCoord:",aCoord)
    print("findBestPath1 wCoords :", wCoords)
    wLen = len(wCoords)
    wStatus = [wCoords]
    travsMatrix = [[]] * (CELLWIDTH * CELLHEIGHT) 
    
    #np.zeros(shape=(CELLWIDTH,CELLHEIGHT,2),)
    ind = 1
     #for i in range(wLen):
        #wStatus[0][i][0] = wCoords[i]['x']
        #wStatus[0][i][1] = wCoords[i]['y']
        #wStatus = np.vstack([wStatus,wCoords])

    while (len(wStatus) > 0):
        isAppleEaten, tmoves, wStatus, travsMatrix = bfs(wStatus,aCoord,travsMatrix)
        if(isAppleEaten):
            #tmoves = np.array(tmoves, dtype=int)
            #tmoves = np.reshape(tmoves,(-1))
            print ("tmoves:", tmoves, ":", len(tmoves))
            '''
            for i in range(len(tmoves)):
                moves = np.append(moves,int(tmoves[i]))
                d_wormCoords,td = moveWorm(wCoords,d)
                if (i != len(tmoves)-1 ):
                    d_wormCoords = np.delete(twormCoords,-1)
            '''   
            return tmoves
        #print("len(wStatus):",len(wStatus), "wStatus:", wStatus)
        #input("Nesxt")
    
    return []
     
def bfs(wStatus,aCoord,travsMatrix):
    wCoords = wStatus[0]
    wStatus = np.delete(wStatus,0,0)

    indx = (CELLWIDTH * wCoords[0]['y']) + wCoords[0]['x']
    omoves = travsMatrix[indx]

    #print("BFS wCoords :", wCoords)
    if (len(omoves)> 100):
        print("Max move depth reached")
        return False, omoves, wStatus, travsMatrix
    else:
        rmoves = [0,1,2,3]

        for d in rmoves:
            twormCoords,td = moveWorm(wCoords,d)
            #twormCoords = np.delete(twormCoords,-1)

            indx = (CELLWIDTH * twormCoords[0]['y']) + twormCoords[0]['x']
            tmoves = np.append(omoves,d)

            
            if(checkIfWormHit(twormCoords) == True):
                #print("Wrong move:", len(tmoves),"Worm head:",twormCoords[0]['x'],twormCoords[0]['y'])
                x=1
                
            elif ((twormCoords[0]['x'] == aCoord['x']) and (twormCoords[0]['y'] == aCoord['y'])):
                travsMatrix[indx] = tmoves
                return True, tmoves, wStatus, travsMatrix
            
            elif (len(travsMatrix[indx]) == 0):
                #print("Index:", indx ,"len(travsMatrix[indx]):",len(travsMatrix[indx]))
                twormCoords = np.delete(twormCoords,-1)
                wStatus = np.vstack([wStatus,twormCoords])
                travsMatrix[indx] = tmoves
            else:
                #print("Index:", indx , "len(travsMatrix[indx]):",len(travsMatrix[indx]))
                x=1
            
        '''
            tmoves,wCoords = bfs(twormCoords,aCoord,moves)
            if(len(tmoves)>0):
                return tmoves,wCoords
        '''
    return False, omoves, wStatus, travsMatrix
    
    

if __name__ == '__main__':
    main()
