import pygame
import sys

#Size
WIDTH      = 800
HEIGHT     = 600
GRID_SIZE  = 6
FPS        = 5

#Set up the colors
BLACK   = (  0,   0,   0)
WHITE   = (255, 255, 255)
RED     = (255,   0,   0)
GREEN   = (  0, 255,   0)
BLUE    = (  0,   0, 255)
GREY    = ( 64,  64,  64)


pygame.init()
display = pygame.display.set_mode((WIDTH,HEIGHT))
pygame.display.set_caption("Dheerendra Kumar Singh")

fpsClock = pygame.time.Clock()

while True:
    #Fill background
    display.fill(BLACK)
    
    #Create Grid
    for i in range(0,WIDTH,GRID_SIZE):
        pygame.draw.line(display,GREY,(i,0),(i,HEIGHT),1)
    for i in range(0,HEIGHT,GRID_SIZE):
        pygame.draw.line(display,GREY,(0,i),(WIDTH,i),1)

    pygame.display.set_caption("Dheerendra Kumar Singh")
    
    for evt in pygame.event.get():
        if evt.type == pygame.QUIT:
            pygame.quit();
            sys.exit()
        elif evt.type == pygame.KEYDOWN:
            if evt.key == pygame.K_UP:
                FPS = FPS+5
            elif evt.key == pygame.K_DOWN:
                FPS = FPS-5
        elif evt.type == pygame.MOUSEBUTTONDOWN:
            pygame.display.set_caption("D K Singh")

        pygame.draw.line(display,RED,(60,60), (35,35),1)
        pygame.draw.circle(display,RED,(160,160),2,0)
        pygame.draw.circle(display,GREEN,(60,60),5,0)
        #pygame.draw.polygon(display, GREEN, ((146, 0), (291, 106), (236, 277), (56, 277), (0, 106)))
        
        pygame.display.update()
        fpsClock.tick(60)
        print(FPS)


def createGrid():
    for x in range(0,WIDTH,5):
        pygame.draw.line(display,GREY,(0,x),(HIGHT,x),1)
