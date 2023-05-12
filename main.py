import pygame
import sys
from pacman import *
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE' #mac thing 
pygame.init()
pygame.font.init()
# Create the screen
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Pac-Man")
font = pygame.font.Font(None, 36)


pacbrain = PacBrain()
num_gen = 10

for gen in range(num_gen):#generations
    print("Generatiom: ",gen)
    # Create Room instance
    room = Room(0)  # 0 is the empty box, 1 is the easy map, 2 is the medium map
    START_POS, walls, dots = room.create_map()

    # Create Pac-Man instance
    pacman = PacBody(START_POS[0], START_POS[1], YELLOW, 0)
    pacman.dir = random_dir()
    
    # Create ghosts
    ghosts = [Ghost(random_pos()[0], random_pos()[1],random_dir(), RED),
              Ghost(random_pos()[0], random_pos()[1],random_dir(), PINK),
              Ghost(random_pos()[0], random_pos()[1],random_dir(), GREEN),
              Ghost(random_pos()[0], random_pos()[1],random_dir(),ORANGE)]

    movers = [pacman] +ghosts

    EndGame = False
    while not EndGame:
        # create Environment
        screen.fill(BLACK)
        room.draw_map(screen)
        score_text = font.render(f'Score: {pacman.dotsEaten}', False, (WHITE))
        screen.blit(score_text, (10, 10))
        # draw ghosts and PacMan
        for mover in movers:
            mover.draw(screen)

        # Event handling need to close the game ?
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                EndGame = True
                sys.exit()

        for mover in movers:
            if mover == pacman:
                EndGame = pacbrain.make_action(pacman,movers,room ) # this needs to get the action and the reward and update model??
            else:
                mover.control(movers)
        # print(pacman.energy)
        # print(pacbrain.fitness)
        # Update the display  
        pygame.display.update()

    # Update model here 
    # pacbrain.update_network()

