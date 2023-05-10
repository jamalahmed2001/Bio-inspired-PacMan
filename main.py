import pygame
import sys
from pacman import *

pygame.init()
pygame.font.init()

# Create the screen
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Pac-Man")
font = pygame.font.Font(None, 36)

# Create Room instance
room = Room(1)  # 0 is the empty box, 1 is the easy map, 2 is the medium map
START_POS, walls, dots = room.create_map()

# Create Pac-Man instance
pacman = PacMan(START_POS[0], START_POS[1], YELLOW, dotsEaten)

# Create ghosts
redPos = random_pos()
redGhost = Ghost(redPos[0], redPos[1], random_dir(), RED)

greenPos = random_pos()
greenGhost = Ghost(greenPos[0], greenPos[1], random_dir(), GREEN)

orangePos = random_pos()
orangeGhost = Ghost(orangePos[0], orangePos[1], random_dir(), ORANGE)

pinkPos = random_pos()
pinkGhost = Ghost(pinkPos[0], pinkPos[1], random_dir(), PINK)

ghosts = [redGhost, greenGhost, orangeGhost, pinkGhost]

running = True
while running:

    # create Environment
    screen.fill(BLACK)
    room.draw_map(screen)
    score_text = font.render(f'Score: {pacman.dotsEaten}', False, (WHITE))
    screen.blit(score_text, (10, 10))

    # draw PacMan and ghosts
    pacman.draw(screen)
    redGhost.draw(screen)
    pinkGhost.draw(screen)
    greenGhost.draw(screen)
    orangeGhost.draw(screen)

    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            sys.exit()
        elif event.type == pygame.KEYDOWN:
            pacman.velocity = 0.2
            if event.key == pygame.K_UP:
                pacman.dir = "up"
            elif event.key == pygame.K_DOWN:
                pacman.dir = "down"
            elif event.key == pygame.K_LEFT:
                pacman.dir = "left"
            elif event.key == pygame.K_RIGHT:
                pacman.dir = "right"

    # pacman movements
    pacman.control()
    pacman.wallCollision()
    pacman.dotCollision()
    for ghost in ghosts:
        pacman.moveableCollision(ghost)

    # ghost movements
    for ghost in ghosts:
        ghost.control()
        ghost.wallCollision()
        # ghost.moveableCollision(ghost)

    redGhost.moveableCollision(greenGhost)
    redGhost.moveableCollision(orangeGhost)
    redGhost.moveableCollision(pinkGhost)
    greenGhost.moveableCollision(orangeGhost)
    greenGhost.moveableCollision(pinkGhost)
    orangeGhost.moveableCollision(pinkGhost)

    # Update the display
    pygame.display.update()
