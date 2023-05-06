import pygame
from constants import *
from pacman import Pacman
from ghost import Ghost
from maze import Maze

pygame.init()

#setting up screen
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Pacman")

#create game objects
pacman = Pacman(PACMAN_START_POS, PACMAN_COLOR)
ghost = Ghost(GHOST_START_POS, GHOST_COLOR)
maze = Maze(MAZE_LAYOUT) #MAZE_LAYOUT is the 2D array representing the maze

#game loop
running = True
while running:
    #handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                pacman.move(0, -1)
            elif event.key == pygame.K_DOWN:
                pacman.move(0, 1)
            elif event.key == pygame.K_LEFT:
                pacman.move(-1, 0)
            elif event.key == pygame.K_RIGHT:
                pacman.move(1, 0)
            elif event.key == pygame.K_w:
                ghost.move(0, -1)
            elif event.key == pygame.K_s:
                ghost.move(0, 1)
            elif event.key == pygame.K_a:
                ghost.move(-1, 0)
            elif event.key == pygame.K_d:
                ghost.move(1, 0)

    #update game objects
    pacman.update()
    ghost.update()
    #draw game objects and maze
    screen.fill((0, 0, 0))
    for i in range(len(maze.layout)):
        for j in range(len(maze.layout[i])):
            tile_type = maze.get_tile_type((i, j))
            if tile_type == 1:
                pygame.draw.rect(screen, WALL_COLOR, (i*TILE_SIZE, j*TILE_SIZE, TILE_SIZE, TILE_SIZE))
    pacman.draw(screen)
    ghost.draw(screen)
    pygame.display.flip()

    """
    #check for collisions
    pacman_tile = maze.get_tile_type(pacman.get_position())
    ghost_tile = maze.get_tile_type(ghost.get_position())
    if pacman_tile == WALL or ghost_tile == WALL:
        #handle collision with wall
        print("Collide with wall")
    elif pacman.get_position() == ghost.get_position():
        #handle collision between Pacman and Ghost
        print("Ghost captures Pacman")
    """