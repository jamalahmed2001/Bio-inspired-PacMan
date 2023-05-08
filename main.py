import pygame
import sys
from pacman import *
pygame.init()

# Game Constants
WIDTH, HEIGHT = 800, 600
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
YELLOW = (255, 255, 0)
BLUE = (0, 0, 255)
PACMAN_SIZE = 30
VELOCITY = 5

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Pac-Man")

# Create Pac-Man instance
pacman = PacMan(WIDTH // 2, HEIGHT // 2)

# Create Room instance
room_width, room_height = 400, 400
room_x = (WIDTH - room_width) // 2
room_y = (HEIGHT - room_height) // 2
room = Room(room_x, room_y, room_width, room_height, BLUE)

running = True
while running:
    # create Environment
    screen.fill(BLACK)
    room.draw(screen)

    # create PacMan
    pacman.draw(screen)

    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            sys.exit()

    pacman.userControl(pygame, room)

    # Update the display
    pygame.display.flip()
    pygame.time.delay(30)
