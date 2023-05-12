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
room = Room(0)  # 0 is the empty box, 1 is the easy map, 2 is the medium map
START_POS, walls, dots = room.create_map()

# Create Pac-Man instance
pacman = PacMan(START_POS[0], START_POS[1], YELLOW, dotsEaten)

# # Create ghosts
# redPos = random_pos()
# redGhost = Ghost(redPos[0], redPos[1], random_dir(), RED)

# greenPos = random_pos()
# greenGhost = Ghost(greenPos[0], greenPos[1], random_dir(), GREEN)

# orangePos = random_pos()
# orangeGhost = Ghost(orangePos[0], orangePos[1], random_dir(), ORANGE)

# pinkPos = random_pos()
# pinkGhost = Ghost(pinkPos[0], pinkPos[1], random_dir(), PINK)

# movers = [redGhost, greenGhost, orangeGhost, pinkGhost, pacman]

# Set the condition to end a generation (e.g., reaching a certain score or a specific number of iterations)
max_generation_iterations = 1000  # Define your desired condition

# Main game loop
generation_counter = 0

neural_ghosts = []
for _ in range(4):
    (x, y) = random_pos()
    neural_ghost = NeuralGhost(x, y, random.choice(
        [PINK, GREEN, ORANGE, RED]), random_dir(), 4, 12, 8)
    neural_ghosts.append(neural_ghost)

running = True
while running and generation_counter < max_generation_iterations:

    # create Environment
    screen.fill(BLACK)
    room.draw_map(screen)
    # score_text = font.render(f'Score: {pacman.dotsEaten}', False, (WHITE))
    # screen.blit(score_text, (10, 10))

    # # draw ghosts and PacMan
    # for mover in movers:
    #     mover.draw(screen)

    pacman.draw(screen)

    for neural_ghost in neural_ghosts:
        neural_ghost.draw(screen)

    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            sys.exit()
        # elif event.type == pygame.KEYDOWN:
        #     pacman.velocity = 0.3
        #     if event.key == pygame.K_UP:
        #         pacman.dir = "up"
        #     elif event.key == pygame.K_DOWN:
        #         pacman.dir = "down"
        #     elif event.key == pygame.K_LEFT:
        #         pacman.dir = "left"
        #     elif event.key == pygame.K_RIGHT:
        #         pacman.dir = "right"

    # for mover in movers:
    #     mover.control(movers)

    # Perform evolution process at the specified interval
    if generation_counter % 10 == 0:
        neural_ghost.evolve_population()

    # Render game objects
    # ...

    for neural_ghost in neural_ghosts:
        neural_ghost.control(neural_ghosts)

    # Update generation counter
    generation_counter += 1

    # Update display
    # pygame.display.flip()

    # Update the display
    pygame.display.update()
