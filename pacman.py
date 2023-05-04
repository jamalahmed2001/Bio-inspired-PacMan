import pygame
import sys

# Initialize Pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 800, 600
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
YELLOW = (255, 255, 0)
BLUE = (0, 0, 255)
PACMAN_SIZE = 30
VELOCITY = 5

# Create screen
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Pac-Man")

# Pac-Man character
class PacMan:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.velocity = VELOCITY

    def draw(self, screen):
        pygame.draw.circle(screen, YELLOW, (self.x, self.y), PACMAN_SIZE // 2)

    def move(self, dx, dy, room):
        new_x = self.x + dx * self.velocity
        new_y = self.y + dy * self.velocity

        if not self.collides_with_room(new_x, new_y, room):
            self.x = new_x
            self.y = new_y

    def collides_with_room(self, x, y, room):
        pacman_rect = pygame.Rect(x - PACMAN_SIZE // 2, y - PACMAN_SIZE // 2, PACMAN_SIZE, PACMAN_SIZE)
        room_rect = pygame.Rect(room.x, room.y, room.width, room.height)
        return not room_rect.contains(pacman_rect)

# Room class
class Room:
    def __init__(self, x, y, width, height, color):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.color = color

    def draw(self, screen):
        pygame.draw.rect(screen, self.color, (self.x, self.y, self.width, self.height))

# Create Pac-Man instance
pacman = PacMan(WIDTH // 2, HEIGHT // 2)

# Create Room instance
room_width, room_height = 200, 200
room_x = (WIDTH - room_width) // 2
room_y = (HEIGHT - room_height) // 2
room = Room(room_x, room_y, room_width, room_height, BLUE)

# Main game loop
running = True
while running:
    screen.fill(BLACK)

    # Draw Room
    room.draw(screen)

    # Draw Pac-Man
    pacman.draw(screen)

    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            sys.exit()

    # Key handling
    keys = pygame.key.get_pressed()
    dx, dy = 0, 0
    if keys[pygame.K_LEFT]:
        dx = -1
    if keys[pygame.K_RIGHT]:
        dx = 1
    if keys[pygame.K_UP]:
        dy = -1
    if keys[pygame.K_DOWN]:
        dy = 1

    # Move Pac-Man
    pacman.move(dx, dy, room)

    # Update the display
    pygame.display.flip()
    pygame.time.delay(30)
