
#screen dimensions
SCREEN_WIDTH = 600
SCREEN_HEIGHT = 600

#player properties
PACMAN_RADIUS = 15
PACMAN_SPEED = 15
PACMAN_START_POS = (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)
PACMAN_COLOR = (255, 255, 0)

#enemy properties
GHOST_RADIUS = 15
GHOST_SPEED = 10
GHOST_START_POS = (SCREEN_WIDTH // 4, SCREEN_HEIGHT // 4)
GHOST_COLOR = (255, 0, 0)

#maze
MAZE_LAYOUT = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 1, 1, 1, 1, 0, 1],
    [1, 0, 1, 0, 0, 0, 0, 1, 0, 1],
    [1, 0, 1, 0, 1, 1, 0, 1, 0, 1],
    [1, 0, 1, 0, 1, 1, 0, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
]
WALL_COLOR = (255, 255, 255)
TILE_SIZE = 20