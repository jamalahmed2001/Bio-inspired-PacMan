import torch
import numpy as np
import torch.optim as optim
import torch.nn as nn
import pygame
import random
import math
import time
import random
import os
from deap import base, creator, tools
import time 

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


# Game Constants
SCREEN_WIDTH = 700
SCREEN_HEIGHT = 750
TILE_SIZE = 50
MOVER_SIZE = TILE_SIZE//2
clock = pygame.time.Clock()

# colours
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
YELLOW = (255, 255, 0)
BLUE = (0, 0, 255)
RED = (255, 0, 0)
PINK = (255, 182, 193)
GREEN = (34, 139, 34)
ORANGE = (255, 165, 0)


# PacMan setup
PACMAN_SPEED = 0.3
PACMAN_DIR = "right"
dotsEaten = 0

# Ghosts setup
ghost_starts = []


def random_pos():
    random_position = random.randint(0, len(dots) - 1)
    if random_position not in ghost_starts:
        random_dot = dots[random_position]
        pos = (random_dot[0], random_dot[1])
        dots.remove(random_dot)
        ghost_starts.append(pos)
    else:
        random_pos()
    return pos


def random_dir():
    directions = ["left", "right", "down", "up"]
    dir = random.choice(directions)
    return dir

# Room
walls = []
dots = []
# Define the map
MAP = [

    # Empty box map

    [[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
     [1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1],
     [1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1],
     [1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1],
     [1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1],
     [1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1],
     [1, 2, 2, 2, 2, 2, 0, 2, 2, 2, 2, 2, 2, 1],
     [1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1],
     [1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1],
     [1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1],
     [1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1],
     [1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1],
     [1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1],
     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]],

    # Map with maze (easy)

    [[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1],
        [1, 2, 2, 2, 2, 2, 1, 2, 2, 1, 1, 1, 2, 1],
        [1, 2, 2, 2, 2, 2, 1, 2, 2, 1, 1, 1, 2, 1],
        [1, 1, 1, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 1],
        [1, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 1],
        [1, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1],
        [1, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 1],
        [1, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 1],
        [1, 2, 2, 2, 2, 2, 1, 2, 2, 2, 1, 1, 1, 1],
        [1, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 1],
        [1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1],
        [1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]],

    # Map with maze (medium?)

    [[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1],
        [1, 2, 1, 2, 1, 2, 1, 1, 2, 1, 2, 2, 2, 1],
        [1, 2, 2, 2, 2, 2, 1, 1, 2, 1, 2, 1, 2, 1],
        [1, 1, 1, 2, 1, 2, 1, 1, 2, 1, 2, 1, 2, 1],
        [1, 2, 2, 2, 1, 2, 2, 2, 2, 1, 2, 1, 2, 1],
        [1, 2, 1, 2, 1, 1, 1, 1, 2, 1, 2, 1, 2, 1],
        [1, 2, 1, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 1],
        [1, 2, 1, 2, 1, 2, 1, 2, 2, 1, 2, 1, 1, 1],
        [1, 2, 2, 2, 2, 2, 2, 2, 1, 1, 2, 2, 2, 1],
        [1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 2, 1],
        [1, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 1],
        [1, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2, 2, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
]


class Room:
    def __init__(self, map):
        self.state = map  # 0, 1, or 2

    def create_map(self):
        for row in range(len(MAP[self.state])):
            for col in range(len(MAP[self.state][row])):
                if MAP[self.state][row][col] == 1:
                    walls.append(
                        (col*TILE_SIZE, row*TILE_SIZE, TILE_SIZE, TILE_SIZE))
                if MAP[self.state][row][col] == 2:
                    dots.append((col*TILE_SIZE+TILE_SIZE//2,
                                row*TILE_SIZE+TILE_SIZE//2))
                if MAP[self.state][row][col] == 0:
                    # Define Pacman's starting position
                    START_POS = (col*TILE_SIZE+TILE_SIZE//2,
                                 row*TILE_SIZE+TILE_SIZE//2)
        return START_POS, walls, dots

    def draw_map(self, screen):
        for wall in walls:
            pygame.draw.rect(screen, BLUE, wall)  # walls
        for dot in dots:
            pygame.draw.circle(screen, WHITE, (dot), 7)  # dots

    #Speed of pacman is determined by the environment    
    def update_pacman_pos(self, pacman):
        x, y = pacman.x, pacman.y
        row, col = int(y // TILE_SIZE), int(x // TILE_SIZE)
        print(pacman.dir)

        if row > 0 and MAP[self.state][row-1][col] == 1:
            print("Obstacle detected above!")
            if pacman.dir != "down":
                return 0,"down"
        if row < len(MAP[self.state])-1 and MAP[self.state][row+1][col] == 1:
            print("Obstacle detected below!")
            if pacman.dir != "up":
                return 0,"up"
        if col > 0 and MAP[self.state][row][col-1] == 1:
            print("Obstacle detected to the left!")
            if pacman.dir !=  "right":
                return 0,"right"
        if col < len(MAP[self.state][row])-1 and MAP[self.state][row][col+1] == 1:
            print("Obstacle detected to the right!")
            if pacman.dir !=  "left":
                return 0,"left"
        return 0.3
        

class Moveable():
    def __init__(self, x, y, color):
        self.x = x
        self.y = y
        self.color = color
        self.size = MOVER_SIZE
        self.rect = pygame.Rect(
            self.x-self.size, self.y-self.size, self.size*2, self.size*2)
        self.distance_travelled = 0
        self.distance_between = float("inf")

    # controls movement of the object
    def control(self, movers):
        for mover in movers:
            self.collision(mover)
        
        if self.dir == "left":
            self.move(-1, 0)
        elif self.dir == "right":
            self.move(1, 0)
        elif self.dir == "up":
            self.move(0, -1)
        elif self.dir == "down":
            self.move(0, 1)

        # defines how to move to a new position
    def move(self, dx, dy):
        self.x = self.x + dx * self.velocity
        self.y = self.y + dy * self.velocity
        self.rect.x = self.x-self.size + dx * self.velocity
        self.rect.y = self.y-self.size + dy * self.velocity
        self.distance_travelled += abs(dx) + abs(dy)

    # detect all collisions
    def collision(self, collider):
        coll = False
        for wall in walls:
            if self.rect.colliderect(wall):
                coll = True
        if self is not collider and self.rect.colliderect(collider.rect):
            coll = True
        if coll:
            if self.dir == "left":
                self.dir = "up"
            elif self.dir == "right":
                self.dir = "down"
            elif self.dir == "up":
                self.dir = "right"
            elif self.dir == "down":
                self.dir = "left"
        return coll

    def get_position(self):
        return self.x, self.y

    def distance_travelled(self):
        return self.distance_travelled

    def get_distance_from(self, mover):
        if self is not mover:
            self.distance = math.sqrt((self.x - mover.x) **
                                      2 + (self.y - mover.y) ** 2)
        return self.distance_between
class Ghost(Moveable):
    def __init__(self, x, y, dir, color):
        super(Ghost, self).__init__(x, y, color)
        self.velocity = 0.3
        self.dir = dir

    def draw(self, screen):
        pygame.draw.rect(
            screen, self.color, self.rect)
class GhostNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GhostNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x




class PacBody():        
    def __init__(self, x, y, color,dotsEaten):
        self.x = x
        self.y = y
        self.color = color
        self.size = MOVER_SIZE
        self.rect = pygame.Rect(
            self.x-self.size, self.y-self.size, self.size*2, self.size*2)
        self.distance_travelled = 0 
        self.velocity = PACMAN_SPEED
        self.dir = PACMAN_DIR
        self.dotsEaten = dotsEaten
        self.energy = 100 - self.distance_travelled +self.dotsEaten
        self.stuck = False

    def draw(self, screen):
        pygame.draw.circle(screen, self.color, (self.x, self.y), self.size)

    def dotCollision(self):
        for dot in dots:
            if self.rect.colliderect(pygame.Rect(dot[0], dot[1], 7, 7)):
                dots.remove(dot)
                self.dotsEaten += 1


    #Gets signal from body as to how that action affected the body?
    def take_action(self, direction, movers,room):
        for mover in movers:
            if self is not mover and self.rect.colliderect(mover.rect): #killed by ghost 
                print("Killed")
                return -10,True
        # self.velocity = room.update_pacman_pos(self) # have to checkl collisions
        self.dir = direction
        if self.dir == "left":
            self.move(-1, 0)
        elif self.dir == "right":
            self.move(1, 0)
        elif self.dir == "up":
            self.move(0, -1)
        elif self.dir == "down":
            self.move(0, 1)
        self.velocity = room.update_pacman_pos(self) 

        if self.dotCollision():#did a good thing 
            return 10,False
        if self.velocity == 0:
            self.dir = random_dir()
            return 0,False

        return 1,False #moving is allways good

    # defines how to move to a new position
    def move(self, dx, dy):
        self.x = self.x + dx * self.velocity
        self.y = self.y + dy * self.velocity
        self.rect.x = self.x-self.size + dx * self.velocity
        self.rect.y = self.y-self.size + dy * self.velocity
        self.distance_travelled += abs(dx) + abs(dy)


 

class PacBrain(nn.Module):
    def __init__(self):
        super(PacBrain, self).__init__()
        self.fc1 = nn.Linear(4, 32)
        self.fc2 = nn.Linear(32, 4)
        self.fitness = 0
        self.optimiser = torch.optim.SGD(self.parameters(), lr=0.01)


    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def make_action(self, pacbody,movers, room):
        position = torch.tensor([pacbody.x, pacbody.y, pacbody.velocity, pacbody.energy], dtype=torch.float32)
        output = self(position.unsqueeze(0))
        direction_index = torch.argmax(output).item()
        directions = ["left", "right", "down", "up"]
        direction = directions[direction_index]
        # direction ='down'
        #take action and check for collisions with ghosts or wall
        reward,dead =  pacbody.take_action(direction,movers,room)
                # print(reward)
        self.fitness += reward
        return dead

     
    def update_pac_brain(self, fitness):
        self.fitness = fitness
        self.optimiser.zero_grad()
        loss = -self.fitness
        loss.backward()
        self.optimiser.step()
       
    def mutate(self, mutation_rate):
        for param in self.parameters():
            if random.random() < mutation_rate:
                param.data += torch.randn_like(param.data)

  


