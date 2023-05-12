import torch.nn as nn
import numpy as np
import random
import pygame
import sys
from pacman import *  # for pacman and the room

"""
IDEAS
1. WALL COLLISIONS - FSM (hardcoded)
2. SENSORS - "Look" 2/3 blocks ahead in all directions and get a list/tensor of objects 
   in them: [object_up, object_down, object_left, object_right]. Can use top-left corner of ghost to find top-left corner of pacman_rect.
3. Values for each object: 1 for an empty space, 5 for pacman
4. In PygameGhost control(): sensor function and call update to do a forward pass using the object tensor as input and PygameGhost.update() to optimal direction
5. Compute fitness in generation loop as soon as the ghost has done a forward pass. Possibly do 4 in this loop as well? Use fitness
   to update_weights/network(). -- how?
6. Need my own neural net to do this.
"""


class GhostBrain(nn.Module):
    def __init__(self):
        super(GhostBrain, self).__init__()
        self.fc1 = nn.Linear(4, 8)
        self.fc2 = nn.Linear(8, 4)
        self.relu = nn.ReLU()
        self.optimiser = torch.optim.SGD(self.parameters(), lr=0.01)
        self.ideal_fitness = 5

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

    def update_network(self, fitness):
        self.optimiser.zero_grad()
        fitness = torch.tensor(
            fitness, dtype=torch.float32, requires_grad=True)
        loss = loss_func(fitness, self.ideal_fitness)
        loss.backward()
        self.optimiser.step()

# Custom MAE loss function
# predicted: calculated fitness, target: ideal_fitness


class MAELoss(nn.Module):
    def __init__(self):
        super(MAELoss, self).__init__()

    def forward(self, predicted, target):
        return torch.mean(torch.abs(predicted - target))


loss_func = MAELoss()


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


class PygameGhost:
    def __init__(self, x, y, color, brain):
        # Initialize ghost attributes
        self.x = x
        self.y = y
        self.dir = random_dir()
        self.size = MOVER_SIZE
        self.distance_travelled = 0
        self.distance_from_pacman = float("inf")
        self.distance_from_walls = []
        self.color = color
        self.velocity = 0.3
        self.rect = pygame.Rect(
            self.x-self.size, self.y-self.size, self.size*2, self.size*2)
        self.num_collisions = 0
        self.collisions_with_pacman = 0
        self.neural_network = brain

    def get_distance_from_walls(self, walls):
        for wall in walls:
            self.distance_from_walls.append(math.sqrt(
                (self.x - wall[0]) ** 2 + (self.y - wall[1]) ** 2))

    def get_collisions_with_pacman(self):
        return self.collisions_with_pacman

    def get_collisions(self):
        return self.num_collisions

    def get_distance_from_pacman(self):
        pacman_x, pacman_y = pacman.get_position()
        self.distance_from_pacman = math.sqrt(
            ((self.x-TILE_SIZE) - pacman_x) ** 2 + ((self.y-TILE_SIZE) - pacman_y) ** 2)
        return self.distance_from_pacman

    def draw(self, screen):
        pygame.draw.rect(
            screen, self.color, self.rect)

    def detect_collision(self, collider):
        coll = False
        for wall in walls:
            if self.rect.colliderect(wall):
                coll = True
                self.num_collisions += 1
        if self is not collider and self.rect.colliderect(collider.rect):
            coll = True
            self.num_collisions += 1
            if isinstance(collider, PacMan):
                self.collisions_with_pacman += 1
        if coll:  # move 90 degrees
            if self.dir == "left":
                self.dir = "up"
            elif self.dir == "right":
                self.dir = "down"
            elif self.dir == "up":
                self.dir = "right"
            elif self.dir == "down":
                self.dir = "left"
        return coll

    # controls movement of the ghosts
    def control(self, movers):

        for mover in movers:
            self.detect_collision(mover)

        self.sensor()

        if self.dir == "left":
            self.move(-1, 0)
        elif self.dir == "right":
            self.move(1, 0)
        elif self.dir == "up":
            self.move(0, -1)
        elif self.dir == "down":
            self.move(0, 1)

    def get_num_collisions(self):
        return self.num_collisions

    def sensor(self):
        # 5 if its pacman, 1 if it's an empty space, 0 if it's a wall or another ghost
        # [object_up, object_down, object_left, object_right]
        objects_around = [1, 1, 1, 1]

        # for pacman detection
        if (pacman.rect.x, pacman.rect.y) == (self.x, self.y-TILE_SIZE):  # check up
            objects_around[0] = 5
        elif (pacman.rect.x, pacman.rect.y) == (self.x, self.y+TILE_SIZE):  # check  down
            objects_around[1] = 5
        elif (pacman.rect.x, pacman.rect.y) == (self.x-TILE_SIZE, self.y):  # check left
            objects_around[2] = 5
        elif (pacman.rect.x, pacman.rect.y) == (self.x+TILE_SIZE, self.y):  # check right
            objects_around[3] = 5

        # for walls
        for wall in walls:
            if (wall[0], wall[1]) == (self.x, self.y-TILE_SIZE):  # check up
                objects_around[0] = -0
            elif (wall[0], wall[1]) == (self.x, self.y+TILE_SIZE):  # check  down
                objects_around[1] = 0
            elif (wall[0], wall[1]) == (self.x-TILE_SIZE, self.y):  # check left
                objects_around[2] = 0
            elif (wall[0], wall[1]) == (self.x+TILE_SIZE, self.y):  # check right
                objects_around[3] = 0

         # for ghosts
        for ghost in ghosts:
            if (ghost.rect.x, ghost.rect.y) == (self.x, self.y-TILE_SIZE):  # check up
                objects_around[0] = 0
            elif (ghost.rect.x, ghost.rect.y) == (self.x, self.y+TILE_SIZE):  # check  down
                objects_around[1] = 0
            elif (ghost.rect.x, ghost.rect.y) == (self.x-TILE_SIZE, self.y):  # check left
                objects_around[2] = 0
            elif (ghost.rect.x, ghost.rect.y) == (self.x+TILE_SIZE, self.y):  # check right
                objects_around[3] = 0

        return objects_around

    def forward(self):
        inputs = self.sensor()  # get sensor inputs

        inputs = torch.tensor(inputs, dtype=torch.float32)

        # Use the neural network to predict the action probabilities
        action_probs = self.neural_network(
            inputs.unsqueeze(0))

        # Determine the index of the highest probability
        max_prob_index = torch.argmax(action_probs).item()
        directions = ["up", "down", "right", "left"]

        optimal_dir = directions[max_prob_index]

        return optimal_dir

    def update(self, optimal_dir):
        self.dir = optimal_dir

    def move(self, dx, dy):
        self.x = self.x + dx * self.velocity
        self.y = self.y + dy * self.velocity
        self.rect.x = self.x-self.size + dx * self.velocity
        self.rect.y = self.y-self.size + dy * self.velocity
        self.distance_travelled += abs(dx) + abs(dy)

# Define the Ghost Chromosome


class GhostChromosome:
    def __init__(self, action, pygame_ghost):
        self.ghost = pygame_ghost
        self.action = action
        self.fitness = 0
        self.ghost = pygame_ghost

    def compute_fitness(self):
        # Compute fitness based on the ghost's attributes
        dist = self.ghost.get_distance_from_pacman()
        times_hit_pacman = self.ghost.get_collisions_with_pacman()
        times_hit_wall = self.ghost.get_collisions()
        fitness = 1 / (dist // TILE_SIZE + 1) + \
            times_hit_pacman - times_hit_wall

        if dist < 3 * TILE_SIZE:
            fitness += 1

        self.fitness = fitness

        return fitness

    def crossover(self, partner, crossover_prob):
        # Perform crossover between two parent chromosomes to create offspring
        if random.random() < crossover_prob:
            offspring_action = [self.action[0]]
        else:
            offspring_action = [partner.action[0]]
        return GhostChromosome(offspring_action, self.ghost)

    def mutate(self, mutation_rate):
        # Perform mutation on the chromosome's action
        mutated_action = self.action.copy()
        for i in range(len(mutated_action)):
            if random.random() < mutation_rate:
                mutated_action[i] = random.choice(
                    ["up", "down", "left", "right"])
        return GhostChromosome(mutated_action, self.ghost)


# Define constants
POPULATION_SIZE = 4
mutation_rate = 0.3
max_gen = 50

colours = [PINK, ORANGE, GREEN, RED]
# Initialize the ghosts
ghosts = []
for _ in range(POPULATION_SIZE):
    pos = random_pos()
    colour = random.choice(colours)
    pygame_ghost = PygameGhost(pos[0], pos[1], colour, GhostBrain())
    colours.remove(colour)
    ghosts.append(pygame_ghost)

# Main genetic algorithm loop
POPULATION_SIZE = 4
mutation_rate = 0.1
num_generations = 10

movers = ghosts + [pacman]


# Initialize the population
population = []

for i in range(POPULATION_SIZE):
    # Example: chromosome length of 10
    action = [random.choice(["up", "down", "left", "right"])]
    chromosome = GhostChromosome(action, ghosts[i])
    population.append(chromosome)


# Main game loop
running = True
i = 0
generation = 0
while running:

    # draw Environment, PacMan, and ghosts
    screen.fill(BLACK)
    room.draw_map(screen)
    pacman.draw(screen)
    for ghost in ghosts:
        ghost.draw(screen)

    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            sys.exit()
        elif event.type == pygame.KEYDOWN:
            pacman.velocity = 0.3
            if event.key == pygame.K_UP:
                pacman.dir = "up"
            elif event.key == pygame.K_DOWN:
                pacman.dir = "down"
            elif event.key == pygame.K_LEFT:
                pacman.dir = "left"
            elif event.key == pygame.K_RIGHT:
                pacman.dir = "right"

    pacman.control(movers)

    # Update the ghosts with the best weights
    for ghost in ghosts:
        ghost.control(movers)
        if i % 131 == 0:
            generation += 1
            for i, chromosome in enumerate(population):
                optimal_dir = chromosome.ghost.forward()
                chromosome.ghost.update(optimal_dir)
                chromosome.ghost.neural_network.update_network(
                    chromosome.compute_fitness())

            # Compute average fitness
            total_fitness = sum(
                chromosome.fitness for chromosome in population)
            average_fitness = total_fitness / len(population)

            # Find the top fitness
            top_fitness = max(chromosome.fitness for chromosome in population)

            # # Print the average fitness and top fitness
            # print("Generation:", generation)
            # print("Average Fitness:", average_fitness)
            # print("Top Fitness:", top_fitness)

            # Select parents for reproduction
            parents = sorted(
                population, key=lambda x: x.fitness, reverse=True)[:2]

            # Create offspring through crossover and mutation
            offspring = []
            crossover_prob = random.random()
            for _ in range(POPULATION_SIZE // 2):
                parent1, parent2 = random.sample(parents, 2)
                child1 = parent1.crossover(parent2, crossover_prob)
                child2 = parent2.crossover(parent1, crossover_prob)
                child1 = child1.mutate(mutation_rate)
                child2 = child2.mutate(mutation_rate)
                offspring.extend([child1, child2])

            # resetting the collisions_with_pacman for each PygameGhost
            for child in offspring:
                child.ghost.collisions_with_pacman = 0

            # Replace the population with the offspring
            population = offspring

            # Select the best chromosome from the final population
            best_chromosome = max(population, key=lambda x: x.fitness)

            ghost.update(best_chromosome.action)
    i += 1

    # Update the display
    pygame.display.update()
