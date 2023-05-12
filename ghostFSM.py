import torch.nn as nn
import numpy as np
import random
import pygame
import sys
from pacman import *  # for pacman and the room


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
    def __init__(self, x, y, color):
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
        return coll

    def handle_coll(self, coll):
        if coll:  # move 90 degrees
            if self.dir == "left":
                self.dir = "up"
            elif self.dir == "right":
                self.dir = "down"
            elif self.dir == "up":
                self.dir = "right"
            elif self.dir == "down":
                self.dir = "left"

    # controls movement of the ghosts
    def control(self):
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

    def update(self, optimal_dir):
        self.dir = optimal_dir

    def move(self, dx, dy):
        self.x = self.x + dx * self.velocity
        self.y = self.y + dy * self.velocity
        self.rect.x = self.x-self.size + dx * self.velocity
        self.rect.y = self.y-self.size + dy * self.velocity
        self.distance_travelled += abs(dx) + abs(dy)


class GhostFSM:
    def __init__(self, pygame_ghost):
        self.state = "Chase"  # Initial state
        # Predefined target position for Scatter state
        self.target_position = (0, 0)
        self.ghost = pygame_ghost

    def update(self):
        # Update the FSM based on inputs and transition to the next state

        if self.state == "Chase":
            for mover in movers:
                coll = self.detect_collision(mover)
            if coll:
                self.state = "Bump"

        elif self.state == "Bump":
            for mover in movers:
                coll = self.detect_collision(mover)
            if not coll:
                self.state = "Chase"

    def get_action(self):
        # Determine the action to be performed based on the current state

        if self.state == "Chase":
            if pacman.x > self.ghost.x:

        elif self.state == "Bump":
            self.ghost.handle_coll()


            # Define the mutation rate
mutation_rate = 0.1


def mutate_fsm(fsm):
    # Iterate through the FSM's states, transitions, or actions
    for state in fsm.states:
        # Check if mutation should occur for the current state
        if random.random() < mutation_rate:
            # Perform a mutation on the state
            mutated_state = mutate_state(state)
            # Update the state in the FSM
            fsm.update_state(state, mutated_state)

    for transition in fsm.transitions:
        # Check if mutation should occur for the current transition
        if random.random() < mutation_rate:
            # Perform a mutation on the transition
            mutated_transition = mutate_transition(transition)
            # Update the transition in the FSM
            fsm.update_transition(transition, mutated_transition)

    for action in fsm.actions:
        # Check if mutation should occur for the current action
        if random.random() < mutation_rate:
            # Perform a mutation on the action
            mutated_action = mutate_action(action)
            # Update the action in the FSM
            fsm.update_action(action, mutated_action)


def mutate_state(state):
    # Implement the mutation operation for a state
    # Example: Randomly change the name of the state
    mutated_name = generate_mutated_name(state.name)
    mutated_state = State(mutated_name)
    return mutated_state


def mutate_transition(transition):
    # Implement the mutation operation for a transition
    # Example: Randomly change the target state of the transition
    mutated_target_state = random.choice(transition.valid_states)
    mutated_transition = Transition(transition.input, mutated_target_state)
    return mutated_transition


def mutate_action(action):
    # Implement the mutation operation for an action
    # Example: Randomly change the behavior of the action
    mutated_behavior = generate_mutated_behavior(action.behavior)
    mutated_action = Action(mutated_behavior)
    return mutated_action


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
    pygame_ghost = PygameGhost(pos[0], pos[1], colour)
    colours.remove(colour)
    ghosts.append(pygame_ghost)

movers = ghosts + [pacman]

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
                optimal_dir = chromosome.ghost.brain()
                print(i, optimal_dir)
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

            # # Select the best chromosome from the final population
            # best_chromosome = max(population, key=lambda x: x.fitness)

            # ghost.update(best_chromosome.action)
    i += 1

    # Update the display
    pygame.display.update()
