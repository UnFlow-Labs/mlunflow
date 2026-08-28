import math
import random
import time
import statistics
import matplotlib.pyplot as plt
from unflow.reasoning.detection import LLMIntentDetector
from unflow.reasoning.llm import LlamaLLM
from unflow.core.unflow_core import unflowdecorator


# ============================================================
# PHYSICS SIMULATION
# ============================================================

G = 9.81                 # gravity (m/s^2)
RHO = 1.225              # air density (kg/m^3)
CD = 0.47                # drag coefficient
AREA = 0.01              # cross-sectional area (m^2)
MASS = 1.0               # rocket mass (kg)

DT = 0.02                # simulation timestep


def simulate_rocket(angle_deg, initial_speed):
    """
    Simulate a projectile/rocket with quadratic air drag.

    Returns:
        range_distance, maximum_height
    """

    angle = math.radians(angle_deg)

    # Initial velocity
    vx = initial_speed * math.cos(angle)
    vy = initial_speed * math.sin(angle)

    # Initial position
    x = 0.0
    y = 0.0

    max_height = 0.0

    while y >= 0:

        # Current speed
        speed = math.sqrt(vx**2 + vy**2)

        # Drag force magnitude
        drag = 0.5 * RHO * CD * AREA * speed**2

        # Drag acceleration
        if speed > 0:
            ax_drag = -(drag / MASS) * (vx / speed)
            ay_drag = -(drag / MASS) * (vy / speed)
        else:
            ax_drag = 0
            ay_drag = 0

        # Total acceleration
        ax = ax_drag
        ay = ay_drag - G

        # Update velocity
        vx += ax * DT
        vy += ay * DT

        # Update position
        x += vx * DT
        y += vy * DT

        max_height = max(max_height, y)

        # Safety limit
        if x > 100000:
            break

    return x, max_height

intent_detector = LLMIntentDetector(llm=LlamaLLM(repo_id="Qwen/Qwen2.5-Coder-7B-Instruct-GGUF", 
        filename="qwen2.5-coder-7b-instruct-q8_0.gguf", n_ctx=4096))
@unflowdecorator(intent_detector=intent_detector)
def optimize_rocket():
    """
    Optimize the rocket's launch parameters to maximize distance.
    """
    POPULATION_SIZE = 40
    GENERATIONS = 100

    ANGLE_MIN = 5
    ANGLE_MAX = 85

    SPEED_MIN = 20
    SPEED_MAX = 150

    def random_individual():
        """
        An individual contains:

            [launch angle, initial velocity]
        """

        return [
            random.uniform(ANGLE_MIN, ANGLE_MAX),
            random.uniform(SPEED_MIN, SPEED_MAX)
        ]

    def fitness(individual):
        angle, speed = individual

        distance, max_height = simulate_rocket(angle, speed)

        # Example constraint:
        # maximum allowed altitude = 5000 m
        MAX_ALTITUDE = 5000

        if max_height > MAX_ALTITUDE:

            # Penalize solutions that violate the constraint
            penalty = (max_height - MAX_ALTITUDE) * 10

            return distance - penalty

        return distance

    def tournament_selection(population, tournament_size=3):

        competitors = random.sample(population, tournament_size)

        return max(competitors, key=fitness)


    def crossover(parent1, parent2):

        child = [
            (parent1[0] + parent2[0]) / 2,
            (parent1[1] + parent2[1]) / 2
        ]

        return child


    def mutation(individual, generation):

        child = individual.copy()

        progress = generation / GENERATIONS
        angle_sigma = 8 * (1 - progress) + 1
        speed_sigma = 15 * (1 - progress) + 2
        mutation_rate = 0.1

        # Mutate angle
        if random.random() < mutation_rate:
            child[0] += random.gauss(0, angle_sigma)

        # Mutate speed
        if random.random() < mutation_rate:
            child[1] += random.gauss(0, speed_sigma)

        # Keep values within limits
        child[0] = max(ANGLE_MIN, min(ANGLE_MAX, child[0]))
        child[1] = max(SPEED_MIN, min(SPEED_MAX, child[1]))

        return child

    population = [
            random_individual()
            for _ in range(POPULATION_SIZE)
        ]

    history = []

    for generation in range(GENERATIONS):

        # Evaluate current population
        scores = [fitness(ind) for ind in population]

        best_index = scores.index(max(scores))
        best = population[best_index]

        best_score = scores[best_index]

        history.append(best_score)

        new_population = []

        while len(new_population) < POPULATION_SIZE:

            parent1 = tournament_selection(population)
            parent2 = tournament_selection(population)

            child = crossover(parent1, parent2)

            child = mutation(child)

            new_population.append(child)

        population = new_population

    best = max(population, key=fitness)

    return best, fitness(best), history
    

if __name__ == "__main__":
    best_params, best_distance, history = optimize_rocket()
    print(f"Best Parameters: Angle = {best_params[0]:.2f} degrees, Speed = {best_params[1]:.2f} m/s")
    print(f"Best Distance Achieved: {best_distance:.2f} meters")

    # Plot the fitness history
    plt.plot(history)
    plt.title("Fitness Over Generations")
    plt.xlabel("Generation")
    plt.ylabel("Distance (m)")
    plt.grid()
    plt.show()

