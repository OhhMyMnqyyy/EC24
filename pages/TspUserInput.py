import matplotlib.pyplot as plt
import random
import numpy as np
import streamlit as st
import seaborn as sns
from itertools import permutations

st.title("Traveling Salesman Problem Solver")

# User inputs for number of cities
num_cities = st.number_input("Enter the number of cities:", min_value=2, max_value=20, value=10)

# Input fields for city names and coordinates
st.subheader("Enter City Names and Coordinates")
cities_names = []
city_coords = {}
for i in range(num_cities):
    city_name = st.text_input(f"City {i + 1} Name", f"City_{i + 1}")
    x_coord = st.slider(f"{city_name} X-coordinate", -20.0, 20.0, random.uniform(-10, 10), key=f"x_{i}")
    y_coord = st.slider(f"{city_name} Y-coordinate", -20.0, 20.0, random.uniform(-10, 10), key=f"y_{i}")
    cities_names.append(city_name)
    city_coords[city_name] = (x_coord, y_coord)

# Parameters for GA
st.subheader("Genetic Algorithm Parameters")
n_population = st.slider("Population Size", min_value=50, max_value=500, value=250, step=50)
crossover_per = st.slider("Crossover Percentage", min_value=0.1, max_value=1.0, value=0.8)
mutation_per = st.slider("Mutation Percentage", min_value=0.0, max_value=1.0, value=0.2)
n_generations = st.slider("Generations", min_value=50, max_value=500, value=200, step=50)

# Pastel color palette
colors = sns.color_palette("pastel", len(cities_names))

# Plot city map
fig, ax = plt.subplots()
for i, (city, (city_x, city_y)) in enumerate(city_coords.items()):
    color = colors[i]
    ax.scatter(city_x, city_y, c=[color], s=1200, zorder=2)
    ax.annotate(city, (city_x, city_y), fontsize=12, ha='center', va='bottom', xytext=(0, -30), textcoords='offset points')

    # Connect cities with opaque lines
    for j, (other_city, (other_x, other_y)) in enumerate(city_coords.items()):
        if i != j:
            ax.plot([city_x, other_x], [city_y, other_y], color='gray', linestyle='-', linewidth=1, alpha=0.1)

fig.set_size_inches(10, 8)
st.pyplot(fig)

# Define functions for Genetic Algorithm
def initial_population(cities_list, n_population=250):
    population_perms = []
    possible_perms = list(permutations(cities_list))
    random_ids = random.sample(range(len(possible_perms)), n_population)
    for i in random_ids:
        population_perms.append(list(possible_perms[i]))
    return population_perms

def dist_two_cities(city_1, city_2):
    city_1_coords = city_coords[city_1]
    city_2_coords = city_coords[city_2]
    return np.sqrt((city_1_coords[0] - city_2_coords[0]) ** 2 + (city_1_coords[1] - city_2_coords[1]) ** 2)

def total_dist_individual(individual):
    total_dist = 0
    for i in range(len(individual)):
        if i == len(individual) - 1:
            total_dist += dist_two_cities(individual[i], individual[0])
        else:
            total_dist += dist_two_cities(individual[i], individual[i+1])
    return total_dist

def fitness_prob(population):
    total_dist_all_individuals = [total_dist_individual(ind) for ind in population]
    max_population_cost = max(total_dist_all_individuals)
    population_fitness = max_population_cost - np.array(total_dist_all_individuals)
    population_fitness_probs = population_fitness / population_fitness.sum()
    return population_fitness_probs

def roulette_wheel(population, fitness_probs):
    cumsum_probs = fitness_probs.cumsum()
    selected_index = len(cumsum_probs[cumsum_probs < np.random.uniform(0, 1)]) - 1
    return population[selected_index]

def crossover(parent_1, parent_2):
    cut = random.randint(1, len(cities_names) - 1)
    offspring_1 = parent_1[:cut] + [city for city in parent_2 if city not in parent_1[:cut]]
    offspring_2 = parent_2[:cut] + [city for city in parent_1 if city not in parent_2[:cut]]
    return offspring_1, offspring_2

def mutation(offspring):
    index_1, index_2 = random.sample(range(len(cities_names)), 2)
    offspring[index_1], offspring[index_2] = offspring[index_2], offspring[index_1]
    return offspring

def run_ga(cities_names, n_population, n_generations, crossover_per, mutation_per):
    population = initial_population(cities_names, n_population)
    best_population = []

    for _ in range(n_generations):
        fitness_probs = fitness_prob(population)
        parents = [roulette_wheel(population, fitness_probs) for _ in range(int(crossover_per * n_population))]

        offspring = []
        for i in range(0, len(parents), 2):
            off_1, off_2 = crossover(parents[i], parents[i + 1])
            if random.random() < mutation_per:
                off_1 = mutation(off_1)
            if random.random() < mutation_per:
                off_2 = mutation(off_2)
            offspring.extend([off_1, off_2])

        population = sorted(parents + offspring, key=total_dist_individual)[:n_population]
        best_population = population[0]

    return best_population

best_route = run_ga(cities_names, n_population, n_generations, crossover_per, mutation_per)
total_distance = total_dist_individual(best_route)

# Plot best route
x_shortest, y_shortest = zip(*(city_coords[city] for city in best_route))
x_shortest, y_shortest = list(x_shortest) + [x_shortest[0]], list(y_shortest) + [y_shortest[0]]

fig, ax = plt.subplots()
ax.plot(x_shortest, y_shortest, '--go', label='Best Route', linewidth=2.5)
for i, city in enumerate(best_route):
    ax.annotate(f"{i+1} - {city}", (x_shortest[i], y_shortest[i]), fontsize=12)
plt.title(f"Shortest Route (Distance: {round(total_distance, 3)})", fontsize=20)
st.pyplot(fig)
