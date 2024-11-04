import streamlit as st
import matplotlib.pyplot as plt
from itertools import permutations
import random
import numpy as np
import seaborn as sns

# Set the page configuration
st.set_page_config(page_title="TSP Solver with Genetic Algorithm", layout="wide")

st.title("Traveling Salesman Problem (TSP) Solver Using Genetic Algorithm")

# Main Section: Input Cities
st.header("Input Cities and Their Coordinates")

with st.form(key='cities_form'):
    city_names = []
    x_coords = []
    y_coords = []
    for i in range(int(n_cities)):
        st.subheader(f"City {i+1}")
        col1, col2, col3 = st.columns(3)
        with col1:
            city_name = st.text_input(f"Name of City {i+1}", value=f"City_{i+1}")
        with col2:
            x = st.number_input(f"X-coordinate of {city_name}", value=0.0)
        with col3:
            y = st.number_input(f"Y-coordinate of {city_name}", value=0.0)
        city_names.append(city_name)
        x_coords.append(x)
        y_coords.append(y)
    submitted = st.form_submit_button("Submit Cities")

if submitted:
    # Create city coordinates dictionary
    city_coords = dict(zip(city_names, zip(x_coords, y_coords)))

    # Pastel Palette
    colors = sns.color_palette("pastel", len(city_names))

    # Optional: Assign icons or use default markers
    # For simplicity, we'll use default markers here
    fig_initial, ax_initial = plt.subplots(figsize=(10, 8))

    # Plot all cities
    for i, (city, (city_x, city_y)) in enumerate(city_coords.items()):
        ax_initial.scatter(city_x, city_y, c=[colors[i]], s=200, zorder=2, label=city)
        ax_initial.annotate(city, (city_x, city_y), fontsize=12, ha='right', va='bottom')

    # Connect all cities with faint lines
    for i, (city1, (x1, y1)) in enumerate(city_coords.items()):
        for j, (city2, (x2, y2)) in enumerate(city_coords.items()):
            if i < j:
                ax_initial.plot([x1, x2], [y1, y2], color='gray', linestyle='-', linewidth=0.5, alpha=0.3)

    ax_initial.set_title("Cities Map", fontsize=16)
    ax_initial.set_xlabel("X-axis")
    ax_initial.set_ylabel("Y-axis")
    ax_initial.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
    st.pyplot(fig_initial)

    # Genetic Algorithm Implementation

    # Genetic Algorithm Parameters
    # n_population, crossover_per, mutation_per, n_generations are taken from user input

    # Define GA functions

    from itertools import permutations

    def initial_population(cities_list, n_population=250):
        population = []
        population = random.sample(list(permutations(cities_list)), n_population)
        return [list(individual) for individual in population]

    def dist_two_cities(city_1, city_2):
        city_1_coords = city_coords[city_1]
        city_2_coords = city_coords[city_2]
        return np.sqrt((city_1_coords[0] - city_2_coords[0])**2 + (city_1_coords[1] - city_2_coords[1])**2)

    def total_dist_individual(individual):
        total_dist = 0
        for i in range(len(individual)):
            total_dist += dist_two_cities(individual[i], individual[(i + 1) % len(individual)])
        return total_dist

    def fitness_prob(population):
        distances = np.array([total_dist_individual(ind) for ind in population])
        max_dist = distances.max()
        fitness = max_dist - distances
        fitness_sum = fitness.sum()
        if fitness_sum == 0:
            return np.ones(len(population)) / len(population)
        return fitness / fitness_sum

    def roulette_wheel(population, fitness_probs):
        return population[np.random.choice(len(population), p=fitness_probs)]

    def crossover(parent1, parent2):
        size = len(parent1)
        start, end = sorted(random.sample(range(size), 2))
        child_p1 = parent1[start:end]
        child_p2 = [item for item in parent2 if item not in child_p1]
        child = child_p2[:start] + child_p1 + child_p2[start:]
        return child

    def mutate(offspring):
        idx1, idx2 = random.sample(range(len(offspring)), 2)
        offspring[idx1], offspring[idx2] = offspring[idx2], offspring[idx1]
        return offspring

    def run_ga(cities_names, n_population, n_generations, crossover_per, mutation_per):
        population = initial_population(cities_names, n_population)
        fitness_probs = fitness_prob(population)

        for generation in range(n_generations):
            new_population = []
            n_offspring = int(crossover_per * n_population)

            # Selection and Crossover
            for _ in range(n_offspring // 2):
                parent1 = roulette_wheel(population, fitness_probs)
                parent2 = roulette_wheel(population, fitness_probs)
                child1 = crossover(parent1, parent2)
                child2 = crossover(parent2, parent1)
                new_population.extend([child1, child2])

            # Mutation
            for i in range(len(new_population)):
                if random.random() < mutation_per:
                    new_population[i] = mutate(new_population[i])

            # Fill the rest of the population
            if len(new_population) < n_population:
                remaining = n_population - len(new_population)
                new_population.extend(random.sample(population, remaining))

            population = new_population
            fitness_probs = fitness_prob(population)

            # Optional: Display progress
            if (generation + 1) % (n_generations // 10) == 0:
                st.write(f"Generation {generation + 1} completed.")

        # Find the best individual
        distances = [total_dist_individual(ind) for ind in population]
        min_distance = min(distances)
        best_index = distances.index(min_distance)
        best_route = population[best_index]

        return best_route, min_distance

    # Run GA
    with st.spinner("Running Genetic Algorithm..."):
        best_route, min_distance = run_ga(city_names, n_population, n_generations, crossover_per, mutation_per)

    st.success("Genetic Algorithm Completed!")

    # Display Results
    st.subheader("Best Route Found")
    st.write(f"**Total Distance:** {round(min_distance, 3)} units")
    st.write("**Route:** " + " → ".join(best_route))

    # Plot the best route
    x_best = [city_coords[city][0] for city in best_route]
    y_best = [city_coords[city][1] for city in best_route]
    # Return to start
    x_best.append(x_best[0])
    y_best.append(y_best[0])

    fig_best, ax_best = plt.subplots(figsize=(10, 8))
    ax_best.plot(x_best, y_best, '--o', color='green', linewidth=2, markersize=8, label='Best Route')

    # Plot all cities
    for i, (city, (x, y)) in enumerate(city_coords.items()):
        ax_best.scatter(x, y, c=[colors[i]], s=200, zorder=3)
        ax_best.annotate(city, (x, y), fontsize=12, ha='right', va='bottom')

    ax_best.set_title("Best TSP Route Found", fontsize=16)
    ax_best.set_xlabel("X-axis")
    ax_best.set_ylabel("Y-axis")
    ax_best.legend()

    st.pyplot(fig_best)

    # Optional: Display Distance Matrix
    with st.expander("View Distance Matrix"):
        distance_matrix = pd.DataFrame(index=city_names, columns=city_names, dtype=float)
        for city1 in city_names:
            for city2 in city_names:
                distance_matrix.loc[city1, city2] = dist_two_cities(city1, city2)
        st.dataframe(distance_matrix.round(3))

