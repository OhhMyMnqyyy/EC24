import csv
import streamlit as st
import random
import pandas as pd

# Function to read the CSV file and convert it to the desired format
def read_csv_to_dict(file_path):
    program_ratings = {}
    with open(file_path, mode='r', newline='') as file:
        reader = csv.reader(file)
        # Skip the header
        header = next(reader)
        for row in reader:
            program = row[0]
            ratings = [float(x) for x in row[1:]]  # Convert the ratings to floats
            program_ratings[program] = ratings
    return program_ratings

# Path to the CSV file
file_path = 'pages/program_ratings.csv'

# Get the data in the required format
program_ratings_dict = read_csv_to_dict(file_path)

# Parameters and dataset
GEN = 100
POP = 50
EL_S = 2

all_programs = list(program_ratings_dict.keys())
all_time_slots = list(range(6, 24))  # Time slots from 6 AM to 11 PM

# Streamlit App
st.title("TV Program Scheduling Optimization")

# User input for mutation rate and crossover rate
crossover_rate = st.slider("Crossover Rate", min_value=0.0, max_value=0.95, value=0.8, step=0.01)
mutation_rate = st.slider("Mutation Rate", min_value=0.01, max_value=0.05, value=0.02, step=0.01)

# Add a button to calculate the schedule
if st.button("Calculate"):
    # Defining fitness function
    def fitness_function(schedule):
        total_rating = 0
        for time_slot, program in enumerate(schedule):
            total_rating += program_ratings_dict[program][time_slot]
        return total_rating

    # Initializing population (optimized to avoid brute force)
    def initialize_population(programs, time_slots, pop_size):
        population = []
        for _ in range(pop_size):
            random_schedule = random.sample(programs, len(time_slots))
            population.append(random_schedule)
        return population

    # Genetic Algorithm functions
    def mutate(schedule):
        mutation_point = random.randint(0, len(schedule) - 1)
        new_program = random.choice(all_programs)
        schedule[mutation_point] = new_program
        return schedule

    def crossover(schedule1, schedule2):
        crossover_point = random.randint(1, len(schedule1) - 2)
        child1 = schedule1[:crossover_point] + schedule2[crossover_point:]
        child2 = schedule2[:crossover_point] + schedule1[crossover_point:]
        return child1, child2

    def genetic_algorithm(generations, population_size, crossover_rate, mutation_rate, elitism_size):
        # Initialize population
        population = initialize_population(all_programs, all_time_slots, population_size)

        for generation in range(generations):
            # Sort population by fitness
            population.sort(key=lambda schedule: fitness_function(schedule), reverse=True)

            # Elitism: keep top individuals
            new_population = population[:elitism_size]

            # Generate new individuals through crossover and mutation
            while len(new_population) < population_size:
                parent1, parent2 = random.sample(population[:10], 2)
                if random.random() < crossover_rate:
                    child1, child2 = crossover(parent1, parent2)
                else:
                    child1, child2 = parent1[:], parent2[:]

                if random.random() < mutation_rate:
                    child1 = mutate(child1)
                if random.random() < mutation_rate:
                    child2 = mutate(child2)

                new_population.extend([child1, child2])

            # Replace population with new generation
            population = new_population[:population_size]

        # Return the best schedule
        return max(population, key=fitness_function)

    # Run the genetic algorithm
    optimal_schedule = genetic_algorithm(
        generations=GEN,
        population_size=POP,
        crossover_rate=crossover_rate,
        mutation_rate=mutation_rate,
        elitism_size=EL_S
    )

    # Display final schedule
    st.write("### Final Optimal Schedule")
    schedule_df = pd.DataFrame({
        "Time Slot": [f"{time:02d}:00" for time in all_time_slots],
        "Program": optimal_schedule
    })
    st.table(schedule_df)

    # Display total ratings
    st.write("### Total Ratings")
    st.write(f"**{fitness_function(optimal_schedule)}**")
