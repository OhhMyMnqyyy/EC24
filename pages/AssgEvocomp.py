import csv
import random
import streamlit as st

# Function to read the CSV file and convert it to the desired format
def read_csv_to_dict(file_path):
    program_ratings = {}
    with open(file_path, mode='r', newline='') as file:
        reader = csv.reader(file)
        header = next(reader)  # Skip the header
        for row in reader:
            program = row[0]
            ratings = [float(x) for x in row[1:]]  # Convert the ratings to floats
            program_ratings[program] = ratings
    return program_ratings

# Upload CSV file
st.title("TV Schedule Optimization")
st.subheader("Upload a CSV file with program ratings")

uploaded_file = st.file_uploader("Upload CSV File", type="csv")

if uploaded_file is not None:
    program_ratings_dict = read_csv_to_dict(uploaded_file)

    # Display the data
    st.write("### Uploaded Ratings Data")
    st.dataframe(program_ratings_dict)

    # Define constants
    GEN = st.slider("Generations", min_value=50, max_value=500, value=100, step=50)
    POP = st.slider("Population Size", min_value=20, max_value=200, value=50, step=10)
    CO_R = st.slider("Crossover Rate", min_value=0.1, max_value=1.0, value=0.8, step=0.1)
    MUT_R = st.slider("Mutation Rate", min_value=0.1, max_value=1.0, value=0.2, step=0.1)
    EL_S = st.slider("Elitism Size", min_value=1, max_value=10, value=2, step=1)

    all_programs = list(program_ratings_dict.keys())  # Programs
    all_time_slots = list(range(6, 24))  # Time slots

    # Fitness function
    def fitness_function(schedule):
        total_rating = 0
        for time_slot, program in enumerate(schedule):
            total_rating += program_ratings_dict[program][time_slot]
        return total_rating

    # Initialize population
    def initialize_pop(programs, time_slots):
        return [[random.choice(programs) for _ in time_slots] for _ in range(POP)]

    # Crossover
    def crossover(schedule1, schedule2):
        crossover_point = random.randint(1, len(schedule1) - 2)
        child1 = schedule1[:crossover_point] + schedule2[crossover_point:]
        child2 = schedule2[:crossover_point] + schedule1[crossover_point:]
        return child1, child2

    # Mutate
    def mutate(schedule):
        mutation_point = random.randint(0, len(schedule) - 1)
        new_program = random.choice(all_programs)
        schedule[mutation_point] = new_program
        return schedule

    # Genetic algorithm
    def genetic_algorithm(generations, population_size, crossover_rate, mutation_rate, elitism_size):
        population = initialize_pop(all_programs, all_time_slots)

        for generation in range(generations):
            new_population = []

            # Elitism
            population.sort(key=lambda schedule: fitness_function(schedule), reverse=True)
            new_population.extend(population[:elitism_size])

            while len(new_population) < population_size:
                parent1, parent2 = random.choices(population, k=2)
                if random.random() < crossover_rate:
                    child1, child2 = crossover(parent1, parent2)
                else:
                    child1, child2 = parent1.copy(), parent2.copy()

                if random.random() < mutation_rate:
                    child1 = mutate(child1)
                if random.random() < mutation_rate:
                    child2 = mutate(child2)

                new_population.extend([child1, child2])

            population = new_population

        return population[0]

    # Generate optimal schedule
    optimal_schedule = genetic_algorithm(GEN, POP, CO_R, MUT_R, EL_S)

    # Display results
    st.write("### Final Optimal Schedule")
    schedule_table = {
        "Time Slot": [f"{slot:02d}:00" for slot in all_time_slots],
        "Program": optimal_schedule,
        "Rating": [program_ratings_dict[program][i] for i, program in enumerate(optimal_schedule)],
    }
    st.dataframe(schedule_table)

    st.write("**Total Ratings:**", fitness_function(optimal_schedule))
