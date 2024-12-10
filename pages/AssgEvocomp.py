import csv
import streamlit as st
import random
import pandas as pd

# Read CSV Function
def read_csv_to_dict(file):
    program_ratings = {}
    reader = csv.reader(file)
    header = next(reader)
    for row in reader:
        program = row[0]
        ratings = [float(x) for x in row[1:]]
        program_ratings[program] = ratings
    return program_ratings

# Fitness Function
def fitness_function(schedule):
    return sum(program_ratings_dict[program][i] for i, program in enumerate(schedule))

# Genetic Algorithm Functions
def mutate(schedule):
    schedule[random.randint(0, len(schedule) - 1)] = random.choice(all_programs)
    return schedule

def crossover(schedule1, schedule2):
    point = random.randint(1, len(schedule1) - 1)
    return schedule1[:point] + schedule2[point:], schedule2[:point] + schedule1[point:]

def genetic_algorithm(generations, population_size, crossover_rate, mutation_rate, elitism_size):
    population = [[random.choice(all_programs) for _ in all_time_slots] for _ in range(population_size)]
    for _ in range(generations):
        population.sort(key=lambda s: fitness_function(s), reverse=True)
        new_population = population[:elitism_size]
        while len(new_population) < population_size:
            p1, p2 = random.sample(population[:10], 2)
            if random.random() < crossover_rate:
                c1, c2 = crossover(p1, p2)
            else:
                c1, c2 = p1[:], p2[:]
            if random.random() < mutation_rate:
                c1 = mutate(c1)
            if random.random() < mutation_rate:
                c2 = mutate(c2)
            new_population.extend([c1, c2])
        population = new_population[:population_size]
    return max(population, key=fitness_function)

# Streamlit App
st.title("TV Program Scheduling Optimization")

uploaded_file = st.file_uploader("Upload CSV File", type="csv")
if uploaded_file:
    program_ratings_dict = read_csv_to_dict(uploaded_file)
    all_programs = list(program_ratings_dict.keys())
    all_time_slots = list(range(6, 24))  # Time slots from 6 AM to 11 PM

    with st.form("input_form"):
        crossover_rate = st.slider("Crossover Rate", 0.0, 0.95, 0.8, 0.01)
        mutation_rate = st.slider("Mutation Rate", 0.01, 0.05, 0.02, 0.01)
        calculate = st.form_submit_button("Calculate")

    if calculate:
        schedule = genetic_algorithm(GEN, POP, crossover_rate, mutation_rate, EL_S)
        st.write("### Final Optimal Schedule")
        df = pd.DataFrame({"Time Slot": [f"{t}:00" for t in all_time_slots], "Program": schedule})
        st.table(df)
        st.write(f"**Total Ratings: {fitness_function(schedule)}**")
