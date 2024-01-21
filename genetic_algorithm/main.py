import random
import numpy as np
import matplotlib.pyplot as plt

NUM_CUSTOMERS = 10
POPULATION_SIZE = 100
TOURNAMENT_SELECTION_SIZE = 4
MUTATION_RATE = 0.1
CROSSOVER_RATE = 0.9
MAX_GENERATIONS = 1000

# Generowanie losowych lokalizacji klientów
customer_locations = np.random.rand(NUM_CUSTOMERS, 2)

# Generowanie losowej macierzy kosztów
cost_matrix = np.random.randint(1, 10, size=(NUM_CUSTOMERS, NUM_CUSTOMERS))
np.fill_diagonal(cost_matrix, 0)  # Zapewnienie kosztu zero dla podróży do tego samego klienta
print(cost_matrix)

# Funkcja obliczająca całkowity koszt trasy przy użyciu macierzy kosztów
def calcCost(route):
    total_cost = 0
    for i in range(len(route) - 1):
        total_cost += cost_matrix[route[i], route[i + 1]]
    total_cost += cost_matrix[route[-1], route[0]]
    return total_cost

# Funkcja inicjalizująca populację
def initializePopulation(population_size, num_customers):
    population = []
    for _ in range(population_size):
        route = list(range(num_customers))
        random.shuffle(route)
        population.append([calcCost(route), route])
    return population

def geneticAlgorithm(population, num_customers, tournament_size, mutation_rate, crossover_rate, max_generations):
    gen_number = 0
    for _ in range(max_generations):
        new_population = []

        # Wybór dwóch najlepszych opcji (elityzm)
        new_population.append(sorted(population)[0])
        new_population.append(sorted(population)[1])

        for _ in range(int((len(population) - 2) / 2)):
            # Krzyżowanie
            random_number = random.random()
            if random_number < crossover_rate:
                parent_chromosome1 = sorted(random.choices(population, k=tournament_size))[0]
                parent_chromosome2 = sorted(random.choices(population, k=tournament_size))[0]

                point = random.randint(0, num_customers - 1)

                child_chromosome1 = parent_chromosome1[1][0:point] + [c for c in parent_chromosome2[1] if c not in parent_chromosome1[1][0:point]]
                child_chromosome2 = parent_chromosome2[1][0:point] + [c for c in parent_chromosome1[1] if c not in parent_chromosome2[1][0:point]]

            # Jeśli krzyżowanie się nie dzieje
            else:
                child_chromosome1 = random.choices(population)[0][1]
                child_chromosome2 = random.choices(population)[0][1]

            # Mutacja
            if random.random() < mutation_rate:
                point1 = random.randint(0, num_customers - 1)
                point2 = random.randint(0, num_customers - 1)
                child_chromosome1[point1], child_chromosome1[point2] = child_chromosome1[point2], child_chromosome1[point1]

                point1 = random.randint(0, num_customers - 1)
                point2 = random.randint(0, num_customers - 1)
                child_chromosome2[point1], child_chromosome2[point2] = child_chromosome2[point2], child_chromosome2[point1]

            new_population.append([calcCost(child_chromosome1), child_chromosome1])
            new_population.append([calcCost(child_chromosome2), child_chromosome2])

        population = new_population

        gen_number += 1

        if gen_number % 10 == 0:
            print(f"Generation: {gen_number}, Best Cost: {sorted(population)[0][0]}")

    answer = sorted(population)[0]

    return answer, gen_number

# Wizualizacja optymalnej trasy dostawy
def draw_route_with_numbers(customer_locations, route):
    plt.plot(customer_locations[:, 0], customer_locations[:, 1], "ro")

    for i, (x, y) in enumerate(customer_locations):
        plt.text(x, y + 0.02, str(route[i]), fontsize=8, ha='center', va='bottom')  # Dostosowanie współrzędnej y dla pozycjonowania labelek

    plt.plot(customer_locations[route, 0], customer_locations[route, 1], "gray")
    plt.plot([customer_locations[route[-1], 0], customer_locations[route[0], 0]],
             [customer_locations[route[-1], 1], customer_locations[route[0], 1]], "gray")
    plt.show()

def main():
    initial_population = initializePopulation(POPULATION_SIZE, NUM_CUSTOMERS)
    answer, gen_number = geneticAlgorithm(initial_population, NUM_CUSTOMERS, TOURNAMENT_SELECTION_SIZE, MUTATION_RATE, CROSSOVER_RATE, MAX_GENERATIONS)

    print("\n----------------------------------------------------------------")
    print(f"Generation: {gen_number}")
    print(f"Best chromosome cost: {answer[0]}")
    print("----------------------------------------------------------------\n")

    draw_route_with_numbers(customer_locations, answer[1])
    print(answer[1])

if __name__ == "__main__":
    main()