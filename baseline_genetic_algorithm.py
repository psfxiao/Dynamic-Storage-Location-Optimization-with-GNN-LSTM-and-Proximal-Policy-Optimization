"""
Genetic Algorithm Baseline
"""

import sys
import pandas as pd
import numpy as np
import random
import time
from run_single_model import SalesDataLoader, WarehouseDataLoader, WarehouseEnv, COL_SKU_W


class GeneticAlgorithm:
    def __init__(self, env,
                 pop_size=50,
                 generations=100,
                 mutation_rate=0.1,
                 crossover_rate=0.8,
                 tournament_size=3,
                 elite_ratio=0.1):
        self.env = env
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size
        self.elite_count = max(1, int(pop_size * elite_ratio))


        self.valid_actions = env.valid_actions
        self.action_dim = len(self.valid_actions)
        self.chromosome_length = 20

    def initialize_population(self):

        population = []
        for _ in range(self.pop_size):

            chromosome = np.random.choice(self.action_dim,
                                          size=self.chromosome_length,
                                          replace=False).tolist()
            population.append(chromosome)
        return population

    def evaluate_fitness(self, chromosome):

        state, _ = self.env.reset()
        for action_idx in chromosome:

            state, _, _, _, _, _ = self.env.step(action_idx)
        cost = self.env._calculate_total_cost(state)
        return -cost

    def evaluate_population(self, population):
        fitness = []
        for chrom in population:
            fit = self.evaluate_fitness(chrom)
            fitness.append(fit)
        return fitness

    def selection(self, population, fitness):
        selected = []
        for _ in range(self.pop_size):
            tournament_idx = np.random.choice(len(population),
                                              size=self.tournament_size,
                                              replace=False)
            tournament_fitness = [fitness[i] for i in tournament_idx]
            winner_idx = tournament_idx[np.argmax(tournament_fitness)]
            selected.append(population[winner_idx].copy())
        return selected

    def crossover(self, parent1, parent2):

        if random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()
        length = len(parent1)
        pt1 = random.randint(1, length - 2)
        pt2 = random.randint(pt1 + 1, length - 1)
        child1 = parent1[:pt1] + parent2[pt1:pt2] + parent1[pt2:]
        child2 = parent2[:pt1] + parent1[pt1:pt2] + parent2[pt2:]
        return child1, child2

    def mutation(self, chromosome):

        for i in range(len(chromosome)):
            if random.random() < self.mutation_rate:
                chromosome[i] = random.randint(0, self.action_dim - 1)
        return chromosome

    def optimize(self):
        population = self.initialize_population()
        best_individual = None
        best_fitness = -np.inf
        best_cost = np.inf
        best_cost_history = []
        improvement_history = []

        for gen in range(self.generations):
            fitness = self.evaluate_population(population)

            gen_best_idx = np.argmax(fitness)
            gen_best_fitness = fitness[gen_best_idx]
            gen_best_cost = -gen_best_fitness

            if gen_best_fitness > best_fitness:
                best_fitness = gen_best_fitness
                best_individual = population[gen_best_idx].copy()
                best_cost = gen_best_cost

            best_cost_history.append(best_cost)

            if gen % 10 == 0:
                improvement = (self.env.original_cost - best_cost) / self.env.original_cost * 100
                improvement_history.append(improvement)
                print(f"Gen {gen}: Best Cost={best_cost:,.2f}, Imp={improvement:.2f}%")


            elite_indices = np.argsort(fitness)[-self.elite_count:]
            elites = [population[i].copy() for i in elite_indices]


            selected = self.selection(population, fitness)


            new_population = []
            for i in range(0, len(selected), 2):
                if i + 1 < len(selected):
                    child1, child2 = self.crossover(selected[i], selected[i + 1])
                    new_population.extend([child1, child2])
                else:
                    new_population.append(selected[i].copy())


            new_population = [self.mutation(chrom) for chrom in new_population]


            if len(elites) <= len(new_population):
                for i, elite in enumerate(elites):
                    new_population[-(i + 1)] = elite

            population = new_population


        state, _ = self.env.reset()
        for action_idx in best_individual:
            state, _, _, _, _, _ = self.env.step(action_idx)
        final_cost = self.env._calculate_total_cost(state)

        return final_cost, best_individual, best_cost_history, improvement_history


def run_single(seed):

    np.random.seed(seed)
    random.seed(seed)

    print("=" * 60)
    print(f"Genetic Algorithm Baseline (seed={seed})")
    print("=" * 60)

    print("Loading data...")
    sales_loader = SalesDataLoader("order data.xlsx")
    warehouse_loader = WarehouseDataLoader("initial storage location assignment.xlsx", sales_loader)
    env = WarehouseEnv(warehouse_loader, max_steps=350)


    state, _ = env.reset()
    env.original_cost = env._calculate_total_cost(state)
    print(f"Initial cost: {env.original_cost:,.2f}")

    ga = GeneticAlgorithm(env,
                          pop_size=30,
                          generations=50,
                          mutation_rate=0.15,
                          crossover_rate=0.8)

    start_time = time.perf_counter()
    best_cost, best_chromosome, cost_history, improvement_history = ga.optimize()
    elapsed = time.perf_counter() - start_time

    improvement = (env.original_cost - best_cost) / env.original_cost * 100


    patience = max(5, int(ga.generations * 0.1))
    converge_step = ga.generations
    if len(cost_history) > patience:
        best_so_far = cost_history[0]
        no_improve_count = 0
        for i, val in enumerate(cost_history):
            if val < best_so_far:
                best_so_far = val
                no_improve_count = 0
            else:
                no_improve_count += 1
            if no_improve_count >= patience:
                converge_step = i
                break

    print(f"\nResults (seed={seed}):")
    print(f"  Best Cost: {best_cost:,.2f}")
    print(f"  Improvement: {improvement:.2f}%")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Converge Step: {converge_step}")


    results = pd.DataFrame([{
        'Algorithm': 'Genetic Algorithm',
        'Seed': seed,
        'Original_Cost': env.original_cost,
        'Best_Cost': best_cost,
        'Improvement_%': improvement,
        'Time_sec': elapsed,
        'Converge_Step': converge_step,
        'Generations': ga.generations,
        'Population_Size': ga.pop_size
    }])
    results.to_excel(f'baseline_ga_results_seed_{seed}.xlsx', index=False)
    print(f"Results saved to baseline_ga_results_seed_{seed}.xlsx")


    history_df = pd.DataFrame({
        'generation': list(range(len(cost_history))),
        'best_cost': cost_history
    })
    history_df.to_csv(f'ga_history_seed_{seed}.csv', index=False)
    print(f"History saved to ga_history_seed_{seed}.csv")


    max_dist = max(env.distance_mapping.values())
    optimized_df = warehouse_loader.df.copy()
    state, _ = env.reset()
    for action_idx in best_chromosome:
        state, _, _, _, _, _ = env.step(action_idx)
    optimized_df['optimized_distance'] = state[:, 1] * max_dist
    optimized_df['optimized_location'] = optimized_df[COL_SKU_W].map(env.sku_to_position)
    optimized_df.to_excel('optimized_layout_ga.xlsx', index=False)
    print("Optimized layout saved to optimized_layout_ga.xlsx")

    return {
        'seed': seed,
        'best_cost': best_cost,
        'time': elapsed,
        'converge_step': converge_step,
        'cost_history': cost_history
    }


def main():

    if len(sys.argv) > 1:
        seeds = [int(sys.argv[1])]
    else:
        seeds = [42, 123, 456, 789, 1010]
        print(f"No seed specified, will run {len(seeds)} seeds: {seeds}")

    all_results = []
    for seed in seeds:
        res = run_single(seed)
        all_results.append(res)


    if len(seeds) > 1:
        costs = [r['best_cost'] for r in all_results]
        times = [r['time'] for r in all_results]
        converges = [r['converge_step'] for r in all_results]
        print("\n" + "=" * 60)
        print("Summary across seeds:")
        print(f"  Mean Best Cost: {np.mean(costs):,.2f} ± {np.std(costs):,.2f}")
        print(f"  Mean Time: {np.mean(times):.2f}s")
        print(f"  Mean Converge Step: {np.mean(converges):.0f}")
        print("=" * 60)


if __name__ == "__main__":
    main()