"""
Random Swap Baseline
Randomly select two SKUs and swap their positions for N iterations.
No learning, purely random exploration.
"""

import sys
import pandas as pd
import numpy as np
import random
import time
from run_single_model import SalesDataLoader, WarehouseDataLoader, WarehouseEnv, COL_SKU_W


def random_swap_optimization(env, n_iterations=30000):

    env.reset()
    best_positions = dict(env.sku_to_position)
    best_cost = env._calculate_total_cost()
    current_cost = best_cost
    cost_history = [best_cost]
    accepted_count = 0

    for i in range(n_iterations):
        sku_list = env.data_loader.df[COL_SKU_W].values
        sku1, sku2 = random.sample(list(sku_list), 2)

        env._swap_physical_positions(sku1, sku2)
        new_cost = env._calculate_total_cost()

        if new_cost < current_cost:
            current_cost = new_cost
            accepted_count += 1
            if new_cost < best_cost:
                best_cost = new_cost
                best_positions = dict(env.sku_to_position)
        else:
            env._swap_physical_positions(sku1, sku2)

        cost_history.append(best_cost)
        if i % 1000 == 0:
            improvement = (env.original_cost - best_cost) / env.original_cost * 100
            print(f"Iteration {i}: Best Cost={best_cost:,.2f}, Imp={improvement:.2f}%")

    return best_cost, best_positions, cost_history, accepted_count


def run_single(seed):

    np.random.seed(seed)
    random.seed(seed)

    print("=" * 60)
    print(f"Random Swap Baseline (seed={seed})")
    print("=" * 60)

    print("Loading data...")
    sales_loader = SalesDataLoader("order data.xlsx")
    warehouse_loader = WarehouseDataLoader("initial storage location assignment.xlsx", sales_loader)
    env = WarehouseEnv(warehouse_loader, max_steps=350)

    # 计算初始成本
    state, _ = env.reset()

    print(f"Initial cost: {env.original_cost:,.2f}")

    start_time = time.perf_counter()
    best_cost, best_positions, cost_history, accepted_count = random_swap_optimization(
        env, n_iterations=30000
    )
    elapsed = time.perf_counter() - start_time

    improvement = (env.original_cost - best_cost) / env.original_cost * 100


    patience = int(30000 * 0.01)
    converge_step = len(cost_history)
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
    print(f"  Accepted swaps: {accepted_count}")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Converge Step: {converge_step}")


    results = pd.DataFrame([{
        'Algorithm': 'Random Swap',
        'Seed': seed,
        'Original_Cost': env.original_cost,
        'Best_Cost': best_cost,
        'Improvement_%': improvement,
        'Time_sec': elapsed,
        'Converge_Step': converge_step,
        'Iterations': 30000,
        'Accepted_Swaps': accepted_count
    }])
    results.to_excel(f'baseline_random_results_seed_{seed}.xlsx', index=False)
    print(f"Results saved to baseline_random_results_seed_{seed}.xlsx")


    history_df = pd.DataFrame({
        'iteration': list(range(len(cost_history))),
        'best_cost': cost_history
    })
    history_df.to_csv(f'random_history_seed_{seed}.csv', index=False)
    print(f"History saved to random_history_seed_{seed}.csv")


    max_dist = max(env.distance_mapping.values())
    optimized_df = warehouse_loader.df.copy()
    optimized_df['optimized_location'] = optimized_df[COL_SKU_W].map(best_positions)

    optimized_df['optimized_distance'] = optimized_df['optimized_location'].map(env.distance_mapping)
    optimized_df.to_excel('optimized_layout_random.xlsx', index=False)

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