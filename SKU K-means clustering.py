"""
K-means + Relocation Baseline Evaluation
------------------------------------------
This script implements a complete baseline pipeline:
    1. K-means clustering on SKU outbound features
    2. Cluster-to-location-block assignment (high-frequency clusters -> near I/O)
    3. Within-cluster sorting by outbound quantity
    4. Efficiency evaluation using the SAME formula as the main GNN-LSTM-PPO model

Output: real (not estimated) picking improvement (%) relative to Initial Allocation.
"""

import pandas as pd
import numpy as np
import math
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ==================== CONFIG ====================
# Adjust these paths to match your files
SKU_FEATURE_FILE = r'SKU outbound times and total outbound quantity.xlsx'
WAREHOUSE_FILE   = r'initial storage location assignment.xlsx'
OUTPUT_FILE      = r'Kmeans_relocation_evaluation.xlsx'

# Column names (match your main model)
COL_SKU_W          = 'SKU'                                      # SKU code in warehouse file
COL_LOCATION       = 'storage location code'                    # location code
COL_DISTANCE       = "storage location's Manhattan distance"    # Manhattan distance
COL_SPRINGSUMMER   = 'Spring-Summer Rank'                       # used for rank weight
COL_SKU_FEAT       = 'SKU'                                      # SKU code in feature file
COL_OUTBOUND_TIMES = 'outbound times'
COL_OUTBOUND_QTY   = 'total outbound quantity'

# Hot-SKU rule: same as main model (top 20% by Spring-Summer Rank)
TOP_K_HOT_RATIO = 0.2
HOT_WEIGHT      = 1.5
NORMAL_WEIGHT   = 1.0

# K-means settings
K_RANGE     = range(2, 11)
RANDOM_SEED = 42

# ==================== 1. Load data ====================
def load_data():
    sku_feat = pd.read_excel(SKU_FEATURE_FILE)
    wh       = pd.read_excel(WAREHOUSE_FILE)

    # Check required columns
    for col in [COL_SKU_FEAT, COL_OUTBOUND_TIMES, COL_OUTBOUND_QTY]:
        if col not in sku_feat.columns:
            raise KeyError(f"Missing column in SKU feature file: {col}")
    for col in [COL_SKU_W, COL_LOCATION, COL_DISTANCE, COL_SPRINGSUMMER]:
        if col not in wh.columns:
            raise KeyError(f"Missing column in warehouse file: {col}")

    print(f"Loaded {len(sku_feat)} SKUs with features, {len(wh)} storage locations.")
    return sku_feat, wh


# ==================== 2. K-means clustering ====================
def run_kmeans(sku_feat):
    features = sku_feat[[COL_OUTBOUND_TIMES, COL_OUTBOUND_QTY]].values
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Pick optimal K by silhouette
    from sklearn.metrics import silhouette_score
    best_k, best_score = None, -1
    for k in K_RANGE:
        km = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
        labels = km.fit_predict(features_scaled)
        score = silhouette_score(features_scaled, labels)
        if score > best_score:
            best_score, best_k = score, k

    print(f"Optimal K = {best_k}, Silhouette = {best_score:.3f}")

    km_final = KMeans(n_clusters=best_k, random_state=RANDOM_SEED, n_init=10)
    sku_feat = sku_feat.copy()
    sku_feat['cluster'] = km_final.fit_predict(features_scaled)

    cluster_centers = scaler.inverse_transform(km_final.cluster_centers_)
    cluster_summary = pd.DataFrame({
        'cluster': range(best_k),
        'avg_outbound_times': cluster_centers[:, 0],
        'avg_outbound_qty':   cluster_centers[:, 1],
        'sku_count':          [ (sku_feat['cluster'] == k).sum() for k in range(best_k) ]
    })
    print("\nCluster summary:")
    print(cluster_summary.to_string(index=False))
    return sku_feat, cluster_summary


# ==================== 3. Cluster-to-location assignment ====================
def assign_clusters_to_locations(sku_feat, wh):
    """
    Assign SKUs to storage locations using the K-means clustering:
        - Clusters are sorted by average outbound frequency (high first)
        - Storage locations are sorted by Manhattan distance (near first)
        - The highest-frequency cluster is assigned to the nearest block,
          the next cluster to the next block, and so on.
        - Within each cluster, SKUs are sorted by total outbound quantity.
    Returns:
        sku_to_location_new : dict {SKU -> new location code}
    """
    # Sort clusters by average outbound times (descending)
    cluster_order = sku_feat.groupby('cluster')[COL_OUTBOUND_TIMES].mean() \
                            .sort_values(ascending=False)
    sorted_clusters = cluster_order.index.tolist()

    # Sort locations by distance (ascending)
    wh_sorted = wh.sort_values(COL_DISTANCE)
    sorted_locations = wh_sorted[COL_LOCATION].tolist()

    # Assign cluster by cluster
    sku_to_location_new = {}
    loc_cursor = 0
    for c in sorted_clusters:
        skus_in_c = sku_feat[sku_feat['cluster'] == c] \
                        .sort_values(COL_OUTBOUND_QTY, ascending=False)[COL_SKU_FEAT] \
                        .tolist()
        for sku in skus_in_c:
            if loc_cursor >= len(sorted_locations):
                # Not enough locations; skip (should not happen if counts match)
                break
            sku_to_location_new[sku] = sorted_locations[loc_cursor]
            loc_cursor += 1
    return sku_to_location_new, sorted_clusters


# ==================== 4. Efficiency evaluation ====================
def compute_efficiency(wh, sku_to_location, hot_skus):
    """
    Identical to WarehouseEnv._calculate_overall_efficiency:
        total = sum over SKUs of  weight(sku) / sqrt(distance(sku))
        weight = (HOT_WEIGHT if hot else NORMAL_WEIGHT) / sqrt(rank)
        efficiency = total / N * 10
    """
    distance_map = dict(zip(wh[COL_LOCATION], wh[COL_DISTANCE]))
    rank_map     = dict(zip(wh[COL_SKU_W],  wh[COL_SPRINGSUMMER]))

    total = 0.0
    N = len(wh)
    for _, row in wh.iterrows():
        sku = row[COL_SKU_W]
        loc = sku_to_location.get(sku)
        if loc is None:
            continue
        dist = distance_map[loc]
        rank = rank_map[sku]
        weight = (HOT_WEIGHT if sku in hot_skus else NORMAL_WEIGHT) / math.sqrt(max(1, rank))
        total += weight / math.sqrt(max(1, dist))
    return total / N * 10


def compute_hot_skus(wh):
    """Same rule as main model: top 20% by Spring-Summer Rank."""
    top_k = int(TOP_K_HOT_RATIO * len(wh))
    hot_skus = set(wh.nsmallest(top_k, COL_SPRINGSUMMER)[COL_SKU_W].tolist())
    return hot_skus


# ==================== 5. Main ====================
def main():
    sku_feat, wh = load_data()

    # ---- K-means ----
    sku_feat, cluster_summary = run_kmeans(sku_feat)

    # ---- Current (initial) assignment ----
    sku_to_location_initial = dict(zip(wh[COL_SKU_W], wh[COL_LOCATION]))
    hot_skus = compute_hot_skus(wh)

    eff_initial = compute_efficiency(wh, sku_to_location_initial, hot_skus)
    print(f"\nInitial efficiency: {eff_initial:.4f}")

    # ---- K-means + relocation ----
    sku_to_location_new, sorted_clusters = assign_clusters_to_locations(sku_feat, wh)
    eff_new = compute_efficiency(wh, sku_to_location_new, hot_skus)
    print(f"K-means + relocation efficiency: {eff_new:.4f}")

    improvement_pct = (eff_new - eff_initial) / eff_initial * 100
    print(f"\n>>> Real K-means + relocation improvement: {improvement_pct:.2f}%")

    # ---- Save results ----
    df_result = wh.copy()
    df_result['cluster']               = df_result[COL_SKU_W].map(
        dict(zip(sku_feat[COL_SKU_FEAT], sku_feat['cluster'])))
    df_result['new_location']          = df_result[COL_SKU_W].map(sku_to_location_new)
    df_result['initial_location']      = df_result[COL_SKU_W].map(sku_to_location_initial)
    df_result['new_distance']          = df_result['new_location'].map(
        dict(zip(wh[COL_LOCATION], wh[COL_DISTANCE])))
    df_result['initial_distance']      = df_result['initial_location'].map(
        dict(zip(wh[COL_LOCATION], wh[COL_DISTANCE])))

    with pd.ExcelWriter(OUTPUT_FILE) as writer:
        df_result.to_excel(writer, sheet_name='SKU_Relocation', index=False)
        cluster_summary.to_excel(writer, sheet_name='Cluster_Summary', index=False)
        pd.DataFrame([
            ['Initial efficiency',                  eff_initial],
            ['K-means + relocation efficiency',     eff_new],
            ['Real improvement (%)',                improvement_pct],
            ['Number of clusters (K)',              len(sorted_clusters)],
        ], columns=['Metric', 'Value']).to_excel(writer, sheet_name='Evaluation', index=False)

    print(f"\nResults saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()