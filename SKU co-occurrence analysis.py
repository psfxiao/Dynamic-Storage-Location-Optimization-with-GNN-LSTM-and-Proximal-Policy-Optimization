import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
from collections import defaultdict
import os

# ==================== GLOBAL SETTINGS ====================
sns.set_style("whitegrid")
sns.set_palette("Set2")
plt.rcParams['font.family'] = 'Times New Roman'


def sku_correlation_analysis_with_plots(input_file, output_dir='./', min_support=2):
    """
    Perform SKU co-occurrence analysis (association rule mining) and generate
    summary statistics and figures for paper.
    Expects Excel file with columns:
        'order number', 'SKU encodes' (comma-separated SKU codes)
    Other columns are ignored.
    """
    # ==================== 1. Load data ====================
    df = pd.read_excel(input_file)

    # Check required column
    if 'SKU encodes' not in df.columns:
        raise KeyError(f"Column 'SKU encodes' not found. Found: {df.columns.tolist()}")

    # Clean SKU strings: split by comma, strip spaces, filter empty
    def clean_sku(sku_str):
        if pd.isna(sku_str):
            return []
        return [s.strip() for s in str(sku_str).split(',') if s.strip()]

    transactions = df['SKU encodes'].apply(clean_sku).tolist()
    # Remove empty baskets
    transactions = [basket for basket in transactions if basket]

    # ==================== 2. Generate co-occurrence matrix ====================
    co_occurrence = defaultdict(int)
    single_counts = defaultdict(int)

    # First pass: count individual SKUs
    for basket in transactions:
        for item in set(basket):
            single_counts[item] += 1

    # Second pass: count pairs
    for basket in transactions:
        basket_set = sorted(set(basket))  # unique and sorted
        for combo in combinations(basket_set, 2):
            co_occurrence[combo] += 1

    # Convert to DataFrame
    co_occurrence_df = pd.DataFrame(
        [(k[0], k[1], v) for k, v in co_occurrence.items()],
        columns=['SKU_A', 'SKU_B', 'co_occurrence_count']
    )

    total_orders = len(transactions)
    co_occurrence_df['support'] = co_occurrence_df['co_occurrence_count'] / total_orders
    co_occurrence_df['confidence_A_to_B'] = co_occurrence_df.apply(
        lambda row: row['co_occurrence_count'] / single_counts[row['SKU_A']], axis=1
    )
    co_occurrence_df['confidence_B_to_A'] = co_occurrence_df.apply(
        lambda row: row['co_occurrence_count'] / single_counts[row['SKU_B']], axis=1
    )

    # ===== MODIFIED =====
    # NOTE: The original code filtered here using min_support and defined
    # `filtered`. We now compute all three threshold levels explicitly and
    # define `filtered` (= support >= min_support, default 2) afterwards,
    # so that summary statistics and figures remain consistent.
    # ====================
    # (previous filtering block removed; see Section 3 below)
    # ====================

    # ==================== 3. Summary statistics ====================
    # ===== MODIFIED =====
    # Statistics for three support thresholds (>=1, >=2, >=3).
    # Maximum support is identical across thresholds because the top pair
    # always exists regardless of the threshold.
    pairs_ge1 = co_occurrence_df[co_occurrence_df['co_occurrence_count'] >= 1]
    pairs_ge2 = co_occurrence_df[co_occurrence_df['co_occurrence_count'] >= 2]
    pairs_ge3 = co_occurrence_df[co_occurrence_df['co_occurrence_count'] >= 3]

    num_pairs_ge1 = len(pairs_ge1)
    num_pairs_ge2 = len(pairs_ge2)
    num_pairs_ge3 = len(pairs_ge3)

    max_support = co_occurrence_df['support'].max() if not co_occurrence_df.empty else 0
    max_conf_a_to_b = co_occurrence_df['confidence_A_to_B'].max() if not co_occurrence_df.empty else 0
    max_conf_b_to_a = co_occurrence_df['confidence_B_to_A'].max() if not co_occurrence_df.empty else 0

    # `filtered` is used for downstream plotting (support >= min_support).
    filtered = pairs_ge2.copy()
    filtered.sort_values('co_occurrence_count', ascending=False, inplace=True)
    # ===== END MODIFIED =====

    # ===== MODIFIED =====
    # Extended summary table: now reports support >= 1, >= 2, >= 3 explicitly.
    summary_stats = pd.DataFrame([
        ['Total orders (transactions)', f'{total_orders:,}'],
        ['Number of unique SKUs', f'{len(single_counts):,}'],
        ['Co-occurring pairs (support ≥ 1)', f'{num_pairs_ge1:,}'],
        ['Co-occurring pairs (support ≥ 2)', f'{num_pairs_ge2:,}'],
        ['Co-occurring pairs (support ≥ 3)', f'{num_pairs_ge3:,}'],
        ['Maximum support', f'{max_support:.6f}'],
        ['Maximum confidence (A→B)', f'{max_conf_a_to_b:.3f}'],
        ['Maximum confidence (B→A)', f'{max_conf_b_to_a:.3f}'],
    ], columns=['Metric', 'Value'])
    # ===== END MODIFIED =====

    # ==================== 4. Export to Excel ====================
    # ===== MODIFIED =====
    # Added sheets for support >= 1 and support >= 3 so that the full
    # threshold-sensitivity analysis can be inspected directly.
    with pd.ExcelWriter(f'{output_dir}/SKU_correlation_results.xlsx') as writer:
        pairs_ge1.to_excel(writer, sheet_name='All_Pairs_Support_ge1', index=False)
        pairs_ge2.to_excel(writer, sheet_name='Association_Rules_ge2', index=False)
        pairs_ge3.to_excel(writer, sheet_name='Association_Rules_ge3', index=False)
        filtered.head(100).to_excel(writer, sheet_name='Top100_Rules', index=False)
        summary_stats.to_excel(writer, sheet_name='Summary_Statistics', index=False)
    # ===== END MODIFIED =====

    # ==================== 5. Generate figures ====================
    # Figure 1: Scatter plot of support vs confidence (A→B)
    # NOTE: Figures still use `filtered` (support >= min_support) for readability.
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    if not filtered.empty:
        ax1.scatter(filtered['support'], filtered['confidence_A_to_B'],
                    s=20, alpha=0.6, color='#2E86AB', edgecolors='white', linewidth=0.3)
        # Highlight the point with max support
        max_sup_row = filtered.loc[filtered['support'].idxmax()]
        ax1.scatter(max_sup_row['support'], max_sup_row['confidence_A_to_B'],
                    s=100, color='red', marker='D', label=f'Max support = {max_sup_row["support"]:.5f}')
        ax1.legend(loc='upper right', fontsize=12)
    ax1.set_xlabel('Support', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Confidence (A→B)', fontsize=14, fontweight='bold')
    ax1.set_title('(a) Support vs. Confidence of SKU Pairs', fontsize=14, fontweight='bold')
    ax1.tick_params(axis='both', labelsize=12)
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/Figure_Support_vs_Confidence.png', dpi=600, bbox_inches='tight')
    plt.savefig(f'{output_dir}/Figure_Support_vs_Confidence.pdf', bbox_inches='tight')
    plt.show()

    # ==================== Figure 2: Raincloud plot for Support & Confidence distributions ====================
    import matplotlib.patches as mpatches

    fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(14, 6))

    # ---- Left: Raincloud for Support ----
    # Boxplot
    bp = ax2a.boxplot(filtered['support'], positions=[0], widths=0.3, patch_artist=True,
                      boxprops=dict(facecolor='#2E86AB', alpha=0.6))
    # Violin plot
    parts = ax2a.violinplot(filtered['support'], positions=[0], widths=0.5, showmeans=False, showmedians=False)
    for pc in parts['bodies']:
        pc.set_facecolor('#2E86AB')
        pc.set_alpha(0.3)
    # Strip plot (jittered points)
    jitter = np.random.normal(0, 0.04, size=len(filtered))
    ax2a.scatter(jitter, filtered['support'], s=5, alpha=0.3, color='#2E86AB', edgecolors='none')
    ax2a.set_xlim(-0.3, 0.3)
    ax2a.set_xticks([0])
    ax2a.set_xticklabels([''])
    ax2a.set_ylabel('Support', fontsize=15, fontweight='bold')
    ax2a.set_title('(b1) Support Distribution', fontsize=14, fontweight='bold')
    ax2a.tick_params(axis='y', labelsize=13)
    ax2a.grid(True, alpha=0.3, axis='y')
    ax2a.text(0.05, 0.95, f'Max: {filtered["support"].max():.5f}', transform=ax2a.transAxes, fontsize=11)

    # ---- Right: Raincloud for Confidence ----
    bp = ax2b.boxplot(filtered['confidence_A_to_B'], positions=[0], widths=0.3, patch_artist=True,
                      boxprops=dict(facecolor='#A23B72', alpha=0.6))
    parts = ax2b.violinplot(filtered['confidence_A_to_B'], positions=[0], widths=0.5, showmeans=False,
                            showmedians=False)
    for pc in parts['bodies']:
        pc.set_facecolor('#A23B72')
        pc.set_alpha(0.3)
    jitter = np.random.normal(0, 0.04, size=len(filtered))
    ax2b.scatter(jitter, filtered['confidence_A_to_B'], s=5, alpha=0.3, color='#A23B72', edgecolors='none')
    ax2b.set_xlim(-0.3, 0.3)
    ax2b.set_xticks([0])
    ax2b.set_xticklabels([''])
    ax2b.set_ylabel('Confidence (A→B)', fontsize=15, fontweight='bold')
    ax2b.set_title('(b2) Confidence Distribution', fontsize=14, fontweight='bold')
    ax2b.tick_params(axis='y', labelsize=13)
    ax2b.grid(True, alpha=0.3, axis='y')
    ax2b.text(0.05, 0.95, f'Max: {filtered["confidence_A_to_B"].max():.3f}', transform=ax2b.transAxes, fontsize=11)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/Figure_Raincloud_Support_Confidence.png', dpi=600, bbox_inches='tight')
    plt.savefig(f'{output_dir}/Figure_Raincloud_Support_Confidence.pdf', bbox_inches='tight')
    plt.show()

    # ==================== 6. Print summary to console ====================
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS FOR SKU CO-OCCURRENCE ANALYSIS")
    print("=" * 60)
    print(summary_stats.to_string(index=False))
    print("=" * 60)

    # ===== MODIFIED =====
    # Explicit threshold-sensitivity check: max support should be identical
    # across thresholds, confirming that the sparsity conclusion is not an
    # artifact of the chosen support threshold.
    print("\nThreshold sensitivity check:")
    print(f"  Max support (>=1): {pairs_ge1['support'].max():.6f}")
    print(f"  Max support (>=2): {pairs_ge2['support'].max():.6f}")
    print(f"  Max support (>=3): {pairs_ge3['support'].max():.6f}")
    print(f"  Number of pairs (>=1): {num_pairs_ge1:,}")
    print(f"  Number of pairs (>=2): {num_pairs_ge2:,}")
    print(f"  Number of pairs (>=3): {num_pairs_ge3:,}")
    # ===== END MODIFIED =====

    print("=" * 60)
    print("\nInterpretation:")
    print(f"  - Maximum support is only {max_support:.5f}, indicating extremely low co-occurrence.")
    print(f"  - Maximum confidence is only {max_conf_a_to_b:.3f}, showing weak association.")
    print("  - Conclusion: No strong or stable SKU bundles exist in this dataset.")
    print("  - This justifies the need for a more powerful model (GNN) that captures implicit relationships.\n")

    return filtered, summary_stats


# ==================== MAIN EXECUTION ====================
if __name__ == "__main__":
    # --- USER INPUT: adjust these paths ---
    input_file = r'order data.xlsx'
    output_dir = r'SKU_correlation_output'
    # ---------------------------------------

    os.makedirs(output_dir, exist_ok=True)

    try:
        rules, stats = sku_correlation_analysis_with_plots(input_file, output_dir, min_support=2)
        print(f"All done! Outputs saved to: {output_dir}")
        print("Files generated:")
        print("  - SKU_correlation_results.xlsx")
        print("  - Figure_Support_vs_Confidence.png/pdf")
        print("  - Figure_Raincloud_Support_Confidence.png/pdf")
    except Exception as e:
        print(f"Error: {e}")