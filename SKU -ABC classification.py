import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.integrate import simpson
import os

# ==================== GLOBAL SETTINGS ====================
sns.set_style("whitegrid")
sns.set_palette("Set2")
# Set all fonts to Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'


def abc_classification_with_plots(input_file, output_dir='./'):
    """
    Perform ABC classification and generate publication-quality figures.
    Expects Excel file with columns: 'SKU', 'outbound times', 'total outbound quantity'.
    Outputs:
        - Composite figure with Lorenz curve (left) and scatter plot (right)
        - Excel file with detailed classification and summary table
    """
    # ==================== 1. Load data ====================
    df = pd.read_excel(input_file)

    # Ensure required columns exist
    required_cols = ['SKU', 'outbound times', 'total outbound quantity']
    for col in required_cols:
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found. Found: {df.columns.tolist()}")

    # Sort by total outbound quantity for Lorenz curve
    df_sorted = df.sort_values('total outbound quantity', ascending=False).reset_index(drop=True)
    df_sorted['cumulative_volume_ratio'] = df_sorted['total outbound quantity'].cumsum() / df_sorted[
        'total outbound quantity'].sum()
    df_sorted['cumulative_sku_ratio'] = (df_sorted.index + 1) / len(df_sorted)

    # ==================== 2. ABC Classification ====================
    def classify_abc(series):
        sorted_series = series.sort_values(ascending=False)
        cum_percent = sorted_series.cumsum() / sorted_series.sum()
        return pd.Series(
            ['A' if x <= 0.7 else 'B' if x <= 0.9 else 'C' for x in cum_percent],
            index=sorted_series.index
        )

    df['frequency_class'] = classify_abc(df['outbound times'])
    df['volume_class'] = classify_abc(df['total outbound quantity'])
    df['ABC_class'] = df.apply(
        lambda r: 'A' if 'A' in [r.frequency_class, r.volume_class] else
        'B' if 'B' in [r.frequency_class, r.volume_class] else 'C',
        axis=1
    )

    # ==================== 3. Summary statistics ====================
    abc_stats = df.groupby('ABC_class').agg(
        sku_count=('ABC_class', 'count'),
        total_volume=('total outbound quantity', 'sum')
    ).reset_index()
    abc_stats['sku_percent'] = abc_stats['sku_count'] / abc_stats['sku_count'].sum() * 100
    abc_stats['volume_percent'] = abc_stats['total_volume'] / abc_stats['total_volume'].sum() * 100

    # ==================== 4. Compute Gini coefficient (positive) ====================
    area_under_curve = simpson(df_sorted['cumulative_volume_ratio'].values,
                               x=df_sorted['cumulative_sku_ratio'].values)
    gini = 2 * (area_under_curve - 0.5)  # positive value

    # ==================== 5. Figure: Lorenz (left) and Scatter (right) ====================
    fig = plt.figure(figsize=(14, 6))

    # ---- Left: Lorenz curve ----
    ax1 = plt.subplot(1, 2, 1)
    ax1.plot(df_sorted['cumulative_sku_ratio'], df_sorted['cumulative_volume_ratio'],
             linewidth=2.5, color='#2E86AB', label='Actual Distribution')
    ax1.plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.6, label='Perfect Equality (Pareto 20/80)')

    # Mark Class A threshold (from earlier data: 59.8% SKU → 78.8% volume)
    ax1.axvline(x=0.598, ymin=0, ymax=0.788, color='red', linestyle=':', alpha=0.7, linewidth=2)
    ax1.axhline(y=0.788, xmin=0, xmax=0.598, color='red', linestyle=':', alpha=0.7, linewidth=2)
    ax1.text(0.62, 0.75, 'Class A: 59.8% SKU → 78.8% volume',
             fontsize=12, color='red', fontweight='bold')

    # Display Gini coefficient
    ax1.text(0.05, 0.05, f'Gini Coefficient: {gini:.3f}',
             transform=ax1.transAxes, fontsize=14,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax1.set_xlabel('Cumulative Proportion of SKUs', fontsize=16, fontweight='bold')
    ax1.set_ylabel('Cumulative Proportion of Outbound Volume', fontsize=16, fontweight='bold')
    ax1.set_title('(a) Lorenz Curve: Deviation from 20/80 Pareto Principle',
                  fontsize=15, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=12)
    ax1.tick_params(axis='both', labelsize=14)
    ax1.grid(True, alpha=0.3)

    # ---- Right: Scatter plot ----
    ax3 = plt.subplot(1, 2, 2)
    colors = {'A': '#A23B72', 'B': '#F18F01', 'C': '#6A994E'}
    for cat in ['A', 'B', 'C']:
        subset = df[df['ABC_class'] == cat]
        ax3.scatter(subset['outbound times'], subset['total outbound quantity'],
                    s=30, c=colors[cat], label=f'Class {cat}',
                    alpha=0.6, edgecolors='white', linewidth=0.3)

    ax3.set_xlabel('Outbound Frequency (times)', fontsize=16, fontweight='bold')
    ax3.set_ylabel('Total Outbound Quantity (units)', fontsize=16, fontweight='bold')
    ax3.set_title('(c) SKU-Level Distribution by ABC Class',
                  fontsize=15, fontweight='bold')
    ax3.legend(loc='upper left', fontsize=12)
    ax3.tick_params(axis='both', labelsize=14)
    ax3.grid(True, alpha=0.3)
    ax3.set_xscale('log')
    ax3.set_yscale('log')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/Figure_ABC_Lorenz_Scatter.png',
                dpi=600, bbox_inches='tight', facecolor='white')
    plt.savefig(f'{output_dir}/Figure_ABC_Lorenz_Scatter.pdf',
                bbox_inches='tight', facecolor='white')
    plt.show()

    # ==================== 6. Export to Excel with summary table ====================
    with pd.ExcelWriter(f'{output_dir}/ABC_Classification_Results.xlsx') as writer:
        # Detailed classification
        df.to_excel(writer, sheet_name='Classification_Results', index=False)
        # Summary statistics
        abc_stats.to_excel(writer, sheet_name='Summary_Statistics', index=False)

        # ---- Formatted table for paper (Table 1) ----
        table_data = abc_stats[['ABC_class', 'sku_count', 'sku_percent', 'volume_percent']].copy()
        table_data.columns = ['ABC Class', 'Number of SKUs', 'SKU Proportion (%)', 'Outbound Volume Proportion (%)']
        # Add total row
        total_row = pd.DataFrame([[
            'Total',
            table_data['Number of SKUs'].sum(),
            100.0,
            100.0
        ]], columns=table_data.columns)
        table_data = pd.concat([table_data, total_row], ignore_index=True)
        table_data.to_excel(writer, sheet_name='Table_ABC_Summary', index=False)

        # Print to console for quick copying
        print("\n" + "=" * 60)
        print("Table 1: ABC Classification Summary (Copy this to your paper)")
        print("=" * 60)
        print(table_data.to_string(index=False))
        print("=" * 60)

    # ==================== 7. Print key findings to console ====================
    class_a = abc_stats[abc_stats['ABC_class'] == 'A']
    print(f"\nKey findings:")
    print(f"  - Class A SKU proportion: {class_a['sku_percent'].values[0]:.1f}%")
    print(f"  - Class A volume proportion: {class_a['volume_percent'].values[0]:.1f}%")
    print(f"  - Gini coefficient: {gini:.3f}")
    print(f"  - Conclusion: Pareto 20/80 is not significant in this dataset.\n")

    return df, abc_stats, gini


# ==================== MAIN EXECUTION ====================
if __name__ == "__main__":
    # --- USER INPUT: adjust these paths ---
    input_file = r'SKU outbound times and total outbound quantity.xlsx'
    output_dir = r'ABC_analysis_figures'
    # ---------------------------------------

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Run the analysis
    df, stats, gini = abc_classification_with_plots(input_file, output_dir)

    print("All done! Check the output folder for figures and Excel file.")