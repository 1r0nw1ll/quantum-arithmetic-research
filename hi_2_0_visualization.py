#!/usr/bin/env python3
"""
HI 2.0 Ablation Study Visualization Suite

Creates publication-ready figures showing:
1. HI 1.0 vs HI 2.0 performance comparison
2. Gender-based discrimination (primitive vs female separation)
3. Domain-specific performance heatmaps
4. 3D component space visualization
5. Weight configuration analysis

Reference: HI 2.0 Implementation & Ablation Study
Date: December 10, 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import json

# Set publication-quality defaults
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
sns.set_palette("husl")

# ============================================================================
# LOAD DATA
# ============================================================================

print("="*100)
print("HI 2.0 ABLATION STUDY - VISUALIZATION SUITE")
print("="*100)
print()

# Load ablation results
print("Loading ablation study results...")
df = pd.read_csv('hi_2_0_ablation_results.csv')
print(f"✓ Loaded {len(df)} experimental combinations")
print()

# Load summary statistics
with open('hi_2_0_ablation_summary.json', 'r') as f:
    summary = json.load(f)

# ============================================================================
# FIGURE 1: HI 1.0 vs HI 2.0 Overall Performance Comparison
# ============================================================================

print("Generating Figure 1: HI 1.0 vs HI 2.0 Performance Comparison...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('HI 2.0 Ablation Study: Performance Comparison', fontsize=16, fontweight='bold')

# Plot 1: Overall HI distribution by configuration
ax1 = axes[0, 0]
config_order = df.groupby('config')['HI_2.0'].mean().sort_values(ascending=False).index
sns.boxplot(data=df, x='config', y='HI_2.0', order=config_order, ax=ax1)
ax1.axhline(y=df[df['config']=='HI_1.0_baseline']['HI_2.0'].mean(),
            color='red', linestyle='--', label='HI 1.0 baseline', linewidth=2)
ax1.set_xlabel('Weight Configuration')
ax1.set_ylabel('HI 2.0 Score')
ax1.set_title('HI 2.0 Distribution by Configuration')
ax1.tick_params(axis='x', rotation=45)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Improvement over HI 1.0
ax2 = axes[0, 1]
improvement_by_config = df.groupby('config')['improvement_pct'].mean().sort_values(ascending=False)
colors = ['green' if x > 0 else 'red' for x in improvement_by_config.values]
improvement_by_config.plot(kind='barh', ax=ax2, color=colors)
ax2.axvline(x=0, color='black', linestyle='-', linewidth=1)
ax2.set_xlabel('Average Improvement vs HI 1.0 (%)')
ax2.set_title('Relative Performance vs HI 1.0 Baseline')
ax2.grid(True, alpha=0.3, axis='x')

# Plot 3: Component contributions
ax3 = axes[1, 0]
# Get balanced config as example
balanced = df[df['config'] == 'Balanced_default']
component_means = balanced[['H_angular', 'H_radial', 'H_family']].mean()
component_means.plot(kind='bar', ax=ax3, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
ax3.set_ylabel('Average Score')
ax3.set_title('Component Contributions (Balanced Config)')
ax3.set_xticklabels(['Angular\n(Pisano)', 'Radial\n(Primitivity)', 'Family\n(Subfamilies)'], rotation=0)
ax3.grid(True, alpha=0.3, axis='y')

# Plot 4: Top performers per tuple
ax4 = axes[1, 1]
top_tuples = df.groupby('tuple_label')['HI_2.0'].max().sort_values(ascending=False).head(10)
top_tuples.plot(kind='barh', ax=ax4, color='steelblue')
ax4.set_xlabel('Maximum HI 2.0 Score')
ax4.set_title('Top 10 Tuples by Best HI 2.0')
ax4.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('hi_2_0_figure1_performance_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: hi_2_0_figure1_performance_comparison.png")
plt.close()

# ============================================================================
# FIGURE 2: Gender-Based Discrimination Analysis
# ============================================================================

print("Generating Figure 2: Gender-Based Discrimination...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Gender-Based Discrimination: Primitive vs Female vs Composite', fontsize=16, fontweight='bold')

# Plot 1: HI 2.0 by gender and config
ax1 = axes[0, 0]
gender_config = df.groupby(['gender', 'config'])['HI_2.0'].mean().reset_index()
gender_pivot = gender_config.pivot(index='config', columns='gender', values='HI_2.0')
gender_pivot.plot(kind='bar', ax=ax1, width=0.8)
ax1.set_ylabel('Average HI 2.0')
ax1.set_title('HI 2.0 by Gender Category and Configuration')
ax1.tick_params(axis='x', rotation=45)
ax1.legend(title='Gender', bbox_to_anchor=(1.05, 1), loc='upper left')
ax1.grid(True, alpha=0.3, axis='y')

# Plot 2: Radial harmonicity by gender (shows primitivity discrimination)
ax2 = axes[0, 1]
gender_radial = df.groupby('gender')['H_radial'].agg(['mean', 'std']).sort_values('mean', ascending=False)
gender_radial['mean'].plot(kind='barh', ax=ax2, xerr=gender_radial['std'],
                            color=['#2ca02c', '#ff7f0e', '#d62728'], capsize=5)
ax2.set_xlabel('Average Radial Harmonicity (1/gcd)')
ax2.set_title('Primitivity Discrimination by Gender')
ax2.grid(True, alpha=0.3, axis='x')

# Plot 3: Separation between primitive and female (key for seismic)
ax3 = axes[1, 0]
primitive_female_sep = []
configs = df['config'].unique()
for config in configs:
    prim = df[(df['config']==config) & (df['gender']=='Male (Primitive)')]['HI_2.0'].mean()
    fem = df[(df['config']==config) & (df['gender']=='Female')]['HI_2.0'].mean()
    primitive_female_sep.append({'config': config, 'separation': prim - fem})

sep_df = pd.DataFrame(primitive_female_sep).sort_values('separation', ascending=False)
sep_df.plot(x='config', y='separation', kind='barh', ax=ax3, color='purple', legend=False)
ax3.set_xlabel('Primitive - Female Separation')
ax3.set_title('Primitive vs Female Discrimination (Higher = Better for Seismic)')
ax3.grid(True, alpha=0.3, axis='x')

# Plot 4: Distribution of HI 2.0 by gender (violin plot)
ax4 = axes[1, 1]
# Use High_radial config (best for discrimination)
high_rad_df = df[df['config'] == 'High_radial']
sns.violinplot(data=high_rad_df, x='gender', y='HI_2.0', ax=ax4)
ax4.set_xlabel('Gender Category')
ax4.set_ylabel('HI 2.0 Score')
ax4.set_title('HI 2.0 Distribution by Gender (High_radial Config)')
ax4.tick_params(axis='x', rotation=15)
ax4.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('hi_2_0_figure2_gender_discrimination.png', dpi=300, bbox_inches='tight')
print("✓ Saved: hi_2_0_figure2_gender_discrimination.png")
plt.close()

# ============================================================================
# FIGURE 3: Domain-Specific Performance Heatmaps
# ============================================================================

print("Generating Figure 3: Domain-Specific Performance Heatmaps...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Domain-Specific Configuration Performance', fontsize=16, fontweight='bold')

# Create feature matrix for each config
config_features = df.groupby('config').agg({
    'H_angular': 'mean',
    'H_radial': 'mean',
    'H_family': 'mean',
    'HI_2.0': 'mean',
    'w_ang': 'first',
    'w_rad': 'first',
    'w_fam': 'first'
}).reset_index()

# Plot 1: Weight configuration heatmap
ax1 = axes[0, 0]
weight_matrix = config_features[['config', 'w_ang', 'w_rad', 'w_fam']].set_index('config')
sns.heatmap(weight_matrix, annot=True, fmt='.2f', cmap='YlOrRd', ax=ax1, cbar_kws={'label': 'Weight'})
ax1.set_title('Weight Configuration Matrix')
ax1.set_ylabel('')

# Plot 2: Component performance heatmap
ax2 = axes[0, 1]
component_matrix = config_features[['config', 'H_angular', 'H_radial', 'H_family']].set_index('config')
sns.heatmap(component_matrix, annot=True, fmt='.3f', cmap='viridis', ax=ax2, cbar_kws={'label': 'Score'})
ax2.set_title('Average Component Scores by Configuration')
ax2.set_ylabel('')

# Plot 3: Performance by tuple family
ax3 = axes[1, 0]
# Extract family info from tuple labels
df['family'] = df['tuple_label'].str.split('_').str[0]
family_config = df.groupby(['family', 'config'])['HI_2.0'].mean().reset_index()
family_pivot = family_config.pivot(index='family', columns='config', values='HI_2.0')
# Select key configs
key_configs = ['HI_1.0_baseline', 'Balanced_default', 'High_angular', 'High_radial', 'High_family']
family_pivot_subset = family_pivot[key_configs]
sns.heatmap(family_pivot_subset, annot=True, fmt='.3f', cmap='coolwarm', ax=ax3, cbar_kws={'label': 'HI 2.0'})
ax3.set_title('HI 2.0 by Tuple Family and Configuration')
ax3.set_xlabel('')

# Plot 4: Config recommendations scatter
ax4 = axes[1, 1]
# Plot configs in weight space (simplified 2D: radial vs family, with angular as size)
scatter = ax4.scatter(config_features['w_rad'], config_features['w_fam'],
                      s=config_features['w_ang']*500,
                      c=config_features['HI_2.0'],
                      cmap='plasma',
                      alpha=0.7,
                      edgecolors='black',
                      linewidths=2)
for idx, row in config_features.iterrows():
    ax4.annotate(row['config'],
                (row['w_rad'], row['w_fam']),
                fontsize=8,
                ha='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
ax4.set_xlabel('Radial Weight (Primitivity)')
ax4.set_ylabel('Family Weight (Subfamilies)')
ax4.set_title('Configuration Space (size = angular weight, color = avg HI 2.0)')
cbar = plt.colorbar(scatter, ax=ax4)
cbar.set_label('Avg HI 2.0')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('hi_2_0_figure3_domain_heatmaps.png', dpi=300, bbox_inches='tight')
print("✓ Saved: hi_2_0_figure3_domain_heatmaps.png")
plt.close()

# ============================================================================
# FIGURE 4: 3D Component Space Visualization
# ============================================================================

print("Generating Figure 4: 3D Component Space Visualization...")

fig = plt.figure(figsize=(16, 12))
fig.suptitle('HI 2.0 Three-Component Space: Angular × Radial × Family', fontsize=16, fontweight='bold')

# Use balanced config for visualization
balanced_df = df[df['config'] == 'Balanced_default'].copy()

# Create color map based on gender
gender_colors = {'Male (Primitive)': '#2ca02c', 'Female': '#ff7f0e', 'Male (Composite)': '#d62728'}
balanced_df['color'] = balanced_df['gender'].map(gender_colors)

# 3D scatter plot
ax1 = fig.add_subplot(221, projection='3d')
scatter = ax1.scatter(balanced_df['H_angular'],
                      balanced_df['H_radial'],
                      balanced_df['H_family'],
                      c=balanced_df['color'],
                      s=balanced_df['HI_2.0']*200,
                      alpha=0.6,
                      edgecolors='black',
                      linewidths=0.5)
ax1.set_xlabel('Angular Harmonicity\n(Pisano Periods)')
ax1.set_ylabel('Radial Harmonicity\n(1/gcd)')
ax1.set_zlabel('Family Harmonicity\n(Subfamilies)')
ax1.set_title('3D Component Space (size = HI 2.0)')

# Create legend
from matplotlib.lines import Line2D
legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor=color,
                          markersize=10, label=gender)
                   for gender, color in gender_colors.items()]
ax1.legend(handles=legend_elements, loc='upper left')

# 2D projections
# Angular vs Radial
ax2 = fig.add_subplot(222)
for gender, color in gender_colors.items():
    subset = balanced_df[balanced_df['gender'] == gender]
    ax2.scatter(subset['H_angular'], subset['H_radial'],
                c=color, label=gender, s=100, alpha=0.6, edgecolors='black')
ax2.set_xlabel('Angular Harmonicity')
ax2.set_ylabel('Radial Harmonicity')
ax2.set_title('Angular × Radial Projection')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Radial vs Family
ax3 = fig.add_subplot(223)
for gender, color in gender_colors.items():
    subset = balanced_df[balanced_df['gender'] == gender]
    ax3.scatter(subset['H_radial'], subset['H_family'],
                c=color, label=gender, s=100, alpha=0.6, edgecolors='black')
ax3.set_xlabel('Radial Harmonicity')
ax3.set_ylabel('Family Harmonicity')
ax3.set_title('Radial × Family Projection')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Angular vs Family
ax4 = fig.add_subplot(224)
for gender, color in gender_colors.items():
    subset = balanced_df[balanced_df['gender'] == gender]
    ax4.scatter(subset['H_angular'], subset['H_family'],
                c=color, label=gender, s=100, alpha=0.6, edgecolors='black')
ax4.set_xlabel('Angular Harmonicity')
ax4.set_ylabel('Family Harmonicity')
ax4.set_title('Angular × Family Projection')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('hi_2_0_figure4_3d_component_space.png', dpi=300, bbox_inches='tight')
print("✓ Saved: hi_2_0_figure4_3d_component_space.png")
plt.close()

# ============================================================================
# FIGURE 5: Detailed Tuple Analysis
# ============================================================================

print("Generating Figure 5: Detailed Tuple Analysis...")

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle('Individual Tuple Performance Across Configurations', fontsize=16, fontweight='bold')

# Plot 1: HI improvement for each tuple (HI 2.0 balanced vs HI 1.0)
ax1 = axes[0, 0]
baseline = df[df['config']=='HI_1.0_baseline'][['tuple_label', 'HI_2.0']].rename(columns={'HI_2.0': 'HI_1.0'})
balanced = df[df['config']=='Balanced_default'][['tuple_label', 'HI_2.0']].rename(columns={'HI_2.0': 'HI_2.0_balanced'})
comparison = baseline.merge(balanced, on='tuple_label')
comparison['improvement'] = comparison['HI_2.0_balanced'] - comparison['HI_1.0']
comparison = comparison.sort_values('improvement', ascending=False)

colors = ['green' if x > 0 else 'red' for x in comparison['improvement'].values]
comparison.plot(x='tuple_label', y='improvement', kind='barh', ax=ax1, color=colors, legend=False)
ax1.axvline(x=0, color='black', linestyle='-', linewidth=1)
ax1.set_xlabel('HI 2.0 - HI 1.0 Improvement')
ax1.set_title('Per-Tuple Improvement with HI 2.0')
ax1.grid(True, alpha=0.3, axis='x')

# Plot 2: Best config for each tuple
ax2 = axes[0, 1]
best_configs = df.loc[df.groupby('tuple_label')['HI_2.0'].idxmax()][['tuple_label', 'config', 'HI_2.0']]
# Count which configs are best most often
config_counts = best_configs['config'].value_counts()
config_counts.plot(kind='bar', ax=ax2, color='steelblue')
ax2.set_xlabel('Configuration')
ax2.set_ylabel('Number of Tuples Where Config is Best')
ax2.set_title('Optimal Configuration Distribution')
ax2.tick_params(axis='x', rotation=45)
ax2.grid(True, alpha=0.3, axis='y')

# Plot 3: HI 2.0 range per tuple across configs
ax3 = axes[1, 0]
tuple_ranges = df.groupby('tuple_label')['HI_2.0'].agg(['min', 'max', 'mean'])
tuple_ranges['range'] = tuple_ranges['max'] - tuple_ranges['min']
tuple_ranges = tuple_ranges.sort_values('range', ascending=False).head(10)

x = range(len(tuple_ranges))
ax3.barh(x, tuple_ranges['range'], color='coral')
ax3.set_yticks(x)
ax3.set_yticklabels(tuple_ranges.index)
ax3.set_xlabel('HI 2.0 Range (max - min across configs)')
ax3.set_title('Top 10 Tuples Most Sensitive to Configuration')
ax3.grid(True, alpha=0.3, axis='x')

# Plot 4: Golden triple special case
ax4 = axes[1, 1]
golden_data = df[df['tuple_label'].str.contains('Fibonacci_primitive_golden')]
golden_pivot = golden_data.pivot_table(index='config', values=['H_angular', 'H_radial', 'H_family', 'HI_2.0'])
golden_pivot[['H_angular', 'H_radial', 'H_family']].plot(kind='bar', ax=ax4, width=0.8)
ax4.set_ylabel('Component Score')
ax4.set_title('Golden Triple (3,4,5): Component Breakdown by Config')
ax4.tick_params(axis='x', rotation=45)
ax4.legend(['Angular', 'Radial', 'Family'])
ax4.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('hi_2_0_figure5_tuple_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Saved: hi_2_0_figure5_tuple_analysis.png")
plt.close()

# ============================================================================
# SUMMARY STATISTICS TABLE
# ============================================================================

print()
print("="*100)
print("VISUALIZATION SUMMARY")
print("="*100)
print()

print("Generated 5 publication-quality figures:")
print("  1. hi_2_0_figure1_performance_comparison.png - Overall performance analysis")
print("  2. hi_2_0_figure2_gender_discrimination.png - Gender-based separation")
print("  3. hi_2_0_figure3_domain_heatmaps.png - Domain-specific recommendations")
print("  4. hi_2_0_figure4_3d_component_space.png - 3D visualization of components")
print("  5. hi_2_0_figure5_tuple_analysis.png - Individual tuple performance")
print()

print("Key Findings:")
print(f"  • Overall HI 1.0 → HI 2.0 improvement: {summary['overall_improvement_pct']:+.2f}%")
print(f"  • Primitive-Female separation: {summary['primitive_female_separation']:.4f} ({summary['primitive_female_separation']/summary['hi_2_0_balanced_mean']*100:.1f}% relative)")
print(f"  • Golden triple (3,4,5) max HI 2.0: {df[df['tuple_label'].str.contains('golden')]['HI_2.0'].max():.4f}")
print()

print("Domain-Specific Best Configurations:")
for domain, config in summary['best_configs_by_domain'].items():
    weights = summary['weight_configurations'][config]
    print(f"  • {domain.upper():12s}: {config:20s} (w_ang={weights[0]:.2f}, w_rad={weights[1]:.2f}, w_fam={weights[2]:.2f})")
print()

print("="*100)
print("VISUALIZATION COMPLETE")
print("="*100)
