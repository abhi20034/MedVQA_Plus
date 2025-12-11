#!/usr/bin/env python3
"""
Generate accuracy and entropy comparison figures for IEEE report.
"""
import matplotlib.pyplot as plt
import numpy as np

# Set style for IEEE publication
plt.style.use('seaborn-v0_8-paper')
plt.rcParams.update({
    'font.size': 10,
    'font.family': 'serif',
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 12,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

# Data from the report
models = ['Baseline', 'MedVQA+']
closed_acc = [60.3, 58.7]
top50_acc = [44.1, 54.6]

closed_entropy = [0.055, 0.693]  # Baseline has low entropy, MedVQA+ has high
top50_entropy = [0.055, 0.670]

# Figure 1: Accuracy Comparison
fig, ax = plt.subplots(figsize=(6, 4))
x = np.arange(len(models))
width = 0.35

bars1 = ax.bar(x - width/2, closed_acc, width, label='Closed', color='#2E86AB', alpha=0.8)
bars2 = ax.bar(x + width/2, top50_acc, width, label='Top-50', color='#A23B72', alpha=0.8)

ax.set_ylabel('Accuracy (%)', fontsize=10)
ax.set_xlabel('Model', fontsize=10)
ax.set_title('Accuracy Comparison: Baseline vs MedVQA+', fontsize=11, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
ax.grid(True, alpha=0.3, axis='y', linestyle='--')
ax.set_ylim([0, 70])

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('output.png', dpi=300, bbox_inches='tight')
print(">>> Saved: output.png (Accuracy Comparison)")
plt.close()

# Figure 2: Entropy Comparison
fig, ax = plt.subplots(figsize=(6, 4))
x = np.arange(len(models))
width = 0.35

bars1 = ax.bar(x - width/2, closed_entropy, width, label='Closed', color='#2E86AB', alpha=0.8)
bars2 = ax.bar(x + width/2, top50_entropy, width, label='Top-50', color='#A23B72', alpha=0.8)

# Add max entropy line
max_entropy = np.log(2)  # ~0.693
ax.axhline(max_entropy, color='red', linestyle='--', linewidth=1.5, 
           label=f'Max Entropy ({max_entropy:.3f})', alpha=0.7)

ax.set_ylabel('Gate Entropy', fontsize=10)
ax.set_xlabel('Model', fontsize=10)
ax.set_title('Gate Entropy Comparison: Baseline vs MedVQA+', fontsize=11, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
ax.grid(True, alpha=0.3, axis='y', linestyle='--')
ax.set_ylim([0, 0.75])

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('output_entropy.png', dpi=300, bbox_inches='tight')
print(">>> Saved: output_entropy.png (Entropy Comparison)")
plt.close()

print("\n✅ All figures generated successfully!")
print("   - output.png: Accuracy comparison")
print("   - output_entropy.png: Entropy comparison")
print("   - gate_analysis_test.png: Gate analysis (already exists)")

