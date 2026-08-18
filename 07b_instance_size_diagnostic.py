"""
Phase 4 diagnostic: lake instance size distribution
Determines a data-driven minimum size threshold to separate real lakes
from speckle noise, rather than guessing.

Prereqs (already installed): pandas, numpy, matplotlib
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('data/lake_instances/lake_instances_all_dates.csv')

print(f'Total instances: {len(df)}')
print(f'Pixel count range: {df["n_pixels"].min()} to {df["n_pixels"].max()}')
print(f'Median pixel count: {df["n_pixels"].median()}')
print(df['n_pixels'].describe())

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(df['n_pixels'], bins=100, color='steelblue')
axes[0].set_xlabel('Instance size (pixels)')
axes[0].set_ylabel('Count')
axes[0].set_title('Lake instance size distribution (linear)')

# Log scale is essential here — a few real lakes are huge (1000s of pixels),
# thousands of noise specks are tiny (4-10 pixels), linear scale hides this
axes[1].hist(np.log10(df['n_pixels']), bins=100, color='darkorange')
axes[1].set_xlabel('log10(instance size in pixels)')
axes[1].set_ylabel('Count')
axes[1].set_title('Lake instance size distribution (log scale)')
for cutoff_pixels in [10, 20, 50, 100]:
    axes[1].axvline(np.log10(cutoff_pixels), linestyle='--', alpha=0.5, label=f'{cutoff_pixels}px')
axes[1].legend()

plt.tight_layout()
plt.savefig('lake_instance_size_histogram.png', dpi=150)
print('\nSaved: lake_instance_size_histogram.png')

# Report how many instances/how much total area survive at a few candidate cutoffs
for cutoff in [4, 10, 20, 50, 100]:
    kept = df[df['n_pixels'] >= cutoff]
    print(f'Cutoff >= {cutoff}px ({cutoff*100/1e6:.4f} km²): '
          f'{len(kept)} instances kept ({len(kept)/len(df)*100:.1f}%), '
          f'{kept["area_km2"].sum()/31:.2f} km² avg per date')