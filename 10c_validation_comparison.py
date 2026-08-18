"""
Phase 6, Step 3: Compare our detected lake area against Qiu & Ran (2023)
Russell Glacier / Paakitsoq AOI — 2019 melt season

Sums the Qiu & Ran area time series across the 12 grid cells overlapping
our AOI, filters to 2019, and plots against our own model-detected area
time series as an independent external validation check.

Prereqs (already installed): pandas, matplotlib
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

TS_DIR = Path('data/validation/SGL_area_time_series_with_uncertainty_assessment/SGL_area_time_series_with_uncertainty_assessment')
OVERLAPPING_GRID_IDS = [168, 1568, 1716, 1913, 1916, 1978, 2478, 2869, 2896, 3132, 3628, 3639]

# --- 1. Load and combine all overlapping grid cell time series ---
all_records = []
for grid_id in OVERLAPPING_GRID_IDS:
    filepath = TS_DIR / f'Area_Grid{grid_id}.txt'
    if not filepath.exists():
        print(f'Missing: {filepath.name}')
        continue
    df = pd.read_csv(filepath, sep='\t', header=None,
                      names=['decimal_year', 'date', 'area_km2', 'uncertainty_km2'])
    df['grid_id'] = grid_id
    all_records.append(df)

qiu_ran_df = pd.concat(all_records, ignore_index=True)
qiu_ran_df['date'] = pd.to_datetime(qiu_ran_df['date'])

# --- 2. Filter to our 2019 melt season window ---
qiu_ran_2019 = qiu_ran_df[
    (qiu_ran_df['date'] >= '2019-06-01') & (qiu_ran_df['date'] <= '2019-09-15')
].copy()

print(f'Qiu & Ran 2019 observations across 12 grid cells: {len(qiu_ran_2019)}')
print(f'Unique dates: {qiu_ran_2019["date"].nunique()}')

# --- 3. Sum area across all grid cells per date ---
qiu_ran_daily_total = qiu_ran_2019.groupby('date')['area_km2'].sum().reset_index()
qiu_ran_daily_total = qiu_ran_daily_total.sort_values('date')

print('\nQiu & Ran total area by date (summed across 12 grid cells):')
print(qiu_ran_daily_total.to_string(index=False))

# --- 4. Load our own detected time series (Phase 2 bootstrap labels, for
#         direct comparability — same underlying detection concept: total
#         lake area from optical imagery) ---
our_df = pd.read_csv('output/lake_area_timeseries_2019_clean.csv')
our_df['date'] = pd.to_datetime(our_df['date'])

# --- 5. Plot both together ---
fig, ax = plt.subplots(figsize=(13, 6))
ax.plot(our_df['date'], our_df['lake_area_km2'], marker='o', label='Our detection (this project)', color='steelblue')
ax.plot(qiu_ran_daily_total['date'], qiu_ran_daily_total['area_km2'], marker='s',
        label='Qiu & Ran (2023) — 12 overlapping grid cells', color='darkorange')
ax.set_xlabel('Date')
ax.set_ylabel('Total lake area (km²)')
ax.set_title('Validation: our detected lake area vs. Qiu & Ran (2023), Russell Glacier/Paakitsoq, 2019')
ax.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('validation_comparison.png', dpi=150)
print('\nSaved: validation_comparison.png')

# --- 6. Summary stats for a direct sanity check ---
print(f'\nOur mean area: {our_df["lake_area_km2"].mean():.2f} km²')
print(f'Qiu & Ran mean area (2019, our AOI): {qiu_ran_daily_total["area_km2"].mean():.2f} km²')