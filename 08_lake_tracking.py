"""
Phase 4, Step 2 (v2 - FIXED): Cross-date lake tracking with strict 1:1 matching
Russell Glacier / Paakitsoq, SW Greenland — 31 dates

BUG FOUND (v1): matching only checked whether the TARGET region already had
an assigned ID, not whether the SOURCE region had already been matched to
something else. Result: one lake in date A could match to multiple regions
in date B (e.g. an apparent split), all incorrectly inheriting the same
lake_id — confirmed via duplicate (lake_id, date) rows: 1,748 duplicates
across 296 lake_ids, ~17% of the total dataset.

FIX: strict one-to-one greedy matching. All candidate overlapping pairs are
collected, sorted by overlap strength (strongest first), then assigned
greedily — once either side of a pair is used, it can't be matched again.

Prereqs (already installed): numpy, pandas, scipy
"""

import numpy as np
import pandas as pd
from scipy import ndimage
from pathlib import Path

INSTANCES_DIR = Path('data/lake_instances')
MIN_PIXELS = 20
MIN_OVERLAP_FRACTION = 0.3

CLEAN_DATES = [
    '2019-06-01', '2019-06-05', '2019-06-08', '2019-06-10', '2019-06-11',
    '2019-06-15', '2019-06-16', '2019-06-18', '2019-06-20',
    '2019-07-03', '2019-07-06', '2019-07-08', '2019-07-10', '2019-07-11',
    '2019-07-15', '2019-07-16', '2019-07-20', '2019-07-21', '2019-07-23',
    '2019-07-31',
    '2019-08-02', '2019-08-04', '2019-08-09', '2019-08-10', '2019-08-12',
    '2019-08-20', '2019-08-24', '2019-08-25', '2019-08-29',
    '2019-09-01', '2019-09-09'
]


def load_filtered_labels(date_str):
    pred_mask = np.load(INSTANCES_DIR / f'{date_str}_pred_mask.npy')
    labeled, n_features = ndimage.label(pred_mask)
    sizes = ndimage.sum(pred_mask, labeled, range(1, n_features + 1))
    keep_labels = np.where(sizes >= MIN_PIXELS)[0] + 1
    filtered = np.where(np.isin(labeled, keep_labels), labeled, 0)
    relabeled, n_kept = ndimage.label(filtered > 0)
    return relabeled, n_kept


def compute_overlaps(labels_a, labels_b):
    mask = (labels_a > 0) & (labels_b > 0)
    pairs = np.stack([labels_a[mask], labels_b[mask]], axis=1)
    if len(pairs) == 0:
        return {}
    unique_pairs, counts = np.unique(pairs, axis=0, return_counts=True)
    return {(int(a), int(b)): int(c) for (a, b), c in zip(unique_pairs, counts)}


def region_sizes(labels, n_regions):
    sizes = ndimage.sum(labels > 0, labels, range(1, n_regions + 1))
    return {i + 1: int(s) for i, s in enumerate(sizes)}


print('Loading and filtering per-date masks...')
date_labels = {}
date_sizes = {}
for date_str in CLEAN_DATES:
    labels, n_regions = load_filtered_labels(date_str)
    date_labels[date_str] = labels
    date_sizes[date_str] = region_sizes(labels, n_regions)
    print(f'  {date_str}: {n_regions} instances after size filter')

next_global_id = 1
local_id_to_global = {d: {} for d in CLEAN_DATES}

for i, date_str in enumerate(CLEAN_DATES):
    for local_label in date_sizes[date_str]:
        if local_label not in local_id_to_global[date_str]:
            local_id_to_global[date_str][local_label] = next_global_id
            next_global_id += 1

    if i == len(CLEAN_DATES) - 1:
        continue

    next_date = CLEAN_DATES[i + 1]
    overlaps = compute_overlaps(date_labels[date_str], date_labels[next_date])

    # --- Build candidate match list with overlap fraction, sorted strongest-first ---
    candidates = []
    for (label_a, label_b), overlap_px in overlaps.items():
        size_a = date_sizes[date_str][label_a]
        size_b = date_sizes[next_date][label_b]
        smaller_size = min(size_a, size_b)
        overlap_fraction = overlap_px / smaller_size
        if overlap_fraction >= MIN_OVERLAP_FRACTION:
            candidates.append((overlap_fraction, label_a, label_b))

    candidates.sort(key=lambda x: x[0], reverse=True)

    # --- Strict greedy 1:1 matching ---
    used_a, used_b = set(), set()
    for overlap_fraction, label_a, label_b in candidates:
        if label_a in used_a or label_b in used_b:
            continue  # either side already matched this round — skip
        global_id = local_id_to_global[date_str][label_a]
        local_id_to_global[next_date][label_b] = global_id
        used_a.add(label_a)
        used_b.add(label_b)

records = []
for date_str in CLEAN_DATES:
    labels = date_labels[date_str]
    sizes = date_sizes[date_str]
    for local_label, n_pixels in sizes.items():
        global_id = local_id_to_global[date_str][local_label]
        region_mask = (labels == local_label)
        rows, cols = np.where(region_mask)
        records.append({
            'lake_id': global_id, 'date': date_str,
            'area_km2': round(n_pixels * 100 / 1e6, 5), 'n_pixels': n_pixels,
            'centroid_row': rows.mean(), 'centroid_col': cols.mean(),
        })

df = pd.DataFrame(records)

# --- Verify the fix: check for duplicates before proceeding ---
dupes = df[df.duplicated(subset=['lake_id', 'date'], keep=False)]
print(f'\nDuplicate (lake_id, date) rows after fix: {len(dupes)} (should be 0)')
assert len(dupes) == 0, 'Fix did not fully resolve duplicates — investigate further before proceeding'

appearance_counts = df.groupby('lake_id')['date'].nunique()
persistent_ids = appearance_counts[appearance_counts >= 2].index
df_persistent = df[df['lake_id'].isin(persistent_ids)].copy()

print(f'\nTotal lake tracks (before persistence filter): {df["lake_id"].nunique()}')
print(f'Persistent lake tracks (appear in 2+ dates): {df_persistent["lake_id"].nunique()}')
print(f'Dropped as single-date noise: {df["lake_id"].nunique() - df_persistent["lake_id"].nunique()}')

df.to_csv('data/lake_instances/lake_tracks_all.csv', index=False)
df_persistent.to_csv('data/lake_instances/lake_tracks_persistent.csv', index=False)
print('\nSaved: data/lake_instances/lake_tracks_persistent.csv (OVERWRITTEN with corrected data)')

track_lengths = df_persistent.groupby('lake_id')['date'].nunique()
print(f'\nTrack length distribution (persistent lakes):')
print(track_lengths.describe())