"""
Diagnostic: check for duplicate (lake_id, date) rows in the tracking output
— would confirm/deny a one-to-many matching bug in 08_lake_tracking.py
"""

import pandas as pd

df = pd.read_csv('data/lake_instances/lake_tracks_persistent.csv')

dupes = df[df.duplicated(subset=['lake_id', 'date'], keep=False)]
print(f'Total rows: {len(df)}')
print(f'Duplicate (lake_id, date) rows: {len(dupes)}')
print(f'Affected unique lake_ids: {dupes["lake_id"].nunique()}')

if len(dupes) > 0:
    print('\nExample duplicates:')
    print(dupes.sort_values(['lake_id', 'date']).head(20))