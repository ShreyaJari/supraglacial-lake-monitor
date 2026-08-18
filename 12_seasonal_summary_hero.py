"""
Phase 7 (viz): Seasonal summary "hero" plot
Total persistent-tracked lake area across the 2019 season, with
high-confidence drainage events marked directly on the curve.

Note: this uses the persistent-tracked total (post noise-filtering), which
will run somewhat lower than the raw NDWI+red threshold total from Phase 2
(that total included single-date noise later filtered out in Phase 4) —
this is the internally consistent number for the detection/tracking/
drainage narrative.

Prereqs (already installed): pandas, matplotlib
"""

import pandas as pd
import matplotlib.pyplot as plt

tracks_df = pd.read_csv('data/lake_instances/lake_tracks_persistent.csv')
events_df = pd.read_csv('drainage_events_v2.csv')

tracks_df['date_parsed'] = pd.to_datetime(tracks_df['date'])
daily_total = tracks_df.groupby('date_parsed')['area_km2'].sum().reset_index().sort_values('date_parsed')

high_conf_events = events_df[events_df['confidence'] == 'HIGH'].copy()
high_conf_events['date_after_parsed'] = pd.to_datetime(high_conf_events['date_after'])

fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(daily_total['date_parsed'], daily_total['area_km2'], marker='o', color='steelblue',
        linewidth=1.5, markersize=4, label='Total tracked lake area', zorder=2)

# Mark each high-confidence drainage event date along the bottom
event_dates = high_conf_events['date_after_parsed']
y_min = daily_total['area_km2'].min()
ax.scatter(event_dates, [y_min * 0.9] * len(event_dates), marker='v', color='crimson',
           s=30, alpha=0.7, label=f'High-confidence drainage events ({len(high_conf_events)})', zorder=3)

ax.set_xlabel('Date')
ax.set_ylabel('Total tracked lake area (km²)')
ax.set_title('Supraglacial lake area and detected drainage events, 2019 melt season\nRussell Glacier / Paakitsoq, SW Greenland')
ax.legend(loc='upper right')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('figures/seasonal_summary_hero.png', dpi=150)
print('Saved: figures/seasonal_summary_hero.png')