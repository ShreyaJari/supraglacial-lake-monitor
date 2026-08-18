"""
Phase 5 (v2): Drainage detection with stability filtering
Russell Glacier / Paakitsoq, SW Greenland — 1,591 tracked lakes

v1 produced 1,059 candidate events from 1,591 tracks (67%) — not physically
credible. Diagnosis from visual inspection of example curves:
    - Lake 1149: area spiked from ~0 to 0.78 km2 in a SINGLE observation,
      then crashed — implausible instant lake formation, likely a spurious
      single-frame detection, not a real filling lake.
    - Lake 2574: only 2 total appearances, 0.65 -> 0 km2 with zero buildup
      history — too fragile to trust as a real event.
    - Lake 1427: oscillated between ~0.3 and ~0.03 km2 REPEATEDLY across
      the season (not just once) — real lakes don't fully drain and refill
      multiple times; this points to either an unstable track (matching
      linking distinct nearby features) or per-lake cloud noise. Either
      way, a track showing this pattern can't be trusted to isolate one
      real event from noise.

New filters:
    1. Minimum track length >= 3 appearances to be eligible at all.
    2. Gradual buildup: at least one appearance before the peak must reach
       >=10% of peak area — rules out instant single-frame "lakes".
    3. Instability exclusion: tracks with >=2 full drop-then-recover cycles
       across their whole history are excluded entirely (untrustworthy).
    4. HIGH confidence additionally requires the peak be supported by >=2
       observations (not a single anomalous reading).

Prereqs (already installed): pandas, numpy, matplotlib
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DROP_THRESHOLD = 0.5
RECOVERY_THRESHOLD = 0.7
STAYS_LOW_THRESHOLD = 0.3
MIN_TRACK_LENGTH = 3
MIN_BUILDUP_FRACTION = 0.1

df = pd.read_csv('data/lake_instances/lake_tracks_persistent.csv')
df['date_parsed'] = pd.to_datetime(df['date'])

events = []
excluded_unstable = []

for lake_id, group in df.groupby('lake_id'):
    track = group.sort_values('date_parsed').reset_index(drop=True)

    if len(track) < MIN_TRACK_LENGTH:
        continue  # Filter 1: too fragile to trust

    areas = track['area_km2'].values

    # --- Filter 3: count full swing cycles across the WHOLE track ---
    swing_count = 0
    running_peak = areas[0]
    in_drawdown = False
    for a in areas[1:]:
        if not in_drawdown:
            running_peak = max(running_peak, a)
            if a <= (1 - DROP_THRESHOLD) * running_peak and running_peak > 0:
                in_drawdown = True
        else:
            if a >= RECOVERY_THRESHOLD * running_peak:
                swing_count += 1
                in_drawdown = False
                running_peak = a  # reset peak tracking after a confirmed cycle

    if swing_count >= 2:
        excluded_unstable.append({'lake_id': lake_id, 'swing_count': swing_count, 'track_length': len(track)})
        continue  # Filter 3: whole track excluded as unstable

    # --- Scan for candidate drainage events on stable tracks ---
    for i in range(len(track) - 1):
        area_after = track.loc[i + 1, 'area_km2']
        date_before = track.loc[i, 'date']
        date_after = track.loc[i + 1, 'date']
        days_between = (track.loc[i + 1, 'date_parsed'] - track.loc[i, 'date_parsed']).days

        peak_area_so_far = track.loc[:i, 'area_km2'].max()
        fractional_drop = (peak_area_so_far - area_after) / peak_area_so_far if peak_area_so_far > 0 else 0

        if fractional_drop < DROP_THRESHOLD:
            continue

        # --- Filter 2: gradual buildup check ---
        pre_peak_areas = track.loc[:i, 'area_km2']
        had_buildup = (pre_peak_areas >= MIN_BUILDUP_FRACTION * peak_area_so_far).sum() >= 1
        # (trivially true for the peak observation itself; we actually want
        # evidence BEFORE the single peak reading, so check all obs except the max)
        pre_peak_excl_max = pre_peak_areas[pre_peak_areas < peak_area_so_far]
        had_real_buildup = len(pre_peak_excl_max) == 0 or (pre_peak_excl_max >= MIN_BUILDUP_FRACTION * peak_area_so_far).any()

        if not had_real_buildup and i >= 1:
            continue  # peak appeared out of nowhere — don't trust it as a real lake state

        peak_supported_by_multiple_obs = (i >= 1)

        later_appearances = track.loc[i + 2:]

        if len(later_appearances) == 0:
            confidence = 'MEDIUM (near season end)' if date_after >= '2019-09-01' else 'MEDIUM (track ends after drop)'
        else:
            max_later_area = later_appearances['area_km2'].max()
            if max_later_area >= RECOVERY_THRESHOLD * peak_area_so_far:
                continue  # recovers — not real drainage
            elif max_later_area <= STAYS_LOW_THRESHOLD * peak_area_so_far:
                confidence = 'HIGH' if peak_supported_by_multiple_obs else 'MEDIUM (peak = single reading)'
            else:
                confidence = 'MEDIUM (partial recovery)'

        events.append({
            'lake_id': lake_id, 'date_before': date_before, 'date_after': date_after,
            'days_between': days_between, 'peak_area_km2': round(peak_area_so_far, 5),
            'area_after_km2': round(area_after, 5), 'fractional_drop': round(fractional_drop, 3),
            'confidence': confidence, 'track_length': len(track),
        })

events_df = pd.DataFrame(events)
excluded_df = pd.DataFrame(excluded_unstable)

events_df.to_csv('drainage_events_v2.csv', index=False)
excluded_df.to_csv('excluded_unstable_tracks.csv', index=False)

print(f'Tracks excluded as unstable (>=2 swing cycles): {len(excluded_df)}')
print(f'Total candidate drainage events (v2): {len(events_df)}')
if len(events_df) > 0:
    print(events_df['confidence'].value_counts())
print(f'\nSaved: drainage_events_v2.csv, excluded_unstable_tracks.csv')

# --- Plots ---
if len(events_df) > 0:
    high_conf = events_df[events_df['confidence'] == 'HIGH']
    med_conf = events_df[events_df['confidence'] != 'HIGH']

    fig, ax = plt.subplots(figsize=(12, 5))
    for subset, label, color in [(high_conf, 'High confidence', 'crimson'), (med_conf, 'Medium confidence', 'orange')]:
        if len(subset) > 0:
            ax.hist(pd.to_datetime(subset['date_after']), bins=20, alpha=0.7, label=label, color=color)
    ax.set_xlabel('Date'); ax.set_ylabel('Number of drainage events detected')
    ax.set_title('Detected supraglacial lake drainage events (v2, filtered), 2019 melt season')
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('drainage_events_timeline_v2.png', dpi=150)
    print('Saved: drainage_events_timeline_v2.png')

    top_events = high_conf.nlargest(min(6, len(high_conf)), 'fractional_drop')
    if len(top_events) > 0:
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        axes = axes.flatten()
        for idx, (_, event) in enumerate(top_events.iterrows()):
            if idx >= 6:
                break
            lake_track = df[df['lake_id'] == event['lake_id']].sort_values('date_parsed')
            axes[idx].plot(lake_track['date_parsed'], lake_track['area_km2'], marker='o')
            axes[idx].axvline(pd.to_datetime(event['date_after']), color='red', linestyle='--', alpha=0.7)
            axes[idx].set_title(f"Lake {event['lake_id']} ({event['fractional_drop']*100:.0f}% drop)")
            axes[idx].tick_params(axis='x', rotation=45)
            axes[idx].set_ylabel('Area (km²)')
        plt.tight_layout()
        plt.savefig('drainage_examples_v2.png', dpi=150)
        print('Saved: drainage_examples_v2.png')