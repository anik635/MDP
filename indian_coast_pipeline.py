import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import math

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONSTANTS & CONFIGURATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
np.random.seed(42)

INPUT_CSV = 'cluster_labels.csv'
ZONE_A_CSV = 'zone_a_ranked.csv'
ZONE_B_CSV = 'zone_b_ranked.csv'
COMBINED_CSV = 'combined_ranking.csv'
NATURAL_DEEP_CSV = 'natural_deep_sites.csv'
DBSCAN_CSV = 'dbscan_clusters.csv'
OUTPUT_PDF = 'h2_hub_indian_coast.pdf'

# Polygon vertices - Raw points
RAW_INDIAN_COASTLINE = [
    # Western coast
    (23.5, 68.2), (22.5, 69.7), (20.9, 70.1), (19.1, 72.8),
    (15.5, 73.8), (12.9, 74.8), (11.0, 75.9), (8.8, 76.9), (8.1, 77.3),
    # Southern tip
    (8.1, 77.3), (8.0, 77.6),
    # Eastern coast
    (8.0, 77.6), (9.2, 79.1), (10.8, 79.8), (12.0, 80.3),
    (13.6, 80.2), (15.9, 80.4), (17.7, 83.3), (19.3, 84.8),
    (20.5, 86.9), (21.5, 87.3)
]

# Densify the coastline to ensure points are strictly on the coast
dense_coastline = []
for i in range(len(RAW_INDIAN_COASTLINE) - 1):
    p1 = np.array(RAW_INDIAN_COASTLINE[i])
    p2 = np.array(RAW_INDIAN_COASTLINE[i+1])
    d = np.linalg.norm(p2 - p1)
    num_points = max(2, int(d / 0.5))
    for t in np.linspace(0, 1, num_points):
        dense_coastline.append(tuple(p1 + t * (p2 - p1)))

# Explicit island references to ensure we capture them
ISLAND_REFS = [
    (10.5, 72.6), (11.7, 92.7), (7.0, 93.8), # original refs
    (8.28, 73.05), (10.57, 72.64), (10.82, 72.19), # Lakshadweep
    (11.67, 92.74), (9.15, 92.82), (6.75, 93.85), (7.00, 93.92) # Andaman & Nicobar
]

INDIAN_COASTLINE = np.array(list(set(dense_coastline + ISLAND_REFS)))

# Geographic Reference Points
GEO_REFS = {
    "Kandla": (23.00, 70.22), "Mumbai": (18.93, 72.83), "Mormugao/Goa": (15.40, 73.80),
    "New Mangalore": (12.90, 74.80), "Kochi": (9.97, 76.27), "Vizhinjam": (8.38, 76.96),
    "Colachel/Enayam": (8.17, 77.27), "Tuticorin/Thoothukudi": (8.80, 78.13),
    "Chennai": (13.08, 80.28), "Visakhapatnam": (17.68, 83.23), "Paradip": (20.32, 86.61),
    "Haldia": (22.02, 88.07), "Minicoy": (8.28, 73.05), "Kavaratti": (10.57, 72.64),
    "Agatti": (10.82, 72.19), "Port Blair": (11.67, 92.74), "Car Nicobar": (9.15, 92.82),
    "Great Nicobar — Galathea Bay": (6.75, 93.85), "Campbell Bay": (7.00, 93.92),
    "Colombo": (6.93, 79.85), "Hambantota": (6.12, 81.12), "Trincomalee": (8.58, 81.23),
    "Galle": (6.03, 80.22), "Cape Comorin/Kanyakumari": (8.08, 77.55),
    "Strait of Malacca North": (5.60, 95.30), "Nine Degree Channel": (9.00, 74.00),
    "Eight Degree Channel": (8.00, 77.00), "Male": (4.18, 73.51), "Addu Atoll": (-0.63, 73.15)
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HELPER FUNCTIONS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

def haversine_vectorized(lat, lon, lat_arr, lon_arr):
    # Quick vectorized distance computation for DBSCAN
    R = 6371.0
    phi1 = np.radians(lat)
    phi2 = np.radians(lat_arr)
    dphi = np.radians(lat_arr - lat)
    dlambda = np.radians(lon_arr - lon)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))

def find_nearest_reference(lat, lon):
    min_dist = float('inf')
    best_name = "Unknown"
    for name, (rlat, rlon) in GEO_REFS.items():
        d = haversine(lat, lon, rlat, rlon)
        if d < min_dist:
            min_dist = d
            best_name = name
    return best_name, min_dist

def classify_zone(lat, lon):
    # Compute dist to nearest coastline vertex
    dists = [haversine(lat, lon, v[0], v[1]) for v in INDIAN_COASTLINE]
    min_coast_dist = min(dists)
    
    # Zone A logic - strictly on coast/islands
    if (6.0 <= lat <= 23.5) and (68.0 <= lon <= 93.5) and (min_coast_dist <= 150):
        return 'A — Indian Coast'
    
    # Zone B logic - near regional (avoiding middle of sea)
    if (-5.0 <= lat <= 25.0) and (55.0 <= lon <= 100.0) and (min_coast_dist <= 300):
        return 'B — Near-India Regional'
        
    return 'Outside study area'

# DBSCAN from scratch
def dbscan_custom(df, eps=150, min_samples=3):
    labels = -1 * np.ones(len(df)) # -1 is noise
    cluster_id = 0
    
    lats = df['lat'].values
    lons = df['lon'].values
    
    for i in range(len(df)):
        if labels[i] != -1:
            continue
            
        dists = haversine_vectorized(lats[i], lons[i], lats, lons)
        neighbors = np.where(dists <= eps)[0]
        
        if len(neighbors) < min_samples:
            labels[i] = -1 # Noise
            continue
            
        labels[i] = cluster_id
        seed_set = list(neighbors)
        seed_set.remove(i)
        
        while seed_set:
            q = seed_set.pop(0)
            if labels[q] == -1: # previously noise
                labels[q] = cluster_id
            if labels[q] != -1: # already assigned
                continue
                
            labels[q] = cluster_id
            q_dists = haversine_vectorized(lats[q], lons[q], lats, lons)
            q_neighbors = np.where(q_dists <= eps)[0]
            
            if len(q_neighbors) >= min_samples:
                for n in q_neighbors:
                    if labels[n] < 0: # unvisited or noise
                        seed_set.append(n)
        cluster_id += 1
    return labels

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN PIPELINE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def run_pipeline():
    print("Loading data from previous run...")
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"Missing {INPUT_CSV}. Please run the global model first.")
        
    df = pd.read_csv(INPUT_CSV)
    
    print("\nSTEP 1: Zone Classification")
    df['Zone'] = df.apply(lambda row: classify_zone(row['lat'], row['lon']), axis=1)
    
    zone_a = df[df['Zone'] == 'A — Indian Coast'].copy()
    zone_b = df[df['Zone'] == 'B — Near-India Regional'].copy()
    
    print(f"Zone A: {len(zone_a)} grid points identified")
    print(f"Zone B: {len(zone_b)} grid points identified")
    
    print("\nSTEP 2: Within-Zone Ranking")
    zone_a = zone_a.sort_values(by='H2_hub_score', ascending=False)
    zone_b = zone_b.sort_values(by='H2_hub_score', ascending=False)
    
    top20_a = zone_a.head(20).copy()
    top20_b = zone_b.head(20).copy()
    
    # Calculate additional columns for top 20s
    def enrich_df(d):
        d['Within_Zone_Rank'] = np.arange(1, len(d) + 1)
        # H2_kg_day = GHI_raw * 1,000,000 * 0.20 * 0.70 / 39.4
        d['H2_Production_kg_day'] = d['solar_ghi_raw'] * 1000000 * 0.20 * 0.70 / 39.4
        ref_names, ref_dists = [], []
        for _, row in d.iterrows():
            name, dist = find_nearest_reference(row['lat'], row['lon'])
            ref_names.append(name)
            ref_dists.append(dist)
        d['Nearest_Reference'] = ref_names
        d['Dist_to_Ref_km'] = ref_dists
        return d
        
    top20_a = enrich_df(top20_a)
    top20_b = enrich_df(top20_b)
    
    top20_a.to_csv(ZONE_A_CSV, index=False)
    top20_b.to_csv(ZONE_B_CSV, index=False)
    print(f"Zone A Top 20 written to {ZONE_A_CSV}")
    print(f"Zone B Top 20 written to {ZONE_B_CSV}")
    
    print("\nSTEP 3: Combined Ranking Table")
    combined = pd.concat([top20_a, top20_b])
    combined = combined.sort_values(by='H2_hub_score', ascending=False).reset_index(drop=True)
    combined['Combined_Rank'] = np.arange(1, len(combined) + 1)
    
    # Generate markers
    def get_markers(row):
        m = []
        if row['Zone'] == 'A — Indian Coast': m.append('[IN-EEZ]')
        if row['depth_m'] <= -18.0: m.append('[DEEP]')
        if row['Tier'] == 1: m.append('[T1]')
        if row['H2_Production_kg_day'] >= 3500: m.append('[H2+]')
        return " ".join(m)
        
    combined['Markers'] = combined.apply(get_markers, axis=1)
    
    print("\nCombined Top 5:")
    print(combined[['Combined_Rank', 'Zone', 'lat', 'lon', 'H2_hub_score', 'Markers']].head(5))
    
    out_cols = ['Combined_Rank', 'Zone', 'lat', 'lon', 'depth_m', 'solar_ghi_raw', 
                'H2_hub_score', 'Tier', 'Nearest_Reference', 'H2_Production_kg_day', 'Markers']
    combined[out_cols].to_csv(COMBINED_CSV, index=False)
    print(f"Combined ranking written to {COMBINED_CSV}")
    
    print("\nSTEP 4: Spatial Cluster Analysis Within Zones (DBSCAN)")
    top50_a = zone_a.head(50).copy()
    top50_b = zone_b.head(50).copy()
    
    top50_a['Cluster_ID'] = dbscan_custom(top50_a)
    top50_a['Cluster_Zone'] = 'A'
    top50_b['Cluster_ID'] = dbscan_custom(top50_b)
    top50_b['Cluster_Zone'] = 'B'
    
    dbscan_res = pd.concat([top50_a, top50_b])
    dbscan_res.to_csv(DBSCAN_CSV, index=False)
    
    print("Discovered Clusters:")
    for z, c_df in zip(['A', 'B'], [top50_a, top50_b]):
        for cid in np.unique(c_df['Cluster_ID']):
            if cid == -1: continue # Noise
            members = c_df[c_df['Cluster_ID'] == cid]
            n_members = len(members)
            c_lat, c_lon = members['lat'].mean(), members['lon'].mean()
            mean_score = members['H2_hub_score'].mean()
            dom_tier = members['Tier'].mode().values[0]
            ref_name, _ = find_nearest_reference(c_lat, c_lon)
            print(f"Zone {z} - Cluster {cid}: {n_members} pts, Centroid: ({c_lat:.2f}, {c_lon:.2f}), "
                  f"Score: {mean_score:.1f}, Tier: {dom_tier}, Ref: {ref_name}")

    print("\nSTEP 5: Depth Constraint Analysis")
    all_ab = pd.concat([zone_a, zone_b])
    deep_sites = all_ab[(all_ab['H2_hub_score'] >= 50) & (all_ab['depth_m'] <= -18.0)].copy()
    deep_sites = deep_sites.sort_values(by='H2_hub_score', ascending=False)
    deep_sites.to_csv(NATURAL_DEEP_CSV, index=False)
    print(f"Found {len(deep_sites)} naturally deep-water sites (No dredging). Written to {NATURAL_DEEP_CSV}")
    print(deep_sites[['lat', 'lon', 'depth_m', 'H2_hub_score', 'Zone']].head())
    
    print("\nSTEP 6: Visualisations")
    from matplotlib.backends.backend_pdf import PdfPages
    
    with PdfPages(OUTPUT_PDF) as pdf:
        # Simplified Indian Coastline for plotting
        cl_lon = [v[1] for v in RAW_INDIAN_COASTLINE]
        cl_lat = [v[0] for v in RAW_INDIAN_COASTLINE]
        
        # Figure 1: Zone A Heatmap
        plt.figure(figsize=(10, 8))
        if len(zone_a) > 0:
            sc = plt.scatter(zone_a['lon'], zone_a['lat'], c=zone_a['H2_hub_score'], cmap='YlOrRd', s=50)
            plt.colorbar(sc, label='H2 Hub Score')
            top10_a = top20_a.head(10)
            for i, (_, row) in enumerate(top10_a.iterrows()):
                plt.annotate(str(i+1), (row['lon'], row['lat']), fontsize=12, fontweight='bold')
        plt.plot(cl_lon, cl_lat, 'k-', lw=2, label='Indian Coastline')
        plt.title('Figure 1: Zone A Score Map')
        plt.legend()
        pdf.savefig()
        plt.close()
        
        # Figure 2: Zone B Heatmap
        plt.figure(figsize=(10, 8))
        if len(zone_b) > 0:
            sc = plt.scatter(zone_b['lon'], zone_b['lat'], c=zone_b['H2_hub_score'], cmap='YlOrRd', s=50)
            plt.colorbar(sc, label='H2 Hub Score')
            top10_b = top20_b.head(10)
            for i, (_, row) in enumerate(top10_b.iterrows()):
                plt.annotate(str(i+1), (row['lon'], row['lat']), fontsize=12, fontweight='bold')
        plt.plot(cl_lon, cl_lat, 'k-', lw=2)
        plt.title('Figure 2: Zone B Score Map')
        pdf.savefig()
        plt.close()
        
        # Figure 3: Combined Score Map
        plt.figure(figsize=(12, 10))
        plt.scatter(zone_a['lon'], zone_a['lat'], c='lightgrey', alpha=0.5, s=20, label='Zone A Background')
        plt.scatter(zone_b['lon'], zone_b['lat'], c='lightblue', alpha=0.5, s=20, label='Zone B Background')
        plt.scatter(top20_a['lon'], top20_a['lat'], c='red', marker='o', s=80, label='Zone A Top 20')
        plt.scatter(top20_b['lon'], top20_b['lat'], c='blue', marker='s', s=80, label='Zone B Top 20')
        top5_comb = combined.head(5)
        for i, (_, row) in enumerate(top5_comb.iterrows()):
            plt.annotate(str(i+1), (row['lon'], row['lat']), fontsize=14, fontweight='bold', color='black')
        plt.plot(cl_lon, cl_lat, 'k-', lw=2)
        plt.title('Figure 3: Combined Score Map')
        plt.legend()
        pdf.savefig()
        plt.close()
        
        # Figure 4: Depth vs Solar Scatter
        plt.figure(figsize=(10, 8))
        all_ab_valid = all_ab.copy()
        all_ab_valid['depth_abs'] = all_ab_valid['depth_m'].abs()
        z_a = all_ab_valid[all_ab_valid['Zone'] == 'A — Indian Coast']
        z_b = all_ab_valid[all_ab_valid['Zone'] == 'B — Near-India Regional']
        
        plt.scatter(z_a['solar_ghi_raw'], z_a['depth_abs'], c='coral', s=z_a['H2_hub_score']/5, label='Zone A', alpha=0.7)
        plt.scatter(z_b['solar_ghi_raw'], z_b['depth_abs'], c='teal', s=z_b['H2_hub_score']/5, label='Zone B', alpha=0.7)
        plt.axhline(y=18, color='r', linestyle='--', label='18m Depth Limit')
        plt.xlabel('Solar GHI (kWh/m²/day)')
        plt.ylabel('Sea Depth Magnitude (m)')
        plt.title('Figure 4: Depth vs Solar Scatter')
        plt.legend()
        pdf.savefig()
        plt.close()
        
        # Figure 5: DBSCAN Clusters
        plt.figure(figsize=(12, 10))
        plt.plot(cl_lon, cl_lat, 'k-', lw=2)
        cmap = plt.get_cmap('tab10')
        color_idx = 0
        for z, c_df in zip(['A', 'B'], [top50_a, top50_b]):
            for cid in np.unique(c_df['Cluster_ID']):
                members = c_df[c_df['Cluster_ID'] == cid]
                if cid == -1:
                    plt.scatter(members['lon'], members['lat'], c='grey', s=30, label='Noise' if z=='A' else "")
                else:
                    plt.scatter(members['lon'], members['lat'], color=cmap(color_idx % 10), s=100, label=f'Zone {z} C{cid}')
                    plt.plot(members['lon'].mean(), members['lat'].mean(), 'kX', markersize=15)
                    color_idx += 1
        plt.title('Figure 5: DBSCAN Geographic Clusters')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        pdf.savefig()
        plt.close()
        
        # Figure 6: H2 Production Bar Chart
        plt.figure(figsize=(12, 8))
        top20_comb = combined.head(20).copy()
        top20_comb = top20_comb.sort_values('H2_Production_kg_day', ascending=True)
        colors = ['red' if t == 1 else 'orange' if t == 2 else 'yellow' for t in top20_comb['Tier']]
        
        bars = plt.barh(np.arange(len(top20_comb)), top20_comb['H2_Production_kg_day'], color=colors)
        plt.yticks(np.arange(len(top20_comb)), [f"#{r} {ref} (Depth: {d:.0f}m)" for r, ref, d in 
                   zip(top20_comb['Combined_Rank'], top20_comb['Nearest_Reference'], top20_comb['depth_m'])])
        plt.xlabel('Estimated H2 Production (kg/day)')
        plt.title('Figure 6: H2 Production Ranking')
        plt.tight_layout()
        pdf.savefig()
        plt.close()
        print(f"\nSaved all figures to {OUTPUT_PDF}")

if __name__ == '__main__':
    run_pipeline()
