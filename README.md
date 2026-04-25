# MDP
MDP: Indian Coastline Green Hydrogen Hub Analysis
This repository contains a data processing and spatial analysis pipeline designed to identify and rank optimal locations for Green Hydrogen Hubs along the Indian coastline and the near-India regional waters.

The project evaluates sites based on solar energy potential (GHI), sea depth, proximity to maritime corridors, and safety metrics.

🚀 Project Overview
The pipeline categorizes potential sites into two distinct zones:

Zone A (Indian Coast): Locations strictly along the Indian mainland and island territories (Lakshadweep, Andaman & Nicobar) within 150km of the coast.

Zone B (Near-India Regional): Broader regional sites within 300km of the coast, including areas near Sri Lanka and international maritime channels.

🛠️ Technical Pipeline (indian_coast_pipeline.py)
The core analysis follows a 6-step automated process:

Zone Classification: Uses a densified model of the Indian coastline and Haversine distance to assign grid points to Zone A or B.

Within-Zone Ranking: Ranks sites by their H2_hub_score and calculates estimated daily Hydrogen production based on Solar GHI.

Combined Ranking: Generates a unified leaderboard with markers for deep-water suitability ([DEEP]), Tier 1 status ([T1]), and high production capacity ([H2+]).

Spatial Clustering (DBSCAN): Implements a custom DBSCAN algorithm to identify geographic "clusters" of high-potential sites to find the best regions for concentrated infrastructure.

Depth Constraint Analysis: Filters for "Natural Deep Sites" (depth ≤ -18m) that do not require dredging for large vessels.

Visualizations: Generates a multi-page PDF report containing heatmaps, scatter plots, and production bar charts.

📊 Key Findings & Data
The analysis produces several structured datasets:

combined_ranking.csv: The master list of top-performing sites.

natural_deep_sites.csv: Sites meeting the -18m depth requirement with high hub scores.

dbscan_clusters.csv: Grouped coordinates showing regional hotspots near references like Minicoy, Colombo, and Port Blair.

Example Top Sites
According to the latest run, some of the highest-ranked locations are found near:

Minicoy (Zone A/B)

Colombo (Zone B)

Eight Degree Channel (Zone B)

Great Nicobar — Galathea Bay (Zone A)

📈 Visual Output (h2_hub_indian_coast.pdf)
The pipeline generates visual summaries including:

Score Heatmaps: Geographic distribution of suitability scores across the Indian Ocean.

Depth vs Solar Scatter: Correlation analysis between sea depth and energy availability.

Production Rankings: A horizontal bar chart comparing the top 20 sites by estimated kg/day production.

💻 Requirements
Python 3.x

Libraries: numpy, pandas, matplotlib.

Input: Requires cluster_labels.csv from the global model to run the regional pipeline.