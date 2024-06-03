#%%
import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.font_manager import FontProperties
import seaborn as sns
import os

# Function to read the edge list and create a networkx graph
def create_networkx_graph(file_path):
    df = pd.read_csv(file_path)
    G = nx.from_pandas_edgelist(df, source='v1', target='v2', edge_attr='asso', create_using=nx.Graph())
    return G

# Function to calculate general topological metrics for a networkx graph
def calculate_network_metrics(G):
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()

    # Using Louvain method for community detection
    communities = list(nx.algorithms.community.louvain_communities(G, seed=12))
    num_communities = len(communities)

    # Calculate modularity
    modularity = nx.algorithms.community.modularity(G, communities)

    # Calculate average degree and density
    degrees = [deg for node, deg in G.degree(weight='asso')]
    avg_degree = np.mean(degrees)
    density = nx.density(G)

    # average clustering coefficient,
    # Create a copy of the graph to modify the weights for the average clustering coefficient calculation
    G_absweights = G.copy()

    # Iterate through the edges and update the weight to its absolute value
    for u, v, data in G_absweights.edges(data=True):
        if 'asso' in data:  
            data['asso'] = abs(data['asso'])

    # Calculate the average clustering coefficient with the modified weights in H
    avg_clustering_coefficient = nx.average_clustering(G_absweights, weight='asso')

    return {
        "Nodes": num_nodes,
        "Edges": num_edges,
        "Number_communities": num_communities,
        "Modularity": modularity,
        "Avg_degree": avg_degree,
        "Avg_clustering_coefficient": avg_clustering_coefficient,
        "Density": density
    }

def create_metrics_inconsistent(threshold, network_type, biome, message, n_nodes = None, n_edges = None):
    """
    Create a placeholder metrics dictionary for cases where no network file is available.
    """
    return {
        "Threshold": threshold,
        "NetworkType": network_type,
        "Biome": biome,
        "Nodes": n_nodes if n_nodes else message,
        "Edges": n_edges if n_edges else message,
        "Number_communities": message,
        "Modularity": message,
        "Avg_degree": message,
        "Avg_clustering_coefficient": message,
        "Density": message  
    }

def process_biome_networks(biome, biome_path, network_type):
    """
    Process all networks in a study, returning a DataFrame with the metrics for each network.
    """
    metrics_list = []

    # Initialize variables to keep track of the top three networks by modularity and their metrics
    top_networks_by_modularity = []

    # Process each network folder within the biome folder
    for network_folder in os.listdir(biome_path):
        if network_folder.startswith(network_type):
            network_folder_path = os.path.join(biome_path, network_folder)
            threshold = network_folder.split('_')[-1]
            # network_files = [f for f in os.listdir(network_folder_path) if f.endswith('.csv')]
            network_files = [f for f in os.listdir(network_folder_path) if f.endswith('.csv') and "nosinglt" in f]

            if not network_files:
                # Case where no network is available
                nonet_metrics = create_metrics_inconsistent(threshold, network_type, biome, 'No network available')
                metrics_list.append(nonet_metrics)
                continue

            for file in network_files:
                file_path = os.path.join(network_folder_path, file)
                G = create_networkx_graph(file_path)
                
                if G.number_of_edges() <= 3:
                    n_nodes = G.number_of_nodes()
                    n_edges = G.number_of_edges()
                    smallnet_metrics = create_metrics_inconsistent(threshold, network_type, biome, 'Network too small (<= 3 edges)', n_nodes, n_edges)
                    metrics_list.append(smallnet_metrics)
                    # Skip networks with fewer than 3 edges
                    continue

                network_metrics = calculate_network_metrics(G)
                # Prepend Threshold, NetworkType, and Biome to the metrics
                metrics = {
                    "Threshold": threshold,
                    "NetworkType": 'cclasso',
                    "Biome": biome,
                    **network_metrics  # Merge the calculated metrics
                }
                metrics_list.append(metrics)

                if network_metrics['Modularity'] > 0:
                    # Keep track of the top three networks by modularity
                    top_networks_by_modularity.append((network_metrics['Modularity'], network_metrics['Avg_clustering_coefficient'], G, threshold))
                    top_networks_by_modularity = sorted(top_networks_by_modularity, reverse=True, key=lambda x: x[0])[:3]

    # Select the best network among the top three by modularity, based on the highest clustering coefficient
    best_network = max(top_networks_by_modularity, key=lambda x: x[1], default=(None, None, None, None))
    best_modularity, best_avg_clustering, best_network_graph, best_threshold = best_network
    
    
    return pd.DataFrame(metrics_list), best_network_graph, best_threshold

def process_all_biomes(base_dir, network_type, output_dir):
    """
    Process each biome and return a unified DataFrame containing the metrics for all biomes.
    This unified DataFrame is then saved as a CSV file.
    """
    all_biomes_metrics = []
    biome_best_networks = {}

    for biome in os.listdir(base_dir):
        biome_path = os.path.join(base_dir, biome)
        if os.path.isdir(biome_path):
            df_biome, best_network_graph, best_threshold = process_biome_networks(biome, biome_path, network_type)
            if not df_biome.empty:
                all_biomes_metrics.append(df_biome)
                biome_best_networks[biome] = (best_network_graph, best_threshold)
                print(f"Calculated metrics for biome: {biome}")

    # Combine all biome DataFrames into one unified DataFrame
    combined_df = pd.concat(all_biomes_metrics, ignore_index=True)

    # Save the unified DataFrame as a CSV file
    output_csv_path = os.path.join(output_dir, f"{network_type}_clust_net_nosinglet_metrics_sp_bybiomeandexptype.csv")
    combined_df.to_csv(output_csv_path, index=False)
    print(f"Unified network metrics saved to {output_csv_path}")

    return combined_df, biome_best_networks

# %%
# Base and output directory
base_dir = 'Data/species_nets_bybiome_and_exptype'
output_dir = 'Output/species_nets_bybiome_and_exptype'

# Process all studies
clust_metrics, biome_best_networks = process_all_biomes(base_dir, 'cclasso', output_dir)

# Export node clustering information for the best network of each biome
for biome, (best_network_graph, best_threshold) in biome_best_networks.items():
    if best_network_graph:
        communities = nx.algorithms.community.louvain_communities(best_network_graph, seed=12)
        node_community_dict = {node: cid for cid, community in enumerate(communities) for node in community}
        df_node_community = pd.DataFrame(node_community_dict.items(), columns=['Node', 'Community'])
        #df_node_community.to_csv(os.path.join(output_dir, "Best_nets/Clust",f'{biome}_best_net_{best_threshold}_node_clustering.csv'), index=False)
# %%
# Filter rows that has "Network too small (<= 3 edges)" in any of the columns
clust_metrics = clust_metrics[~clust_metrics.eq('Network too small (<= 3 edges)').any(axis=1)]
clust_metrics = clust_metrics[~clust_metrics.eq('No network available').any(axis=1)]

# Filter the df metrics to keep just clustering-related metrics
clust_metrics_filtered = clust_metrics[['Biome', 'Threshold', 'Nodes', 'Edges']]

# Set the aesthetic style of the plots
sns.set_style("whitegrid")

# Choose a varied color palette
palette = sns.color_palette("husl", n_colors=len(clust_metrics_filtered['Biome'].unique()))

# Define the metrics and their improved names
metrics = ['Nodes', 'Edges']

metric_name_mapping = {
    'Nodes': 'Number of Nodes',
    'Edges': 'Number of Edges'
}

# Biome name replacements
biome_replacements = {
    "Wastewater_assembly": "Wastewater - Assembly",
    "Wastewater_metagenomic": "Wastewater - Metagenomic",
    "Wastewater_metatranscriptomic": "Wastewater - Metatranscriptomic",
    "Wastewater_Activated_Sludge_assembly": "Activated Sludge - Assembly",
    "Wastewater_Activated_Sludge_metagenomic": "Activated Sludge - Metagenomic",
    "Wastewater_Activated_Sludge_metatranscriptomic": "Activated Sludge - Metatranscriptomic",
    "Wastewater_Industrial_wastewater_metagenomic": "Industrial Wastewater - Metagenomic",
    "Wastewater_Water_and_sludge_metagenomic": "Water and Sludge - Metagenomic",
}

# Replace biome names in the data
data_renamed_biomes = clust_metrics_filtered.replace({"Biome": biome_replacements})

# Function to format y-axis ticks
def format_ticks_adjusted(value, pos):
    if value >= 1000000:
        return f'{int(value/1000000)}M'
    elif value >= 1000:
        return f'{int(value/1000)}k'
    else:
        return int(value)

# Metrics that need formatted y-axis ticks
metrics_needing_formatting = ['Nodes', 'Edges']

# Font size configuration
axis_label_font_size = 20 
tick_label_font_size = 18
legend_font_size = 20   

# Adjusting the figure layout
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))

# Convert 'Threshold' column to numeric
data_renamed_biomes['Threshold'] = pd.to_numeric(data_renamed_biomes['Threshold'], errors='coerce')

# Determine the range for x-axis ticks outside the loop
x_min = min(data_renamed_biomes['Threshold'])
x_max = max(data_renamed_biomes['Threshold'])

# Create a list of ticks from x_min to x_max at intervals of 0.1
x_ticks = np.arange(x_min, x_max + 0.1, 0.1)

# Plotting data
for i, metric in enumerate(metrics):
    for biome in data_renamed_biomes['Biome'].unique():
        sns.lineplot(ax=axes.flatten()[i], x='Threshold', y=metric, 
                     data=data_renamed_biomes[data_renamed_biomes['Biome'] == biome], 
                     marker='o', markersize = 9, linewidth = 2.5, 
                     color=palette[data_renamed_biomes['Biome'].unique().tolist().index(biome)])
    axes.flatten()[i].set_ylabel(metric_name_mapping[metric], fontsize=axis_label_font_size)
    axes.flatten()[i].set_xlabel('Association Threshold', fontsize=axis_label_font_size)
    axes.flatten()[i].tick_params(axis='both', labelsize=tick_label_font_size)
    axes.flatten()[i].set_xticks(x_ticks)
    
    if metric in metrics_needing_formatting:
        axes.flatten()[i].yaxis.set_major_formatter(ticker.FuncFormatter(format_ticks_adjusted))

# Creating legend handles according to the order in biome_replacements
legend_handles = []
for original_name, pretty_name in biome_replacements.items():
    color_index = data_renamed_biomes['Biome'].unique().tolist().index(pretty_name)
    line = plt.Line2D([], [], color=palette[color_index], marker='o', markersize=10, 
                      linestyle='-', linewidth=3, label=pretty_name)
    legend_handles.append(line)

# Define the font properties for the legend title
title_font = FontProperties(weight='bold', size=legend_font_size)

# Adding the legend above the plots
fig.legend(handles=legend_handles, loc='upper center', title='Biome - Experiment type', ncol = 3,
           bbox_to_anchor=(0.5, 1), fontsize=legend_font_size, title_fontproperties=title_font)

# Adjust layout
plt.tight_layout(rect=[0, 0, 1, 0.95])

# Saving the plots as both PNG and SVG
fig.savefig("Output/species_nets_bybiome_and_exptype/cclasso_wwt_clustnet_nosinglet_nodesedges_sp_bybiome_andextype.png", format='png')
fig.savefig("Output/species_nets_bybiome_and_exptype/cclasso_wwt_clustnet_nosinglet_nodesedges_sp_bybiome_andextype.svg", format='svg')
# %%
# Filter rows that has "Network too small (<= 3 edges)" in any of the columns
clust_metrics = clust_metrics[~clust_metrics.eq('Network too small (<= 3 edges)').any(axis=1)]
clust_metrics = clust_metrics[~clust_metrics.eq('No network available').any(axis=1)]

# Filter the df metrics to keep just clustering-related metrics
clust_metrics_filtered = clust_metrics[['Biome', 'Threshold',
                                        'Number_communities', 'Avg_degree', 
                                        'Modularity', 'Avg_clustering_coefficient']]

# Set the aesthetic style of the plots
sns.set_style("whitegrid")

# Choose a varied color palette
palette = sns.color_palette("husl", n_colors=len(clust_metrics_filtered['Biome'].unique()))

# Define the metrics and their improved names
metrics = ['Number_communities', 'Avg_degree', 
           'Modularity', 'Avg_clustering_coefficient']

metric_name_mapping = {
    'Number_communities': 'Number of Communities',
    'Avg_degree': 'Average Degree',
    'Modularity': 'Modularity',
    'Avg_clustering_coefficient': 'Average Clustering Coefficient'
}

# Biome name replacements
biome_replacements = {
    "Wastewater_assembly": "Wastewater - Assembly",
    "Wastewater_metagenomic": "Wastewater - Metagenomic",
    "Wastewater_metatranscriptomic": "Wastewater - Metatranscriptomic",
    "Wastewater_Activated_Sludge_assembly": "Activated Sludge - Assembly",
    "Wastewater_Activated_Sludge_metagenomic": "Activated Sludge - Metagenomic",
    "Wastewater_Activated_Sludge_metatranscriptomic": "Activated Sludge - Metatranscriptomic",
    "Wastewater_Industrial_wastewater_metagenomic": "Industrial Wastewater - Metagenomic",
    "Wastewater_Water_and_sludge_metagenomic": "Water and Sludge - Metagenomic",
}

# Replace biome names in the data
data_renamed_biomes = clust_metrics_filtered.replace({"Biome": biome_replacements})

# Font size configuration
axis_label_font_size = 20 
tick_label_font_size = 18
legend_font_size = 20   

# Adjusting the figure layout
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(20, 15))

# Convert 'Threshold' column to numeric
data_renamed_biomes['Threshold'] = pd.to_numeric(data_renamed_biomes['Threshold'], errors='coerce')

# Determine the range for x-axis ticks outside the loop
x_min = min(data_renamed_biomes['Threshold'])
x_max = max(data_renamed_biomes['Threshold'])

# Create a list of ticks from x_min to x_max at intervals of 0.1
x_ticks = np.arange(x_min, x_max + 0.1, 0.1)

# Plotting data
for i, metric in enumerate(metrics):
    for biome in data_renamed_biomes['Biome'].unique():
        sns.lineplot(ax=axes.flatten()[i], x='Threshold', y=metric, 
                     data=data_renamed_biomes[data_renamed_biomes['Biome'] == biome], 
                     marker='o', markersize = 9, linewidth = 2.5, 
                     color=palette[data_renamed_biomes['Biome'].unique().tolist().index(biome)])
    axes.flatten()[i].set_ylabel(metric_name_mapping[metric], fontsize=axis_label_font_size)
    axes.flatten()[i].set_xlabel('Association Threshold' if i >= 2 else '', fontsize=axis_label_font_size)
    axes.flatten()[i].tick_params(axis='both', labelsize=tick_label_font_size)
    axes.flatten()[i].set_xticks(x_ticks)

# Creating legend handles according to the order in biome_replacements
legend_handles = []
for original_name, pretty_name in biome_replacements.items():
    color_index = data_renamed_biomes['Biome'].unique().tolist().index(pretty_name)
    line = plt.Line2D([], [], color=palette[color_index], marker='o', markersize=10, 
                      linestyle='-', linewidth=3, label=pretty_name)
    legend_handles.append(line)

# Define the font properties for the legend title
title_font = FontProperties(weight='bold', size=legend_font_size)

# Adding the legend above the plots
fig.legend(handles=legend_handles, loc='upper center', title='Biome - Experiment type', ncol = 3,
           bbox_to_anchor=(0.5, 1.02), fontsize=legend_font_size, title_fontproperties=title_font)

# Adjust layout
plt.tight_layout(rect=[0, 0, 1, 0.95])

# Saving the plots as both PNG and SVG
fig.savefig("Output/species_nets_bybiome_and_exptype/cclasso_wwt_clustnet_nosinglet_metrics_plots_sp_bybiome_andextype.png", format='png')
fig.savefig("Output/species_nets_bybiome_and_exptype/cclasso_wwt_clustnet_nosinglet_metrics_plots_sp_bybiome_andextype.svg", format='svg')

# %%
