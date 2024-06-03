#%%
import pandas as pd
import networkx as nx
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

# Function to read the edge list and create a networkx graph
def create_networkx_graph(file_path):
    df = pd.read_csv(file_path)
    G = nx.from_pandas_edgelist(df, source='v1', target='v2', edge_attr='asso', create_using=nx.Graph())
    return G

# Function to extract the giant component from a networkx graph
def get_giant_component(G):
    giant_component = max(nx.connected_components(G), key=len)
    G_giant = G.subgraph(giant_component).copy()
    return G_giant

# Function to transform the weights of a networkx graph to absolute values
def transform_weights_to_absolute(G):
    # Creating a new graph with absolute weights
    G_abs = G.copy()
    for u, v, d in G_abs.edges(data=True):
        d['asso'] = abs(d['asso'])
    return G_abs

# Function to calculate general topological metrics for a networkx graph
def calculate_network_metrics(G):
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()

    # Using Louvain method for community detection
    communities = list(nx.algorithms.community.louvain_communities(G, weight='asso'))
    num_communities = len(communities)

    # Calculate modularity
    modularity = nx.algorithms.community.modularity(G, communities)

    # Calculate average path length and diameter
    if nx.is_connected(G):
        avg_path_length = nx.average_shortest_path_length(G, weight='asso')
        diameter = nx.diameter(G, e=None, usebounds=False, weight='asso')
    else:
        avg_path_length = float('inf')
        diameter = float('inf')

    # Calculate degree distribution, average degree, average clustering coefficient, and density
    degrees = [deg for node, deg in G.degree(weight='asso')]

    # Convert the list of degrees to a df
    df_degrees = pd.DataFrame(degrees, columns=['Degree'])
    avg_degree = np.mean(degrees)
    avg_clustering_coefficient = nx.average_clustering(G, weight='asso')
    density = nx.density(G)

    return {
        "Nodes": num_nodes,
        "Edges": num_edges,
        "Number_communities": num_communities,
        "Modularity": modularity,
        "Avg_path_length": avg_path_length,
        "Diameter": diameter,
        "Avg_degree": avg_degree,
        "Avg_clustering_coefficient": avg_clustering_coefficient,
        "Density": density
    }, df_degrees

#%%
# Path to the edge list file
file_path = 'Data/Wastewater/cclasso_0.50/Wastewater_net_cclasso_050_edgelist.csv'

# Create a networkx graph from the edge list
G = create_networkx_graph(file_path)

# Transform the weights of the network to absolute values
G_abs = transform_weights_to_absolute(G)

# Extract the giant component
G_giant_abs = get_giant_component(G_abs)

# Calculate metrics
metrics, degree_distrib = calculate_network_metrics(G_giant_abs)

# Convert the metrics to a DataFrame
df_metrics = pd.DataFrame([metrics])

df_metrics
degree_distrib
# %%
# Defining biomes and thresholds
biomes = ["Wastewater_Activated_Sludge", "Wastewater_Industrial_wastewater", "Wastewater", "Wastewater_Water_and_sludge"]
thresholds = [0.1, 0.2, 0.3, 0.4, 0.45, 0.47, 0.5]
network_type = "cclasso"

# DataFrame to store the results for all networks
df_all_networks = pd.DataFrame()

# Iterating over each biome and threshold combination
for biome in biomes:
    for thresh in thresholds:
        # Formatting the threshold value for the folder and file name
        thresh_formatted_folder = f"{thresh:.2f}"
        thresh_formatted_file = f"0{thresh:.2f}".replace('0.', '') if thresh < 1 else f"{thresh:.2f}"

        # Construct file path for the network file and output files
        folder_name = f"{network_type}_{thresh_formatted_folder}"
        net_file_name = f"{biome}_net_{network_type}_{thresh_formatted_file}_edgelist.csv"
        net_path = os.path.join("Data", biome, folder_name, net_file_name)
        degree_distrib_file_name = f"{biome}_{network_type}_{thresh_formatted_file}_degree_distribution.csv"
        degree_distrib_path = os.path.join("Output/Degree_distrib", biome, degree_distrib_file_name)

        # Check if the network file exists
        if os.path.exists(net_path):
            # Create a networkx graph from the edge list
            G = create_networkx_graph(net_path)

            # Transform the weights of the network to absolute values
            G_abs = transform_weights_to_absolute(G)

            # Extract the giant component
            G_giant_abs = get_giant_component(G_abs)

            # Calculate metrics
            metrics, degree_distrib = calculate_network_metrics(G_giant_abs)

            # Save the degree distribution to a CSV file
            degree_distrib.to_csv(degree_distrib_path, index=False)

            # Create a DataFrame from metrics and add columns for biome, threshold, and network type
            metrics_df = pd.DataFrame([metrics])
            metrics_df['Network_type'] = network_type
            metrics_df['Biome'] = biome
            metrics_df['Threshold'] = thresh

            # Reorder columns to have network type, biome, and threshold first
            cols = ['Network_type', 'Biome', 'Threshold'] + [col for col in metrics_df.columns if col not in ['Network_type', 'Biome', 'Threshold']]
            metrics_df = metrics_df[cols]

            # Round the DataFrame to three decimal places
            metrics_df = metrics_df.round(4)

            # Append to the all-networks DataFrame
            df_all_networks = pd.concat([df_all_networks, metrics_df], ignore_index=True)
        else:
            print(f"Network file not found: {net_path}")

# Save the all-networks DataFrame as a CSV file
save_file_name_all = f"Output/{network_type}_all_biomes_network_metrics.csv"
df_all_networks.to_csv(save_file_name_all, index=False)

print(f"All networks metrics saved to {save_file_name_all}")
# %%
# Load the all-networks DataFrame from the CSV file
save_file_name_all = f"Output/cclasso_all_biomes_network_metrics.csv"
data = pd.read_csv(save_file_name_all)

# Set the aesthetic style of the plots
sns.set_style("whitegrid")

# Choose a varied color palette
palette = sns.color_palette("husl", n_colors=len(data['Biome'].unique()))

# Define the metrics and their improved names
metrics = ['Nodes', 'Edges', 'Number_communities', 'Modularity', 'Avg_path_length', 
           'Diameter', 'Avg_degree', 'Avg_clustering_coefficient', 'Density']
metric_name_mapping = {
    'Nodes': 'Number of Nodes',
    'Edges': 'Number of Edges',
    'Number_communities': 'Number of Communities',
    'Modularity': 'Modularity',
    'Avg_path_length': 'Average Path Length',
    'Diameter': 'Diameter',
    'Avg_degree': 'Average Degree',
    'Avg_clustering_coefficient': 'Average Clustering Coefficient',
    'Density': 'Density'
}

# Biome name replacements
biome_replacements = {
    "Wastewater": "Wastewater",
    "Wastewater_Industrial_wastewater": "Industrial wastewater",
    "Wastewater_Water_and_sludge": "Water and sludge",
    "Wastewater_Activated_Sludge": "Activated Sludge"
}

# Replace biome names in the data
data_renamed_biomes = data.replace({"Biome": biome_replacements})

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
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(20, 20))  # More balanced aspect ratio

# Plotting data
for i, metric in enumerate(metrics):
    for biome in data_renamed_biomes['Biome'].unique():
        sns.lineplot(ax=axes.flatten()[i], x='Threshold', y=metric, 
                     data=data_renamed_biomes[data_renamed_biomes['Biome'] == biome], 
                     marker='o', markersize = 9, linewidth = 2.5, 
                     color=palette[data_renamed_biomes['Biome'].unique().tolist().index(biome)])
    axes.flatten()[i].set_ylabel(metric_name_mapping[metric], fontsize=axis_label_font_size)
    axes.flatten()[i].set_xlabel('Association Threshold' if i // 3 == 2 else '', fontsize=axis_label_font_size)
    axes.flatten()[i].tick_params(axis='both', labelsize=tick_label_font_size)
    
    if metric in metrics_needing_formatting:
        axes.flatten()[i].yaxis.set_major_formatter(ticker.FuncFormatter(format_ticks_adjusted))

# Creating legend handles
legend_handles = [plt.Line2D([], [], color=palette[i], marker='o', markersize = 10, 
                             linestyle='-', linewidth = 3, label=biome) for i, biome in enumerate(data_renamed_biomes['Biome'].unique())]

# Adding the legend above the plots
fig.legend(handles=legend_handles, loc='upper center', title='Biome', 
           ncol=len(data_renamed_biomes['Biome'].unique()), bbox_to_anchor=(0.5, 1.0), 
                    fontsize=legend_font_size, title_fontsize=legend_font_size)

# Adjust layout
plt.tight_layout(rect=[0, 0, 1, 0.95])

# Saving the plots as both PNG and SVG
fig.savefig("Output/cclasso_wwt_net_metrics_plots.png", format='png')
fig.savefig("Output/cclasso_wwt_net_metrics_plots.svg", format='svg')
# %%
