#%%
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def load_networks(path):
    """
    Load networks from csv files in the given path.
    """
    networks = {}
    for network_file in os.listdir(path):
        parts = network_file.split('_')
        biome = parts[0]
        exptype = parts[1]
        biome_exptype = f"{biome}_{exptype}"
        network_path = os.path.join(path, network_file)
        networks[biome_exptype] = pd.read_csv(network_path)[['v1', 'v2']]
    return networks

def compare_network_edges(networks):
    """
    Compare edges between all pairs of networks.
    """
    all_edges = {}
    for study_id_biome, network_data in networks.items():
        edges = set(map(frozenset, network_data[['v1', 'v2']].values))  # Use frozenset for undirected edge comparison
        all_edges[study_id_biome] = edges
    
    study_ids = list(all_edges.keys())
    edge_coincidence_matrix = pd.DataFrame(index=study_ids, columns=study_ids, data=0)

    for i, study_id1 in enumerate(study_ids):
        edges1 = all_edges[study_id1]
        for j, study_id2 in enumerate(study_ids):
            if i < j:
                edges2 = all_edges[study_id2]
                common_edges = len(edges1.intersection(edges2))
                edge_coincidence_matrix.at[study_id1, study_id2] = common_edges
                edge_coincidence_matrix.at[study_id2, study_id1] = common_edges
            elif i == j:
                edge_coincidence_matrix.at[study_id1, study_id2] = len(edges1)  # Count of unique edges in the network
    
    return edge_coincidence_matrix

def plot_heatmap(edge_coincidence_matrix, title, file_path):
    """
    Plot and save a heatmap of the upper triangle of the community coincidence matrix with a blue-to-red color palette.
    """
    plt.figure(figsize=(10, 8))
    mask = np.triu(np.ones_like(edge_coincidence_matrix, dtype=bool))

    # Blue-to-red color palette
    cmap = sns.diverging_palette(240, 10, as_cmap=True)

    sns.heatmap(edge_coincidence_matrix, mask=mask, annot=True, cmap=cmap, fmt='g')
    plt.title(title)
    plt.xlabel("Study ID")
    plt.ylabel("Study ID")

    plt.tight_layout()  # Adjust the layout
    plt.savefig(file_path)
    plt.close()

def process_and_plot_biomes(base_dir, output_dir):
    """
    Process each biome to generate edge coincidence matrices and save them as CSV files.
    """
    for biome in ['ActivatedSludge', 'Wastewater']:
        highest_nets_path = os.path.join(base_dir, biome)
        if os.path.exists(highest_nets_path):
            networks = load_networks(highest_nets_path)

            # Calculate and save the individual biome matrix
            biome_matrix = compare_network_edges(networks)
            biome_output_path = os.path.join(output_dir, f"{biome}_edge_coincidence_matrix_mod_avgclust_nosinglet.csv")
            biome_matrix.to_csv(biome_output_path)
            print(f"Edge coincidence matrix for biome {biome} saved to {biome_output_path}")

            # Plot the heatmap for the current biome
            plot_title = f"{biome} Edge Coincidence Heatmap"
            plot_path = os.path.join(output_dir, f"{biome}_edge_coincidence_heatmap.png")
            plot_heatmap(biome_matrix, plot_title, plot_path)
            print(f"Heatmap for {biome} saved to {plot_path}")


#%% Main execution
base_dir = 'Output/species_nets_bybiome_and_exptype/Best_nets/Nets'
output_dir = 'Output/species_nets_bybiome_and_exptype/Best_nets/Nets/edge_overlap'
edge_coincidence_matrix = process_and_plot_biomes(base_dir, output_dir)
# %%
