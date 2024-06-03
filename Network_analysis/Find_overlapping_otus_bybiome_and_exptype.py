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
    networks = {}  # Dictionary to store networks: {study_id_biome: network_data}
    for network_file in os.listdir(path):
        # Extracting study_id and biome from the filename
        parts = network_file.split('_')
        biome = parts[0]
        exptype = parts[1]
        biome_exptype = f"{biome}_{exptype}"  # First study_id then biome
        network_path = os.path.join(path, network_file)
        networks[biome_exptype] = pd.read_csv(network_path)
    return networks

def extract_otus(network_data):
    """
    Extract OTUs from network data.
    """
    all_otus = set(network_data['v1']).union(set(network_data['v2']))
    return all_otus

def calculate_coincidences(otus):
    """
    Calculate pairwise OTU coincidences between networks, including the total OTUs in each network on the main diagonal.
    """
    study_ids = list(otus.keys())
    coincidence_matrix = pd.DataFrame(index=study_ids, columns=study_ids, data=0)

    for i, study_id1 in enumerate(study_ids):
        for j, study_id2 in enumerate(study_ids):
            if i == j:
                # Diagonal: number of unique OTUs in the network
                coincidence_matrix.at[study_id1, study_id2] = len(otus[study_id1])
            else:
                # Off-diagonal: number of common OTUs between different studies
                common_otus = otus[study_id1].intersection(otus[study_id2])
                coincidence_matrix.at[study_id1, study_id2] = len(common_otus)
                coincidence_matrix.at[study_id2, study_id1] = len(common_otus)
    
    return coincidence_matrix

def plot_heatmap(otu_coincidence_matrix, title, file_path):
    """
    Plot and save a heatmap of the upper triangle of the community coincidence matrix with a blue-to-red color palette.
    """
    plt.figure(figsize=(10, 8))
    mask = np.triu(np.ones_like(otu_coincidence_matrix, dtype=bool))

    # Blue-to-red color palette
    cmap = sns.diverging_palette(240, 10, as_cmap=True)

    sns.heatmap(otu_coincidence_matrix, mask=mask, annot=True, cmap=cmap, fmt='g')
    plt.title(title)
    plt.xlabel("Study ID")
    plt.ylabel("Study ID")

    plt.tight_layout()  # Adjust the layout
    plt.savefig(file_path)
    plt.close()

def process_and_plot_biomes(base_dir, output_dir):
    """
    Process each biome to generate OTU coincidence matrices and save them as CSV files.
    """
    for biome in ['ActivatedSludge', 'Wastewater']:
        highest_nets_path = os.path.join(base_dir, biome)
        if os.path.exists(highest_nets_path):
            networks = load_networks(highest_nets_path)
            otus = {net_id: extract_otus(data) for net_id, data in networks.items()}

            # Calculate and save the individual biome matrix
            biome_matrix = calculate_coincidences(otus)
            biome_output_path = os.path.join(output_dir, f"{biome}_otu_coincidence_matrix_mod_avgclust_nosinglet.csv")
            biome_matrix.to_csv(biome_output_path)
            print(f"OTU coincidence matrix for biome {biome} saved to {biome_output_path}")

            # Plot the heatmap for the current biome
            plot_title = f"{biome} OTU Coincidence Heatmap"
            plot_path = os.path.join(output_dir, f"{biome}_otu_coincidence_heatmap.png")
            plot_heatmap(biome_matrix, plot_title, plot_path)
            print(f"Heatmap for {biome} saved to {plot_path}")

#%%
base_dir = 'Output/species_nets_bybiome_and_exptype/Best_nets/Nets'
output_dir = 'Output/species_nets_bybiome_and_exptype/Best_nets/Nets/otu_overlap' 
otu_coincidence_matrix = process_and_plot_biomes(base_dir, output_dir)

#%%
