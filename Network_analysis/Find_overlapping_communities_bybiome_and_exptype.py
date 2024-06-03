#%%
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def get_louvain_communities(G):
    """
    Calculate communities in a graph using the Louvain method.
    """
    communities = nx.algorithms.community.louvain_communities(G, weight='asso', seed=12)
    return communities

def load_networks_and_compute_communities(highest_nets_path):
    """
    Load networks and compute communities, returning both the communities and the network graphs.
    """
    communities_dict = {}
    networks_dict = {}
    for network_file in os.listdir(highest_nets_path):
        parts = network_file.split('_')
        biome = parts[0]
        exptype = parts[1]
        biome_exptype = f"{biome}_{exptype}"
        network_path = os.path.join(highest_nets_path, network_file)
        df = pd.read_csv(network_path)
        G = nx.from_pandas_edgelist(df, source='v1', target='v2', edge_attr='asso')

        communities = nx.algorithms.community.louvain_communities(G, weight='asso', seed=12)
        communities_dict[biome_exptype] = communities
        networks_dict[biome_exptype] = G  # Storing the graph in the dictionary

    return communities_dict, networks_dict

def calculate_community_coincidences(communities_dict, networks_dict):
    """
    Calculate pairwise community coincidences between studies based on shared OTUs.
    """
    study_ids = list(communities_dict.keys())
    community_coincidence_matrix = pd.DataFrame(index=study_ids, columns=study_ids, data=0)

    # Mapping each OTU to its community
    otu_community_mapping = {}
    for study_id, communities in communities_dict.items():
        mapping = {}
        for i, community in enumerate(communities):
            for otu in community:
                mapping[otu] = i
        otu_community_mapping[study_id] = mapping

    for study_id in study_ids:
        for other_study_id in study_ids:
            if study_id != other_study_id:
                # Retrieve nodes (OTUs) for both networks and find shared OTUs
                G1_nodes = set(networks_dict[study_id].nodes())
                G2_nodes = set(networks_dict[other_study_id].nodes())
                shared_otus = G1_nodes.intersection(G2_nodes)

                # Count shared community memberships
                shared_count = 0
                for otu in shared_otus:
                    if (otu in otu_community_mapping[study_id] and 
                        otu in otu_community_mapping[other_study_id] and 
                        otu_community_mapping[study_id][otu] == otu_community_mapping[other_study_id][otu]):
                        shared_count += 1
                community_coincidence_matrix.at[study_id, other_study_id] = shared_count

        # Set diagonal values to zero or a predetermined value
        community_coincidence_matrix.at[study_id, study_id] = 0  # or another appropriate value

    return community_coincidence_matrix

def plot_heatmap(community_coincidence_matrix, title, file_path):
    """
    Plot and save a heatmap of the upper triangle of the community coincidence matrix with a blue-to-red color palette.
    """
    plt.figure(figsize=(10, 8))
    
    # Replace NaN values with 0 or an appropriate value
    community_coincidence_matrix = community_coincidence_matrix.fillna(0)
    mask = np.triu(np.ones_like(community_coincidence_matrix, dtype=bool))

    # Set the colorbar range based on the data range, or explicitly define if needed
    vmax = community_coincidence_matrix.max().max()
    vmin = 0

    # Blue-to-red color palette
    cmap = sns.diverging_palette(240, 10, as_cmap=True)

    sns.heatmap(community_coincidence_matrix, annot=True, cmap=cmap,
                fmt='g', vmin=vmin, vmax=vmax, mask=mask)
    plt.title(title)
    plt.xlabel("Study ID")
    plt.ylabel("Study ID")

    plt.tight_layout()  # Adjust the layout
    plt.savefig(file_path)
    plt.close()

def process_and_plot_biomes(base_dir, output_dir):
    """
    Process each biome to generate community coincidence matrices.
    """

    for biome in ['ActivatedSludge', 'Wastewater']:
        highest_nets_path = os.path.join(base_dir, biome)
        if os.path.exists(highest_nets_path):
            communities, networks = load_networks_and_compute_communities(highest_nets_path)

            biome_matrix = calculate_community_coincidences(communities, networks)
            biome_output_path = os.path.join(output_dir, f"{biome}_community_coincidence_matrix_mod_avgclust_nosinglet.csv")
            biome_matrix.to_csv(biome_output_path)
            print(f"Community coincidence matrix for biome {biome} saved to {biome_output_path}")

            # Plot the heatmap for the current biome
            plot_title = f"{biome} Community Coincidence Heatmap"
            plot_path = os.path.join(output_dir, f"{biome}_community_coincidence_heatmap.png")
            plot_heatmap(biome_matrix, plot_title, plot_path)
            print(f"Heatmap for {biome} saved to {plot_path}")

#%% Main execution
base_dir = 'Output/species_nets_bybiome_and_exptype/Best_nets/Nets'
output_dir = 'Output/species_nets_bybiome_and_exptype/Best_nets/Nets/community_overlap' 
community_coincidence_matrix = process_and_plot_biomes(base_dir, output_dir)

# %%