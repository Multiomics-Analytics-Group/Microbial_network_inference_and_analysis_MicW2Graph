#%%
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import re
#%%
# Load the GraphML files
species_graph = nx.read_graphml('./Data/KG/Species_functional_nodes.graphml')
genus_graph = nx.read_graphml('./Data/KG/Genus_functional_nodes.graphml')

# Set the aesthetic style of the plots
sns.set_style("whitegrid")

# Choose a varied color palette
#palette = sns.color_palette("husl", n_colors=5)

# Calculate the out-degree for each node
species_out_degrees = [d for n, d in species_graph.out_degree()]
genus_out_degrees = [d for n, d in genus_graph.out_degree()]

# Filter out nodes with degree of 0
species_out_degrees_filtered = [d for d in species_out_degrees if d > 0]
genus_out_degrees_filtered = [d for d in genus_out_degrees if d > 0]

# Create histograms for the filtered out-degrees
plt.figure(figsize=(15, 7))

plt.subplot(1, 2, 1)
sns.histplot(species_out_degrees_filtered, bins=10, alpha=0.7, color = "#e3cd72ff")
plt.xlabel('Degree to functional nodes', fontsize=20)
plt.ylabel('Number of species', fontsize=20)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(5))
plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(20))
plt.grid(False)

plt.subplot(1, 2, 2)
sns.histplot(genus_out_degrees_filtered, bins=10, alpha=0.7, color = "#7ad1ffff", discrete=True)
plt.xlabel('Degree to functional nodes', fontsize=20)
plt.ylabel('Number of genera', fontsize=20)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(1))
plt.gca().yaxis.get_major_locator().set_params(integer=True)
plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(2))
plt.grid(False)

plt.tight_layout()

# Saving the plots as both PNG and SVG
plt.savefig("Output/Kg/Histograms_species_and_genera_to_funct_nodes.png", format='png')
plt.savefig("Output/Kg/Histograms_species_and_genera_to_funct_nodes.svg", format='svg')
# %%
# Define custom colors for each target node label from the provided configuration
target_node_colors = {
    'ActivityAndBehavior': '#77b3ddc9',
    'BiologicalProcess': '#cdfb98ff',
    'ChemicalEntity': '#e9aaaaff',
    'EnvironmentalFeature': '#ecb5ebff',
    'PhenotypicFeature': '#f3ab8eff'
}

# Function to insert spaces before uppercase letters in camel case strings
def camel_case_to_spaces(s):
    return re.sub(r'(?<!^)(?=[A-Z])', ' ', s)

# Define a function to categorize nodes and calculate degrees using the target node names and their colors
def categorize_and_calculate_degrees_target_with_colors(graph):
    data = []
    for u, v, edge_attrs in graph.edges(data=True):
        out_degree = graph.out_degree(u)
        in_degree = graph.in_degree(v)
        if 'label' in edge_attrs:
            relationship_type = edge_attrs['label']
            target_node_label = graph.nodes[v]['labels'].split(':')[-1].replace('_', ' ')  # Replace underscores with spaces
            color = target_node_colors.get(target_node_label, 'gray')
            data.append({
                'source_node': u,
                'target_node': v,
                'relationship_type': f"{relationship_type} ({target_node_label})",
                'out_degree': out_degree,
                'in_degree': in_degree,
                'color': color
            })
    df = pd.DataFrame(data)
    return df

# Categorize nodes and calculate degrees
species_df_target_colors = categorize_and_calculate_degrees_target_with_colors(species_graph)

# Create a list to store the relationship types and corresponding colors
relationship_colors = []

relationship_types = species_df_target_colors['relationship_type'].unique()
for relationship_type in relationship_types:
    subset = species_df_target_colors[species_df_target_colors['relationship_type'] == relationship_type]
    color = subset['color'].iloc[0]  # Get the color for this relationship type
    relationship_colors.append((relationship_type, color))

# Update the dataframe to have relationship labels and target node names for the x-axis in two lines
species_df_target_colors['combined_label'] = species_df_target_colors.apply(
    lambda row: f"{row['relationship_type'].split(' ')[0]}\n({camel_case_to_spaces(species_graph.nodes[row['target_node']]['labels'].split(':')[-1])})", axis=1)

# Plot all violins in the same plot with combined labels on the x-axis in two lines
plt.figure(figsize=(20, 10))
sns.violinplot(x='combined_label', y='out_degree', data=species_df_target_colors, inner='box', palette=[color for _, color in relationship_colors])
plt.xlabel('Relationship label (Target node label)', fontsize=20)
plt.ylabel('Species degree to functional nodes', fontsize=20)
plt.xticks(rotation=0, ha='center', fontsize=18) 
plt.yticks(fontsize=18)
plt.grid(False)

plt.tight_layout()

# Saving the plots as both PNG and SVG
plt.savefig("Output/Kg/Violin_plots_species_deg_to_funct_nodes.png", format='png')
plt.savefig("Output/Kg/Violin_plots_species_deg_to_funct_nodes.svg", format='svg')
# %%
# Plot all box plots in the same plot with combined labels on the x-axis in two lines
plt.figure(figsize=(20, 10))
sns.boxplot(x='combined_label', y='out_degree', data=species_df_target_colors, palette=[color for _, color in relationship_colors])
plt.xlabel('Relationship label (Target node label)', fontsize=20)
plt.ylabel('Species degree to functional nodes', fontsize=20)
plt.xticks(rotation=0, ha='center', fontsize=18) 
plt.yticks(fontsize=18)
plt.grid(False)

plt.tight_layout()

# Saving the plots as both PNG and SVG
plt.savefig("Output/Kg/Box_plots_species_deg_to_funct_nodes.png", format='png')
plt.savefig("Output/Kg/Box_plots_species_deg_to_funct_nodes.svg", format='svg')
# %%
