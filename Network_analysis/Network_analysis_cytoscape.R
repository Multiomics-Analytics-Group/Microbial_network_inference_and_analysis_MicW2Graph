library(RCy3)
library(igraph)
library(dplyr)
library(rstudioapi)
library(paletteer) 

# Set seed for reproducibility
set.seed(123)

# Define base path, biomes, network types, and thresholds
root_path <- "Output/Biomes_exptypes"
network_type <- "cclasso"
biomes <- c("Wastewater_Activated_Sludge", "Wastewater_Industrial_wastewater", 
            "Wastewater", "Wastewater_Water_and_sludge")
# biome <- "Wastewater_Industrial_wastewater"
# thresh <- 0.47
extypes <- c("assembly", "metagenomic", "metatranscriptomic")
biomes_extypes <- c("Wastewater_Activated_Sludge_assembly", "Wastewater_Activated_Sludge_metagenomic",
                    "Wastewater_Activated_Sludge_metatranscriptomic", "Wastewater_assembly",
                    "Wastewater_Industrial_wastewater_metagenomic","Wastewater_metagenomic", 
                    "Wastewater_metatranscriptomic", "Wastewater_Water_and_sludge_metagenomic")
thresholds <- seq(0.10, 0.70, by = 0.05)

# Define colors for positive and negative edges
positive_color <- "#ac1214"  # Red for positive
negative_color <- "#0d1fad"  # Blue for negative

# Loop through biomes and thresholds
for (biome in biomes_extypes) {
  for (thresh in thresholds) {
      # Construct file path
      folder_name <- sprintf("%s_%0.2f", network_type, thresh)
      folder_path <- file.path(root_path, biome, folder_name)
      net_path <- list.files(folder_path, pattern = "\\.csv$", full.names = TRUE)
      net_path <- net_path[grepl("nosinglt", net_path)]
      
      # Check if net_path is empty
      if (length(net_path) == 0) {
        cat("No network files found in folder: ", folder_path, "\n")
        next  # Skip to the next iteration
      }
      
      # Load network if it exists and has more than 3 edges
      if (file.exists(net_path)) {
        net <- read.csv(net_path)
        if (nrow(net) > 3) {
          cat("Network file found and has more than 3 edges: ", net_path, "\n")
          # Further processing can be done here
        } else {
          cat("Network file found but has 3 or fewer edges, skipping: ", net_path, "\n")
          next  # Skip to the next iteration
        }
      } else {
        cat("Network file not found: ", net_path, "\n")
        next  # Skip to the next iteration
      }
      
      # Convert to igraph object
      igraph_net <- graph_from_data_frame(net, directed = FALSE)
      
      # Add an attribute for edge sign
      E(igraph_net)$sign <- sign(E(igraph_net)$asso)
      
      # Calculate degree distribution and min-max
      V(igraph_net)$degree <- degree(igraph_net)
      min_degree <- min(V(igraph_net)$degree)
      max_degree <- max(V(igraph_net)$degree)
      
      # Calculate clusters
      clusters <- cluster_louvain(igraph_net) 
      V(igraph_net)$cluster <- clusters$membership
      
      # Generate a color palette
      num_clusters <- length(unique(V(igraph_net)$cluster))
      base_palette <- paletteer_d("ggsci::category20_d3", 20)
      base_palette <- substr(base_palette, 1, 7)
      
      # Extend the color palette by repeating it if necessary
      repeat_times <- ceiling(num_clusters / 20)
      color_palette <- rep(base_palette, times = repeat_times)[1:num_clusters]
      
      # Create a mapping between cluster number and color
      cluster_color_mapping <- setNames(color_palette, unique(V(igraph_net)$cluster))
      
      # Create a mapping between cluster number and color
      cluster_color_mapping <- setNames(color_palette, unique(V(igraph_net)$cluster))
      
      # Create network in Cytoscape
      RCy3::createNetworkFromIgraph(igraph_net, title=folder_name, collection=biome)
      
      # Get net ID
      networkSuid = getNetworkSuid()
      
      # Create a new visual style
      style_name <- paste0("Style_", folder_name, biome)
      createVisualStyle(style_name)
      setVisualStyle(style_name, network = networkSuid)
      
      # Set node size mapping based on degree
      RCy3::setNodeSizeMapping('degree', c(min_degree, round(max_degree * 0.2), max_degree), c(10, 20, 60), 
                               network = networkSuid, style.name = style_name)
      
      # Set node color mapping based on cluster
      RCy3::setNodeColorMapping('cluster', as.numeric(names(cluster_color_mapping)), mapping.type = "d",
                                cluster_color_mapping, network = networkSuid, style.name = style_name)
      
      # Set edge color mapping based on 'sign'
      RCy3::setEdgeColorMapping('sign', c(-1, 1), c(negative_color, positive_color),
                                network = networkSuid, style.name = style_name)
    }
  }

# Update default node shape and size
RCy3::setNodeShapeDefault('ellipse')
# RCy3::setNodeSizeDefault(20)

###### TEST ####################
biome <- "Wastewater_Industrial_wastewater"
thresh <- 0.4
folder_name <- sprintf("%s_%0.1f", network_type, thresh)
net_file_name <- sprintf("%s_net_%s_%02.0f_edgelist.csv", biome, network_type, thresh * 10)
net_path <- file.path(root_path, biome, folder_name, net_file_name)

# Load network if it exists
if (file.exists(net_path)) {
  net <- read.csv(net_path)
} else {
  cat("Network file not found: ", net_path, "\n")
  next  # Skip to the next iteration
}

# Convert to igraph object
igraph_net <- graph_from_data_frame(net, directed = FALSE)

# Calcuate degree distribution and min-max values
V(igraph_net)$degree <- degree(igraph_net)
min_degree <- min(V(igraph_net)$degree)
max_degree <- max(V(igraph_net)$degree)

# Apply a logarithmic transformation to degree values for node sizing
V(igraph_net)$log_degree <- log(V(igraph_net)$degree + 1)  # Adding 1 to avoid log(0)

# Calculate the range for log-transformed degree
min_log_degree <- min(V(igraph_net)$log_degree)
max_log_degree <- max(V(igraph_net)$log_degree)

# Calculate clusters
clusters <- cluster_louvain(igraph_net)  # or choose another clustering algorithm
V(igraph_net)$cluster <- clusters$membership

RCy3::createNetworkFromIgraph(igraph_net, title=folder_name, collection=biome)

# Define a style name
style_name <- paste(folder_name, biome, sep="_")

# Create a custom style
defaults <- list(NODE_SHAPE="ellipse",
                 NODE_SIZE=20,  # Default node size
                 EDGE_TRANSPARENCY=120)

# Generate a color palette for clusters
num_clusters <- length(unique(V(igraph_net)$cluster))
color_palette <- paletteer_d("ggsci::category20_d3", num_clusters)
cluster_color_mapping <- setNames(color_palette, sort(unique(V(igraph_net)$cluster)))

# Define the range for the node sizes
# min_size <- 1
# max_size <- 10

# Map the log-transformed degree to the size range
nodeSizeMapping <- mapVisualProperty('node size', 'log_degree', 'c', 
                                     c(min_log_degree, max_log_degree), 
                                     c(min_size, max_size))

# Map the colors to the clusters
nodeColorMapping <- mapVisualProperty('node fill color', 'cluster', 'd', 
                                      names(cluster_color_mapping), 
                                      cluster_color_mapping)

# Create and apply the visual style
createVisualStyle(style_name, defaults, list(nodeSizeMapping, nodeColorMapping))
setVisualStyle(style_name)

