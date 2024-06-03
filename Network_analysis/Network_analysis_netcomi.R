# Import libraries
library(NetCoMi)
library(dplyr)

# Set seed to make results reproducible
set.seed(123)

# Function to load networks
loadNetwork <- function(file_path) {
  net <- readRDS(file_path)
  return(net)
}

# Function to analyze networks 
analyzeNetwork <- function(net) {
  props <- netAnalyze(net, 
                      centrLCC = FALSE,
                      avDissIgnoreInf = TRUE,
                      sPathNorm = TRUE,
                      clustMethod = "cluster_fast_greedy",
                      hubPar = "degree", # or other hub definition
                      hubQuant = 0.9,
                      lnormFit = TRUE,
                      normDeg = FALSE)
  return(props)
}

# Function to summarize network properties 
summarizeProperties <- function(net, degree_data, props, file_path) {
  # Calculate number of nodes and edges
  num_nodes <- length(unique(c(net$edgelist1$v1, net$edgelist1$v2)))
  num_edges <- nrow(net$edgelist1)
  
  # Filter out singletons and calculate average degree
  avg_degree <- degree_data %>%
    dplyr::filter(degree != 0) %>%
    dplyr::summarize(avg_degree = mean(degree, na.rm = TRUE)) %>%
    dplyr::pull(avg_degree)
  
  # Capture the summary, additional properties, and summary from props
  summary_text <- capture.output({
    cat("Number of Nodes: ", num_nodes, "\n")
    cat("Number of Edges: ", num_edges, "\n")
    cat("Average Degree (excluding singletons): ", avg_degree, "\n")
    cat("\n")
    print(summary(props, numbNodes = 5L, digits = 3L))
  })
  
  # Write to file
  cat(summary_text, file = file_path, sep = "\n", append = FALSE)
}

# Function to plot network 
plotNetwork <- function(props, file_path) {
  png(file_path, width=900, height=700)
  
  plot(props, layout = "spring", sameLayout = FALSE, layoutGroup = "union",
       repulsion = 0.8, nodeColor = "cluster", edgeTranspLow = 0,
       edgeTranspHigh = 40, nodeSize = "degree", posCol = "#ac1214", 
       negCol = "#0d1fad", labels = FALSE, #shortenLabels = "none", labelLength = 7, labelScale = FALSE, cexHubLabels = 1, cexLabels = 1.2,
       rmSingles = TRUE, cexNodes = 2, nodeSizeSpread = 2.6, hubBorderCol = "snow4")
  
  legend(-1, 1, title = "Estimated associations:", legend = c("+","-"), 
         col = c("#ac1214","#0d1fad"), inset = 0.02, cex = 1.4, lty = 1, lwd = 3.5, 
         bty = "n", horiz = TRUE, y.intersp = 0.7)
  
  dev.off()
}

# Base path, network types, and thresholds
root_path <- "Output"
# network_types <- c("spring", "cclasso")
network_type <- "cclasso"
biome <- "Wastewater_Activated_Sludge"
# thresholds <- seq(0.1, 0.5, by = 0.1)
thresh <- 0.1

# Loop through each network type and threshold
#for (network_type in network_types) {
# for (thresh in thresholds) {
  # Construct file paths for the network
folder_name <- sprintf("%s_%0.1f", network_type, thresh)
net_file_name <- sprintf("%s_net_%s_%02.0f.rds", biome, network_type, thresh * 10)
net_path <- file.path(root_path, biome, folder_name, net_file_name)

# Load, analyze, and summarize properties of the network if the file exists
if (file.exists(net_path)) {
  net <- loadNetwork(net_path)
  props <- analyzeNetwork(net)
  
  # Extract degree data from props
  degree_data <- as.data.frame(props$centralities$degree1)
  colnames(degree_data) <- "degree"
  
  # Summarize properties
  summary_file_name <- sprintf("summ_props_%s_%s_%0.1f.txt", biome, network_type, thresh)
  summary_file_path <- file.path(root_path, folder_name, summary_file_name)
  summarizeProperties(net, degree_data, props, summary_file_path)
  
  # Plot network
  plot_file_name <- sprintf("%s_%s_net_gen_deg_%0.1f.png", biome, network_type, thresh)
  plot_file_path <- file.path(root_path, folder_name, plot_file_name)
  plotNetwork(props, plot_file_path)
} else {
  cat("Network file not found: ", net_path, "\n")
}
# }
# }


