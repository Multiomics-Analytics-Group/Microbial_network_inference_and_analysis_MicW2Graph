# Import libraries
library(phyloseq)
library(dplyr)        # filter and reformat data frames
library(tibble)       # Needed for converting column to row names
library(microbiome)
library(stringr)

process_biome_experiment_type_data <- function(biome_experiment_type_name, base_folder = "Data/Biomes_exptypes"){
  # Create file paths using the biome_experiment_type name
  metadata_file_path <- file.path(base_folder, paste0(biome_experiment_type_name, "/", biome_experiment_type_name, "_merged_samples_metadata.csv"))
  abundance_file_path <- file.path(base_folder, paste0(biome_experiment_type_name, "/", biome_experiment_type_name, "_merged_abund_tables_species.csv"))
  taxonomic_file_path <- file.path(base_folder, paste0(biome_experiment_type_name, "/", biome_experiment_type_name, "_merged_taxa_tables_species.csv"))
  
  # Load data
  samples_metadata_df <- read.csv(metadata_file_path)
  abundance_df_species <- read.csv(abundance_file_path)
  taxonomic_df_species <- read.csv(taxonomic_file_path)
  
  # Add the Genus_Species column
  taxonomic_df_species <- taxonomic_df_species %>%
    mutate(Genus_Species = paste0(Genus, "_", Species))
  
  # Get the columns from the abundance table
  abundance_df_species_columns <- colnames(abundance_df_species)
  
  # Split 'assembly_run_ids', deduplicate, check each ID, and recombine valid IDs
  samples_metadata_df <- samples_metadata_df %>%
    mutate(assembly_run_ids = strsplit(as.character(assembly_run_ids), ";")) %>%
    rowwise() %>%
    mutate(assembly_run_ids = list(unique(Filter(function(id) id %in% abundance_df_species_columns, unlist(assembly_run_ids))))) %>%
    ungroup()
  
  # Ensure no rows have empty 'assembly_run_ids'
  samples_metadata_df <- samples_metadata_df %>%
    filter(lengths(assembly_run_ids) > 0)
  
  # Define the row names from the sample column
  samples_metadata_df <- samples_metadata_df %>% 
    tibble::column_to_rownames("assembly_run_ids") 
  
  # Set OTU as rowname
  taxonomic_df_species <- taxonomic_df_species %>% 
    tibble::column_to_rownames("OTU")
  
  abundance_df_species <- abundance_df_species %>% 
    tibble::column_to_rownames("OTU")
  
  # Transform into matrices otu and tax tables
  abund_mat_species <- as.matrix(abundance_df_species)
  tax_mat_species <- as.matrix(taxonomic_df_species)
  
  # Transform to phyloseq objects
  Abund_species = otu_table(abund_mat_species, taxa_are_rows = TRUE)
  Tax_species = tax_table(tax_mat_species)
  samples = sample_data(samples_metadata_df)
  
  phyloseq_file <- phyloseq(Abund_species, Tax_species, samples)
  
  return(phyloseq_file)
}

# List of biome_experiment_types
biome_experiment_type_names <- list.files(path = "Data/Biomes_exptypes", full.names = FALSE)

# Process each biome_experiment_type
for (biome_experiment_type in biome_experiment_type_names) {
  # Process the data for the biome_experiment_type
  biome_experiment_type_phyloseq <- process_biome_experiment_type_data(biome_experiment_type)
  
  # Construct the file path for saving the phuloseq object
  file_path <- paste0("Data/Biomes_exptypes/", biome_experiment_type, "/", biome_experiment_type, "_species_phyloseqfile.rds")
  
  # Save the biome_experiment_type_phyloseq object
  saveRDS(biome_experiment_type_phyloseq, file = file_path)
}

