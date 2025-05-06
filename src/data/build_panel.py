"""
Build a panel dataset for economic analysis.

This script creates a panel dataset from the merged dataset, applying transformations
and filtering to prepare the data for econometric analysis.
"""
import os
import pandas as pd
import numpy as np
from pathlib import Path
import logging

from src.data.data_cleaning import main as run_raw_cleaning, merge_datasets

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def build_panel():
    """
    Build a panel dataset from the merged dataset.
    
    Steps:
    1. Run data cleaning if needed
    2. Create merged dataset if needed
    3. Read merged dataset
    4. Transform variables (logs, renames)
    5. Filter countries and years
    6. Save panel dataset
    
    Returns:
        pd.DataFrame: The processed panel dataset
    """
    # Step 1 & 2: Ensure raw data is cleaned and merged dataset exists
    ensure_merged_dataset_exists()
    
    # Step 3: Read the merged dataset
    logger.info("Reading merged dataset...")
    merged_file = 'data/processed/merged_dataset_1975_2024.csv'
    df = pd.read_csv(merged_file)
    logger.info(f"Loaded merged dataset with {df.shape[0]} rows and {df.shape[1]} columns")
    
    # Step 4: Transform variables
    logger.info("Transforming variables...")
    df = transform_variables(df)
    
    # Step 5: Filter countries and years
    logger.info("Filtering countries and years...")
    df = filter_dataset(df)
    
    # Step 6: Save panel dataset
    output_file = 'data/processed/panel_2003_2023.csv'
    logger.info(f"Saving panel dataset to {output_file}...")
    df.to_csv(output_file, index=False)
    logger.info(f"Panel dataset saved with {df.shape[0]} rows and {df.shape[1]} columns")
    
    return df

def ensure_merged_dataset_exists():
    """
    Make sure the merged dataset exists, running the cleaning and merging steps if needed.
    """
    merged_file = 'data/processed/merged_dataset_1975_2024.csv'
    
    if not os.path.exists(merged_file):
        logger.info("Merged dataset not found, running cleaning process...")
        run_raw_cleaning()
        
        if not os.path.exists(merged_file):
            logger.info("Running merge_datasets directly...")
            merge_datasets()
    else:
        logger.info(f"Found existing merged dataset: {merged_file}")

def transform_variables(df):
    """
    Transform variables in the dataset:
    - Create log transformations
    - Rename variables
    
    Args:
        df (pd.DataFrame): The input merged dataset
        
    Returns:
        pd.DataFrame: The transformed dataset
    """
    # Mapping of variable transformations
    transformations = {
        # Original variable name: (new name, transformation function)
        'Poverty headcount ratio at $3.65 a day (2017 PPP) (% of population)': 
            ('lnpovhead', lambda x: np.log1p(x)),
            
        'Poverty headcount ratio at $2.15 a day (2017 PPP) (% of population)': 
            ('lnpovhead215', lambda x: np.log1p(x)),
            
        'General government final consumption expenditure (% of GDP)': 
            ('lngovt', lambda x: np.log1p(x)),
            
        'Trade Openness (% of GDP)': 
            ('lntradeopen', lambda x: np.log1p(x)),
            
        'Rule of Law - estimate': 
            ('ruleoflaw', lambda x: x),  # Keep as is
            
        'GDP growth (annual %)': 
            ('gdpgrowth', lambda x: x),  # Keep as is
            
        'Gini index': 
            ('gini', lambda x: x),  # Keep as is
            
        'Inequality Index': 
            ('ineq', lambda x: x),  # Keep as is
            
        'Gender Inequality Index': 
            ('gii', lambda x: x),  # Keep as is
    }
    
    # Create copies of variables with transformations
    for original_var, (new_var, transform_func) in transformations.items():
        if original_var in df.columns:
            try:
                # Convert to numeric, coercing errors to NaN
                df[original_var] = pd.to_numeric(df[original_var], errors='coerce')
                
                # Apply transformation
                df[new_var] = transform_func(df[original_var])
                logger.info(f"Created transformed variable: {new_var}")
            except Exception as e:
                logger.warning(f"Could not transform {original_var}: {e}")
        else:
            logger.warning(f"Original variable not found: {original_var}")
    
    return df

def filter_dataset(df):
    """
    Filter the dataset to include:
    - Only specified countries (excluding the 7 with worst data)
    - Only years 2003-2023
    - Only select columns
    
    Args:
        df (pd.DataFrame): The input transformed dataset
        
    Returns:
        pd.DataFrame: The filtered dataset
    """
    # Countries to exclude (ISO3 codes)
    excluded_iso3s = {'BLZ', 'CUB', 'GTM', 'GUY', 'HTI', 'PRI', 'SUR'}
    
    # Map ISO3 to country names for filtering
    iso3_to_country = {
        'BLZ': 'Belize',
        'CUB': 'Cuba',
        'GTM': 'Guatemala',
        'GUY': 'Guyana',
        'HTI': 'Haiti',
        'PRI': 'Puerto Rico',
        'SUR': 'Suriname'
    }
    
    countries_to_exclude = [iso3_to_country[iso3] for iso3 in excluded_iso3s]
    logger.info(f"Excluding countries: {', '.join(countries_to_exclude)}")
    
    # Filter out the excluded countries
    filtered_df = df[~df['Country'].isin(countries_to_exclude)]
    logger.info(f"After country filtering: {filtered_df.shape[0]} rows")
    
    # Filter years to 2003-2023
    filtered_df = filtered_df[(filtered_df['Year'] >= 2003) & (filtered_df['Year'] <= 2023)]
    logger.info(f"After year filtering: {filtered_df.shape[0]} rows")
    
    # Select columns to keep
    columns_to_keep = [
        'Country', 'Year',
        'gdpgrowth', 'gini', 'ineq', 'gii', 'ruleoflaw',
        'lnpovhead', 'lnpovhead215', 'lngovt', 'lntradeopen'
    ]
    
    # Only keep columns that exist
    available_columns = [col for col in columns_to_keep if col in filtered_df.columns]
    missing_columns = set(columns_to_keep) - set(available_columns)
    
    if missing_columns:
        logger.warning(f"Some columns not available: {', '.join(missing_columns)}")
    
    # Select available columns
    final_df = filtered_df[available_columns].copy()
    logger.info(f"Final dataset: {final_df.shape[0]} rows and {final_df.shape[1]} columns")
    
    return final_df

if __name__ == "__main__":
    logger.info("Starting panel dataset creation...")
    panel_df = build_panel()
    logger.info("Panel dataset creation completed.") 