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
    4. Load additional datasets (FII, GII)
    5. Transform variables (logs, renames)
    6. Filter countries and years
    7. Update final dataset with GII values directly
    8. Save panel dataset
    
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
    
    # Step 4: Load additional datasets (FII, GII)
    logger.info("Loading additional datasets...")
    fi_data = load_financial_inclusion_index()
    gii_data = load_gender_inequality_index()
    
    # Step 5: Transform variables
    logger.info("Transforming variables...")
    df = transform_variables(df)
    
    # Step 6: Filter countries and years
    logger.info("Filtering countries and years...")
    df = filter_dataset(df)
    
    # Step 7: Add GII values directly to filtered dataset
    if gii_data is not None:
        logger.info("Adding GII values to final dataset...")
        df = add_gii_values(df, gii_data)
    
    # Step 8: Save panel dataset
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

def load_financial_inclusion_index():
    """
    Load the Financial Inclusion Index dataset.
    
    Returns:
        pd.DataFrame or None: The loaded and cleaned FII data, or None if it couldn't be loaded
    """
    try:
        logger.info("Loading Financial Inclusion Index data...")
        fi_data = pd.read_csv('data/interim/FI_index.csv')
        
        # Handle column names correctly regardless of capitalization
        fi_columns = fi_data.columns.tolist()
        country_col = next((col for col in fi_columns if col.upper() == 'COUNTRY'), None)
        year_col = next((col for col in fi_columns if col.lower() == 'year'), None)
        fi_col = next((col for col in fi_columns if col not in [country_col, year_col]), None)
        
        if country_col and year_col and fi_col:
            # Create a copy with standardized column names
            fi_data_std = fi_data.rename(columns={
                country_col: 'Country',
                year_col: 'Year',
                fi_col: 'fi_index'
            })
            
            logger.info("Financial Inclusion Index data loaded successfully")
            return fi_data_std
        else:
            logger.warning(f"Could not identify proper columns in FI_index.csv: {fi_columns}")
            return None
    except Exception as e:
        logger.warning(f"Could not load Financial Inclusion Index data: {e}")
        return None

def load_gender_inequality_index():
    """
    Load the Gender Inequality Index dataset.
    
    Returns:
        dict or None: Dictionary mapping (country, year) to GII value, or None if it couldn't be loaded
    """
    try:
        logger.info("Loading Gender Inequality Index data...")
        gii_df = pd.read_csv('data/interim/GII_cleaned.csv')
        
        # Clean and standardize the GII data
        logger.info("Cleaning GII data...")
        
        # Standardize column names
        gii_df.columns = ['country', 'year', 'gii_value']
        
        # Convert to numeric values
        gii_df['year'] = pd.to_numeric(gii_df['year'], errors='coerce')
        gii_df['gii_value'] = pd.to_numeric(gii_df['gii_value'], errors='coerce')
        
        # Standardize GII values (values > 1 are converted to 0-1 scale)
        mask = gii_df['gii_value'] > 1
        gii_df.loc[mask, 'gii_value'] = gii_df.loc[mask, 'gii_value'] / 100.0
        
        # Remove duplicates, keeping the most recent entry
        gii_df = gii_df.sort_values('gii_value').drop_duplicates(subset=['country', 'year'], keep='last')
        
        # Create a filled dataset for all countries and years
        logger.info("Creating filled GII dataset...")
        
        # Get unique countries and year range
        countries = gii_df['country'].unique()
        all_years = range(2003, 2024)  # Years 2003-2023
        
        # Dictionary to store filled values
        filled_gii = {}
        
        # For each country, fill in values for all years
        for country in countries:
            country_data = gii_df[gii_df['country'] == country].copy()
            
            # If no data for this country, skip
            if len(country_data) == 0:
                continue
                
            # For each year, get or interpolate a value
            for year in all_years:
                # Try to get the value for this year
                year_data = country_data[country_data['year'] == year]
                
                if len(year_data) > 0:
                    # Use the actual value for this year
                    filled_gii[(country, year)] = year_data.iloc[0]['gii_value']
                else:
                    # No value for this year, find the closest available year
                    closest_before = country_data[country_data['year'] < year].sort_values('year', ascending=False)
                    closest_after = country_data[country_data['year'] > year].sort_values('year')
                    
                    if len(closest_before) > 0 and len(closest_after) > 0:
                        # Interpolate between closest before and after
                        before_year = closest_before.iloc[0]['year']
                        before_val = closest_before.iloc[0]['gii_value']
                        after_year = closest_after.iloc[0]['year']
                        after_val = closest_after.iloc[0]['gii_value']
                        
                        # Linear interpolation
                        val = before_val + (after_val - before_val) * (year - before_year) / (after_year - before_year)
                        filled_gii[(country, year)] = val
                        
                    elif len(closest_before) > 0:
                        # Use last available value
                        filled_gii[(country, year)] = closest_before.iloc[0]['gii_value']
                        
                    elif len(closest_after) > 0:
                        # Use first available value
                        filled_gii[(country, year)] = closest_after.iloc[0]['gii_value']
        
        logger.info(f"GII data dictionary created with {len(filled_gii)} entries")
        return filled_gii
    except Exception as e:
        logger.warning(f"Could not load Gender Inequality Index data: {e}")
        import traceback
        logger.warning(traceback.format_exc())
        return None

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
            
        'Wealth': 
            ('wealth', lambda x: x),  # Keep as is
            
        'Gender Inequality Index': 
            ('gii', lambda x: x),  # Keep as is, but will be overwritten if GII_cleaned.csv exists
            
        'fi_index': 
            ('fii', lambda x: x),  # Financial Inclusion Index
            
        'School enrollment, secondary (% gross)': 
            ('lnenrollment', lambda x: np.log1p(x)),  # Log of school enrollment
            
        'Inflation, consumer prices (annual %)': 
            ('lninflation', lambda x: np.log1p(np.abs(x)) * np.sign(x)),  # Log transformation with sign preservation
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
        'gdpgrowth', 'gini', 'wealth', 'gii', 'ruleoflaw', 
        'lnpovhead215', 'lngovt', 'lntradeopen',
        'fii', 'lninflation', 'lnenrollment'
    ]
    
    # Ensure 'gii' is included in the columns to keep
    if 'gii' not in columns_to_keep:
        columns_to_keep.append('gii')
    
    # Only keep columns that exist
    available_columns = [col for col in columns_to_keep if col in filtered_df.columns]
    missing_columns = set(columns_to_keep) - set(available_columns)
    
    if missing_columns:
        logger.warning(f"Some columns not available: {', '.join(missing_columns)}")
    
    # Select available columns, but retain the GII column
    final_df = filtered_df[available_columns].copy()
    
    # Log GII info
    gii_count = final_df['gii'].notna().sum()
    logger.info(f"GII values after filtering: {gii_count} out of {len(final_df)} ({gii_count/len(final_df):.2%})")
    
    logger.info(f"Final dataset: {final_df.shape[0]} rows and {final_df.shape[1]} columns")
    
    return final_df

def add_gii_values(df, gii_data):
    """
    Add GII values to the filtered dataset.
    
    Args:
        df (pd.DataFrame): The filtered dataset
        gii_data (dict): Dictionary mapping (country, year) to GII value
        
    Returns:
        pd.DataFrame: The dataset with GII values added
    """
    if gii_data is None:
        logger.warning("No GII data available to add")
        return df
    
    # Create a copy to avoid modifying the original
    result_df = df.copy()
    
    # Get all unique countries in the dataset
    countries = set(df['Country'].unique())
    
    # For each row in the dataframe, try to find a matching GII value
    for idx, row in result_df.iterrows():
        country = row['Country']
        year = int(row['Year'])
        
        # Try different case variations of the country name
        for gii_country in [c for c in gii_data.keys() if c[0].lower() == country.lower()]:
            if (gii_country[0], year) in gii_data:
                result_df.at[idx, 'gii'] = gii_data[(gii_country[0], year)]
                break
    
    # Check how many GII values were added
    gii_count = result_df['gii'].notna().sum()
    logger.info(f"GII values added: {gii_count} out of {len(result_df)} ({gii_count/len(result_df):.2%})")
    
    return result_df

def load_additional_datasets(df):
    """
    Legacy function for loading additional datasets.
    
    Args:
        df (pd.DataFrame): The main dataframe to merge into
        
    Returns:
        None (modifies df in-place)
    """
    # Load Financial Inclusion Index (FII)
    try:
        fi_data_std = load_financial_inclusion_index()
        if fi_data_std is not None:
            # Merge with main dataframe
            merged = df.merge(fi_data_std, on=['Country', 'Year'], how='left')
            df['fi_index'] = merged['fi_index']
    except Exception as e:
        logger.warning(f"Could not load Financial Inclusion Index data: {e}")
    
    # Gender Inequality Index is now loaded separately

if __name__ == "__main__":
    logger.info("Starting panel dataset creation...")
    panel_df = build_panel()
    logger.info("Panel dataset creation completed.") 