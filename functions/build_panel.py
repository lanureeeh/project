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
import sys
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
# Add the project root to the path for direct script execution
if '__file__' in globals():
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

try:
    from data_cleaning import main as run_raw_cleaning, merge_datasets
except ImportError:
    # Try relative import for direct script execution
    from data_cleaning import main as run_raw_cleaning, merge_datasets

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
    8. Impute missing values
    9. Save panel dataset
    
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
    
    # Step 8: Impute missing values
    logger.info("Imputing missing values...")
    df = impute_missing_values(df)
    
    # Step 9: Save panel dataset
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
            
        # Transformations for lagged variables - directly from merged dataset
        'Access to electricity (% of population) (lag 5)': 
            ('electricity_lag5', lambda x: x),  # Keep as is
            
        'Account ownership at a financial institution or with a mobile-money-service provider (% of population ages 15+) (lag 5)': 
            ('account_ownership_lag5', lambda x: x),  # Keep as is
            
        'Commercial bank branches (per 100,000 adults) (lag 7)': 
            ('bank_branches_lag7', lambda x: x),  # Keep as is
            
        'Domestic credit to private sector (% of GDP) (lag 7)': 
            ('domestic_credit_lag7', lambda x: x),  # Keep as is
            
        'Individuals using the Internet (% of population) (lag 4)': 
            ('internet_users_lag4', lambda x: x),  # Keep as is
            
        'Mobile cellular subscriptions (per 100 people) (lag 4)': 
            ('mobile_subscriptions_lag4', lambda x: x),  # Keep as is
            
        'Urban population (% of total population) (lag 5)': 
            ('urban_population_lag5', lambda x: x),  # Keep as is
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
    - Only specified countries (excluding Venezuela and 7 with worst data)
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
    
    # Add Venezuela to the excluded countries
    countries_to_exclude.append('Venezuela')
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
        'fii', 'lninflation', 'lnenrollment',
        # Include the lagged variables
        'electricity_lag5', 'account_ownership_lag5', 'bank_branches_lag7',
        'domestic_credit_lag7', 'internet_users_lag4', 'mobile_subscriptions_lag4',
        'urban_population_lag5'
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
    
    # Log lagged variables info
    for col in ['electricity_lag5', 'account_ownership_lag5', 'bank_branches_lag7',
                'domestic_credit_lag7', 'internet_users_lag4', 'mobile_subscriptions_lag4',
                'urban_population_lag5']:
        if col in final_df.columns:
            count = final_df[col].notna().sum()
            logger.info(f"{col} values: {count} out of {len(final_df)} ({count/len(final_df):.2%})")
    
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

def impute_missing_values(df):
    """
    Impute missing values in the panel dataset.
    
    Imputation strategy:
    1. Linear interpolation for small gaps within country time series
    2. Forward/backward fill for isolated gaps
    3. KNN imputation based on similar countries for larger blocks
    
    Args:
        df (pd.DataFrame): The panel dataset with missing values
        
    Returns:
        pd.DataFrame: The dataset with imputed values
    """
    logger.info("Starting missing value imputation...")
    
    # Create a copy to avoid modifying the original
    imputed_df = df.copy()
    
    # Get numeric columns excluding identifiers
    numeric_cols = imputed_df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col not in ['Year']]
    
    # Log missing values before imputation
    missing_before = imputed_df[numeric_cols].isna().sum()
    logger.info(f"Missing values before imputation:\n{missing_before}")
    
    # Step 1: Linear interpolation for each country's time series
    logger.info("Step 1: Applying linear interpolation for each country...")
    for country in imputed_df['Country'].unique():
        country_mask = imputed_df['Country'] == country
        country_data = imputed_df.loc[country_mask, numeric_cols]
        
        # Only interpolate if there are enough non-NaN values
        for col in numeric_cols:
            non_null = country_data[col].notna().sum()
            if non_null >= 2:  # Need at least 2 points for interpolation
                imputed_df.loc[country_mask, col] = imputed_df.loc[country_mask, col].interpolate(method='linear')
    
    # Step 2: Forward fill and backward fill for remaining small gaps
    logger.info("Step 2: Applying forward and backward fill for isolated gaps...")
    for country in imputed_df['Country'].unique():
        country_mask = imputed_df['Country'] == country
        
        # Forward fill with limit to avoid filling large blocks
        imputed_df.loc[country_mask, numeric_cols] = imputed_df.loc[country_mask, numeric_cols].ffill(limit=2)
        
        # Backward fill with limit to avoid filling large blocks
        imputed_df.loc[country_mask, numeric_cols] = imputed_df.loc[country_mask, numeric_cols].bfill(limit=2)
    
    # Log missing values after basic imputation
    missing_after_basic = imputed_df[numeric_cols].isna().sum()
    logger.info(f"Missing values after basic imputation:\n{missing_after_basic}")
    
    # Step 3: KNN imputation for remaining values
    logger.info("Step 3: Applying KNN imputation based on similar countries...")
    
    # Get economic indicator columns for clustering
    economic_indicators = [col for col in numeric_cols if col not in ['gii']]
    
    # Get countries with sufficient data for clustering
    countries_data = []
    for country in imputed_df['Country'].unique():
        country_data = imputed_df[imputed_df['Country'] == country][economic_indicators]
        if country_data.isna().mean().mean() < 0.5:  # Less than 50% missing values
            # Calculate mean of each indicator for this country
            country_means = country_data.mean().to_dict()
            country_means['Country'] = country
            countries_data.append(country_means)
    
    if len(countries_data) >= 5:  # Need at least 5 countries for meaningful clustering
        # Create a dataframe of country means
        countries_df = pd.DataFrame(countries_data)
        countries_df = countries_df.set_index('Country')
        
        # Handle missing values in the clustering data
        countries_df = countries_df.fillna(countries_df.mean())
        
        # Normalize data for clustering
        scaler = MinMaxScaler()
        countries_scaled = scaler.fit_transform(countries_df)
        
        # Cluster countries into groups (adjust n_clusters based on data)
        n_clusters = min(5, len(countries_df) // 2)  # Use fewer clusters if data is limited
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(countries_scaled)
        
        # Create cluster mapping
        country_clusters = dict(zip(countries_df.index, clusters))
        
        # Apply KNN imputation within each cluster
        for cluster_id in range(n_clusters):
            # Get countries in this cluster
            cluster_countries = [country for country, cluster in country_clusters.items() if cluster == cluster_id]
            
            if len(cluster_countries) >= 3:  # Need at least 3 countries for meaningful KNN
                # Get data for countries in this cluster
                cluster_mask = imputed_df['Country'].isin(cluster_countries)
                cluster_data = imputed_df[cluster_mask].copy()
                
                # Create features for KNN: Year and dummy variables for countries
                X = pd.get_dummies(cluster_data[['Year', 'Country']], columns=['Country'])
                
                # For each numeric column with missing values, apply KNN imputation
                for col in numeric_cols:
                    if cluster_data[col].isna().any():
                        # Find rows that need imputation
                        missing_mask = cluster_data[col].isna()
                        if missing_mask.sum() > 0 and (~missing_mask).sum() >= 3:  # Ensure enough data points
                            # Apply KNN imputation (n_neighbors based on cluster size)
                            n_neighbors = min(5, (~missing_mask).sum() // 2)
                            imputer = KNNImputer(n_neighbors=n_neighbors)
                            
                            # Extract the column to impute and reshape for KNN
                            y = cluster_data[col].values.reshape(-1, 1)
                            
                            # Combine X and y for imputation
                            X_with_y = np.hstack((X.values, y))
                            
                            # Apply imputation
                            imputed_values = imputer.fit_transform(X_with_y)
                            
                            # Update the values in the dataframe
                            cluster_data[col] = imputed_values[:, -1]
                
                # Update the main dataframe with imputed values
                imputed_df.loc[cluster_mask, numeric_cols] = cluster_data[numeric_cols]
    
    # Fill any remaining missing values with global medians as a last resort
    logger.info("Filling remaining missing values with global medians...")
    global_medians = imputed_df[numeric_cols].median()
    imputed_df[numeric_cols] = imputed_df[numeric_cols].fillna(global_medians)
    
    # Log missing values after imputation
    missing_after = imputed_df[numeric_cols].isna().sum()
    logger.info(f"Missing values after all imputation steps:\n{missing_after}")
    
    # Calculate and log imputation percentage
    imputed_count = missing_before - missing_after
    logger.info(f"Imputed {imputed_count.sum()} values across {len(numeric_cols)} variables")
    
    return imputed_df

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