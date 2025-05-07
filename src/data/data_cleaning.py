"""
Data cleaning script that combines functionality from:
- data_cleaning.py
- GII_Processing.ipynb
- FI Index.ipynb
- FI Index copy.ipynb
- Instruments.py

Changes:
- Modified input paths from raw_data/ to data/raw/
- Modified output paths to data/interim/ and data/processed/
"""
import pandas as pd
import re
import os
import sys
import numpy as np
from pathlib import Path
import warnings

# Suppress pandas warnings
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)

# Create directories if they don't exist
try:
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/interim', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    print("Directory structure validated.")
except Exception as e:
    print(f"Error creating directories: {e}")
    sys.exit(1)

def clean_fas_data():
    """
    Clean the Financial Access Survey (FAS) data.
    Read from data/raw/FAS.csv
    Save to data/interim/FAS_cleaned.csv
    """
    # Read the FAS.csv file
    file_path = 'data/raw/FAS.csv'
    data = pd.read_csv(file_path)
    print("FAS.csv - Initial Data:")
    print(data.head())

    # Define the list of countries to keep
    countries_to_keep = [
        'Mexico', 'Peru', 'Belize', 'Costa Rica', 'El Salvador', 'Guatemala',
        'Honduras', 'Nicaragua', 'Panama', 'Cuba', 'Dominican Republic', 'Haiti',
        'Puerto Rico', 'Argentina', 'Bolivia', 'Brazil', 'Chile', 'Colombia',
        'Ecuador', 'Guyana', 'Paraguay', 'Suriname', 'Uruguay', 'Venezuela, República Bolivariana de'
    ]

    # Filter the data to keep only the specified countries
    data_filtered = data[data['COUNTRY'].isin(countries_to_keep)]

    # Exceptional name standardization for 'Venezuela, República Bolivariana de'
    # the ' sign leading to missing values in the COUNTRY column after exporting to csv
    # Replacing Venezuel, República Bolivariana de with Venezuela
    data_filtered['COUNTRY'] = data_filtered['COUNTRY'].replace('Venezuela, República Bolivariana de', 'Venezuela')

    # Columns to delete
    delete_columns = [
        'DATASET', 'SERIES_CODE', 'OBS_MEASURE', 'FREQUENCY',
        'SCALE', 'PRECISION', 'DECIMALS_DISPLAYED', 'INSTR_ASSET', 'FA_INDICATORS',
        'SECTOR', 'COUNTERPART_SECTOR', 'ACCOUNTING_ENTRY', 'SEX', 'TRANSFORMATION',
        'UNIT', 'DERIVATION_TYPE', 'OVERLAP', 'STATUS', 'DOI', 'FULL_DESCRIPTION',
        'AUTHOR', 'PUBLISHER', 'DEPARTMENT', 'CONTACT_POINT', 'TOPIC', 'TOPIC_DATASET',
        'KEYWORDS', 'KEYWORDS_DATASET', 'LANGUAGE', 'PUBLICATION_DATE', 'UPDATE_DATE',
        'METHODOLOGY', 'METHODOLOGY_NOTES', 'ACCESS_SHARING_LEVEL', 'ACCESS_SHARING_NOTES',
        'SECURITY_CLASSIFICATION', 'SOURCE', 'SHORT_SOURCE_CITATION', 'FULL_SOURCE_CITATION',
        'LICENSE', 'SUGGESTED_CITATION', 'KEY_INDICATOR', 'SERIES_NAME'
    ]

    # Drop the specified columns
    data_cleaned = data_filtered.drop(columns=delete_columns)

    selected_indicators = [
    # Branches
    'Branches excluding headquarters, Other deposit takers',
    'Branches excluding headquarters, Commercial banks',
    'Branches excluding headquarters, Credit unions and credit cooperatives',
    'Branches excluding headquarters, Deposit taking microfinance institutions',
    'Branches excluding headquarters, Non-deposit taking microfinance institutions',
    'Number of commercial bank branches',
    'Number of credit union and credit cooperative branches',
    'Number of other deposit taker branches',
    'Number of all microfinance institution branches',

    # ATMs
    'Number of automated teller machines (ATMs)',

    # Depositors 
    'Number of depositors, Commercial banks',
    'Depositors, Deposit taking microfinance institutions',
    'Depositors, Commercial banks',
    'Number of depositors, Commercial banks',
    'Number of depositors, Credit unions and credit cooperatives',

    # Deposit accounts 
    'Number of deposit accounts, Commercial banks',
    'Deposit accounts, Commercial banks',
    'Deposit accounts, Deposit taking microfinance institutions',
    'Deposit accounts, Credit unions and credit cooperatives',
    'Number of deposit accounts, Commercial banks',
    'Number of deposit accounts, Credit unions and credit cooperatives',

    # Borrowers
    'Number of borrowers, Commercial banks',
    'Borrowers, Commercial banks',
    'Borrowers, Deposit taking microfinance institutions',
    'Borrowers, Credit unions and credit cooperatives',
    'Borrowers, Non-deposit taking microfinance institutions',

    # Loan accounts
    'Number of loan accounts, Commercial banks',
    'Loan accounts, Commercial banks',
    'Loan accounts, Deposit taking microfinance institutions',
    'Loan accounts, Credit unions and credit cooperatives',
    'Loan accounts, Non-deposit taking microfinance institutions',
    ]

    data_cleaned_filtered_indicators = data_cleaned[data_cleaned['INDICATOR'].isin(selected_indicators)]

    # Steps to convert the data to long format
    # 1) Identify the year-columns via regex (exactly four digits)
    year_cols = [col for col in data_cleaned_filtered_indicators.columns if re.fullmatch(r"\d{4}", col)]

    # 2) Melt into long format
    df_long = data_cleaned_filtered_indicators.melt(
        id_vars=["COUNTRY", "INDICATOR", 'TYPE_OF_TRANSFORMATION'],
        value_vars=year_cols,
        var_name="year",
        value_name="value"
    )

    # 3) Convert types
    df_long["year"] = df_long["year"].astype(int)
    df_long["value"] = pd.to_numeric(df_long["value"], errors="coerce")

    # Filter for the years 2004 to 2023
    df_long = df_long[(df_long['year'] >= 2004) & (df_long['year'] <= 2023)]
   
    # Sort the DataFrame by country, Indicator and year
    df_long = df_long.sort_values(by=["COUNTRY", "INDICATOR", "year"])

    # Save the cleaned data to a new CSV file
    df_long.to_csv('data/interim/FAS_cleaned.csv', index=False)
    print("FAS.csv cleaning complete. 'data/interim/FAS_cleaned.csv' has been created.")


def clean_wgi_data():
    """
    Clean the World Governance Indicators (WGI) data.
    Read from data/raw/WGI.csv
    Save to data/interim/WGI_cleaned.csv
    """
    # Read the WGI.csv file
    file_path = 'data/raw/WGI.csv'
    data = pd.read_csv(file_path, delimiter=';')
    print("WGI.csv - Initial Data:")
    print(data.head())

    # Define the list of countries to keep
    countries_to_keep = [
        'Mexico', 'Peru', 'Belize', 'Costa Rica', 'El Salvador', 'Guatemala',
        'Honduras', 'Nicaragua', 'Panama', 'Cuba', 'Dominican Republic', 'Haiti',
        'Puerto Rico', 'Argentina', 'Bolivia', 'Brazil', 'Chile', 'Colombia',
        'Ecuador', 'Guyana', 'Paraguay', 'Suriname', 'Uruguay', 'Venezuela, RB'
    ]

    # Delete the 'codeindyr' and 'code' columns
    data = data.drop(columns=['codeindyr', 'code'])

    # Filter rows where 'indicator' is 'rl'
    data_filtered = data[data['indicator'] == 'rl']

    # Delete the 'indicator' column
    data_cleaned = data_filtered.drop(columns=['indicator'])
    
    # Filter to keep only rows for the specified countries
    data_cleaned = data_cleaned[data_cleaned['countryname'].isin(countries_to_keep)]

    print("WGI.csv - Cleaned Data:")
    print(data_cleaned.head())

    # Save the cleaned data to a new CSV file
    data_cleaned.to_csv('data/interim/WGI_cleaned.csv', index=False)

    print("WGI.csv cleaning complete. 'data/interim/WGI_cleaned.csv' has been created.")


def clean_wdi_data():
    """
    Clean the World Development Indicators (WDI) data.
    Read from data/raw/WDI.csv
    Save to data/interim/WDI_cleaned.csv
    """
    # Read the WDI.csv file
    file_path = 'data/raw/WDI.csv'
    data = pd.read_csv(file_path)
    print("WDI.csv - Initial Data:")
    print(data.head())

    # Delete the 'Country Code' and 'Series Code' columns
    data = data.drop(columns=['Country Code', 'Series Code'])

    # Remove ' [YRxxxx]' from year columns
    data.columns = data.columns.str.replace(r' \[YR\d{4}\]', '', regex=True)

    print("WDI.csv - Cleaned Data:")
    print(data.head())

    # Save the cleaned data to a new CSV file
    data.to_csv('data/interim/WDI_cleaned.csv', index=False)

    print("WDI.csv cleaning complete. 'data/interim/WDI_cleaned.csv' has been created.")


def clean_wid_data():
    """
    Clean the World Inequality Database (WID) data.
    Read from data/raw/WID.csv
    Save to data/interim/WID_cleaned.csv
    """
    # Read the WID.csv file
    file_path = 'data/raw/WID.csv'
    data = pd.read_csv(file_path, delimiter=';')
    print("WID.csv - Initial Data:")
    print(data.head())

    # Delete the 'Percentile' column
    data = data.drop(columns=['Percentile'])

    # Pivot the data so that years are columns and countries are rows
    data_pivoted = data.set_index('Year').T

    # Rename the first column to 'Country'
    data_pivoted.index.name = 'Country'

    print("WID.csv - Cleaned Data:")
    print(data_pivoted.head())

    # Save the cleaned data to a new CSV file
    data_pivoted.to_csv('data/interim/WID_cleaned.csv')

    print("WID.csv cleaning complete. 'data/interim/WID_cleaned.csv' has been created.")


def clean_gii_data():
    """
    Clean the Gender Inequality Index (GII) data.
    Read from data/raw/GII.csv
    Save to data/interim/GII_cleaned.csv
    
    Adapted from GII_Processing.ipynb
    """
    # Read the GII.csv file (attempting with different encodings)
    file_path = 'data/raw/GII.csv'
    
    try:
        # First try with UTF-8 encoding
        print(f"Attempting to read {file_path} with UTF-8 encoding...")
        data = pd.read_csv(file_path)
    except UnicodeDecodeError:
        try:
            # If that fails, try with Latin-1 encoding
            print(f"UTF-8 encoding failed. Attempting with Latin-1 encoding...")
            data = pd.read_csv(file_path, encoding='latin1')
        except Exception as e:
            # If that also fails, try with Windows-1252 encoding (common for Windows files)
            print(f"Latin-1 encoding failed. Attempting with Windows-1252 encoding...")
            try:
                data = pd.read_csv(file_path, encoding='cp1252')
            except Exception as e2:
                print(f"All encoding attempts failed. Error: {e2}")
                print("Skipping GII processing due to encoding issues.")
                return
    
    print("GII.csv - Initial Data:")
    print(data.head())
    
    # Define the list of countries to keep
    countries_to_keep = [
        'Mexico', 'Peru', 'Belize', 'Costa Rica', 'El Salvador', 'Guatemala',
        'Honduras', 'Nicaragua', 'Panama', 'Cuba', 'Dominican Republic', 'Haiti',
        'Puerto Rico', 'Argentina', 'Bolivia', 'Brazil', 'Chile', 'Colombia',
        'Ecuador', 'Guyana', 'Paraguay', 'Suriname', 'Uruguay', 'Venezuela'
    ]
    
    # Check if 'country' column exists
    if 'country' not in data.columns:
        print("Warning: 'country' column not found in GII.csv")
        # Look for alternative column names
        possible_country_cols = [col for col in data.columns if 'country' in col.lower()]
        if possible_country_cols:
            print(f"Using '{possible_country_cols[0]}' as country column")
            data = data.rename(columns={possible_country_cols[0]: 'country'})
        else:
            print("Could not find a country column. Skipping GII processing.")
            return
    
    # Filter data to keep only rows for the specified countries
    data_filtered = data[data['country'].isin(countries_to_keep)]
    
    # Check if we have any GII-related columns
    gii_columns = [col for col in data.columns if 'gii' in col.lower()]
    
    if not gii_columns:
        print("Warning: No GII-related columns found in the data")
        print("Available columns:", data.columns.tolist())
        print("Skipping GII processing.")
        return
    
    print(f"Found {len(gii_columns)} GII-related columns: {gii_columns[:5]}...")
    
    # Extract GII-related columns
    country_column = ['country']
    selected_columns = country_column + gii_columns
    
    # Select only the needed columns
    data_cleaned = data_filtered[selected_columns]
    
    # Reshape data from wide to long format
    years = [col.split('_')[1] if '_' in col else col for col in gii_columns]
    id_vars = ['country']
    
    try:
        # Melt the data
        data_long = data_cleaned.melt(
            id_vars=id_vars,
            value_vars=gii_columns,
            var_name='year',
            value_name='gii_value'
        )
        
        # Extract year from the 'year' column
        if '_' in data_long['year'].iloc[0]:
            data_long['year'] = data_long['year'].str.replace(r'.*_', '', regex=True)
    except Exception as e:
        print(f"Error in melting GII data: {e}")
        print("Saving partially cleaned data instead.")
        data_cleaned.to_csv('data/interim/GII_cleaned.csv', index=False)
        return
    
    # Save the cleaned data
    data_long.to_csv('data/interim/GII_cleaned.csv', index=False)
    print("GII.csv cleaning complete. 'data/interim/GII_cleaned.csv' has been created.")


def standardize_country_names(file_path):
    """
    Standardize country names in a dataset.
    """
    # Read the cleaned CSV file
    data = pd.read_csv(file_path)
    
    # Define a mapping of country names to their standardized format
    country_name_mapping = {
        'Mexico': 'Mexico',
        'Peru': 'Peru',
        'Belize': 'Belize',
        'Costa Rica': 'Costa Rica',
        'El Salvador': 'El Salvador',
        'Guatemala': 'Guatemala',
        'Honduras': 'Honduras',
        'Nicaragua': 'Nicaragua',
        'Panama': 'Panama',
        'Cuba': 'Cuba',
        'Dominican Republic': 'Dominican Republic',
        'Haiti': 'Haiti',
        'Puerto Rico': 'Puerto Rico',
        'Argentina': 'Argentina',
        'Bolivia': 'Bolivia',
        'Brazil': 'Brazil',
        'Chile': 'Chile',
        'Colombia': 'Colombia',
        'Ecuador': 'Ecuador',
        'Guyana': 'Guyana',
        'Paraguay': 'Paraguay',
        'Suriname': 'Suriname',
        'Uruguay': 'Uruguay',
        'Venezuela': 'Venezuela',
        'Venezuela, RB': 'Venezuela',
    }
    
    # Check for possible column names for countries
    country_column = None
    if 'Country' in data.columns:
        country_column = 'Country'
    elif 'COUNTRY' in data.columns:
        country_column = 'COUNTRY'
    elif 'countryname' in data.columns:
        country_column = 'countryname'
    elif 'Country Name' in data.columns:
        country_column = 'Country Name'
    elif 'country' in data.columns:
        country_column = 'country'
    else:
        raise KeyError("No country column found in the dataset.")

    # Print the column names for debugging
    print(f"Columns in {file_path}: {data.columns.tolist()}")
    
    # Print the first few rows before standardization for debugging
    print(f"Data before standardization in {file_path}:")
    print(data.head())

    # Filter countries based on our list
    filtered_countries = ['Mexico', 'Peru', 'Belize', 'Costa Rica', 'El Salvador', 
                       'Guatemala', 'Honduras', 'Nicaragua', 'Panama', 'Cuba', 
                       'Dominican Republic', 'Haiti', 'Puerto Rico', 'Argentina', 
                       'Bolivia', 'Brazil', 'Chile', 'Colombia', 'Ecuador', 
                       'Guyana', 'Paraguay', 'Suriname', 'Uruguay', 'Venezuela', 'Venezuela, RB']
    
    # Create a mask for rows that have countries in our list
    mask = data[country_column].isin(filtered_countries)
    
    # Apply mapping only to those rows
    data.loc[mask, country_column] = data.loc[mask, country_column].map(country_name_mapping)

    # Print the first few rows after standardization for debugging
    print(f"Data after standardization in {file_path}:")
    print(data.head())

    # Save the updated data back to the CSV file
    data.to_csv(file_path, index=False)

    print(f"Country names standardized in {file_path}.")


def merge_datasets():
    """
    Merge all cleaned datasets into a single comprehensive dataset.
    Read from data/interim/
    Save to data/processed/merged_dataset_1975_2024.csv
    """
    # Check which datasets are available
    interim_files = os.listdir('data/interim')
    print("Merging the following datasets:")
    for f in interim_files:
        # if f.endswith('_cleaned.csv'):
            print(f"  - {f}")
    
    # Initialize empty DataFrames
    fas = pd.DataFrame()
    wgi = pd.DataFrame()
    wdi = pd.DataFrame()
    wid = pd.DataFrame()
    gii = pd.DataFrame()
    instruments = pd.DataFrame()
    fi_index = pd.DataFrame()
    instruments_fii = pd.DataFrame()
    
    # Read all cleaned datasets if they exist
    if 'FAS_cleaned.csv' in interim_files:
        try:
            fas = pd.read_csv('data/interim/FAS_cleaned.csv')
            print(f"Loaded FAS data: {fas.shape[0]} rows, {fas.shape[1]} columns")
        except Exception as e:
            print(f"Error loading FAS data: {e}")
    
    if 'WGI_cleaned.csv' in interim_files:
        try:
            wgi = pd.read_csv('data/interim/WGI_cleaned.csv')
            print(f"Loaded WGI data: {wgi.shape[0]} rows, {wgi.shape[1]} columns")
        except Exception as e:
            print(f"Error loading WGI data: {e}")
    
    if 'WDI_cleaned.csv' in interim_files:
        try:
            wdi = pd.read_csv('data/interim/WDI_cleaned.csv')
            print(f"Loaded WDI data: {wdi.shape[0]} rows, {wdi.shape[1]} columns")
        except Exception as e:
            print(f"Error loading WDI data: {e}")
    
    if 'WID_cleaned.csv' in interim_files:
        try:
            wid = pd.read_csv('data/interim/WID_cleaned.csv')
            print(f"Loaded WID data: {wid.shape[0]} rows, {wid.shape[1]} columns")
        except Exception as e:
            print(f"Error loading WID data: {e}")
    
    if 'GII_cleaned.csv' in interim_files:
        try:
            gii = pd.read_csv('data/interim/GII_cleaned.csv')
            print(f"Loaded GII data: {gii.shape[0]} rows, {gii.shape[1]} columns")
        except Exception as e:
            print(f"Error loading GII data: {e}")
    
    if 'Instruments_Poverty_cleaned.csv' in interim_files:
        try:
            instruments = pd.read_csv('data/interim/Instruments_Poverty_cleaned.csv')
            print(f"Loaded Instruments data: {instruments.shape[0]} rows, {instruments.shape[1]} columns")
        except Exception as e:
            print(f"Error loading Instruments data: {e}")
    
    if 'FI_index.csv' in interim_files:
        try:
            fi_index = pd.read_csv('data/interim/FI_index.csv')
            print(f"Loaded FI_index data: {fi_index.shape}")
        except Exception as e:
            print(f"Error loading FI_index data: {e}")
    
    if 'Instruments_FII_lagged.csv' in interim_files:
        try:
            instruments_fii = pd.read_csv('data/interim/Instruments_FII_lagged.csv')
            print(f"Loaded Instruments_FII data: {instruments_fii.shape}")
        except Exception as e:
            print(f"Error loading Instruments_FII_lagged data: {e}")

    # Check if any datasets were loaded
    if all(df.empty for df in [fas, wgi, wdi, wid, gii, instruments, fi_index, instruments_fii]):
        print("No datasets available for merging. Exiting.")
        return
    
    print("Reading all cleaned datasets...")
    
    # Define the years range from 1975 to 2024
    years = list(range(1975, 2025))
    
    # Get the list of target countries - excluding the countries with worst data
    target_countries = [
        'Mexico', 'Peru', 'Costa Rica', 'El Salvador', 'Honduras', 
        'Nicaragua', 'Panama', 'Dominican Republic', 'Argentina', 
        'Bolivia', 'Brazil', 'Chile', 'Colombia', 'Ecuador', 
        'Paraguay', 'Uruguay', 'Venezuela'
    ]
    
    # Create a multi-index DataFrame with countries and years
    index = pd.MultiIndex.from_product([target_countries, years], names=['Country', 'Year'])
    merged_data = pd.DataFrame(index=index).reset_index()
    
    # Initialize all data columns with NaN values
    # Variables as shown in the screenshot
    merged_data['GDP growth (annual %)'] = float('nan')
    merged_data['Population, total'] = float('nan')
    merged_data['Gini index'] = float('nan')
    merged_data['School enrollment, secondary (% gross)'] = float('nan')
    merged_data['Poverty headcount ratio at $2.15 a day (2017 PPP) (% of population)'] = float('nan')
    merged_data['Poverty headcount ratio at $3.65 a day (2017 PPP) (% of population)'] = float('nan')
    merged_data['General government final consumption expenditure (% of GDP)'] = float('nan')
    merged_data['Exports of goods and services (% of GDP)'] = float('nan')
    merged_data['Imports of goods and services (% of GDP)'] = float('nan')
    merged_data['Trade Openness (% of GDP)'] = float('nan')
    merged_data['Inflation, consumer prices (annual %)'] = float('nan')
    merged_data['Rule of Law - estimate'] = float('nan')
    merged_data['Rule of Law - pctrank'] = float('nan')
    merged_data['Wealth'] = float('nan')
    # merged_data['fi_index'] = float('nan')
    # merged_data['Access to electricity (% of population)'] = float('nan')
    # merged_data['Domestic credit to private sector (% of GDP)'] = float('nan')
    # merged_data['Commercial bank branches (per 100,000 adults)'] = float('nan')
    # merged_data['Urban population (% of total population)'] = float('nan')
    # merged_data['Account ownership at a financial institution or with a mobile-money-service provider (% of population ages 15+)']  = float('nan')
    # merged_data['Mobile cellular subscriptions (per 100 people)'] = float('nan')
    # merged_data['Individuals using the Internet (% of population)'] = float('nan')
    
    # Add Gender Inequality Index if available
    if not gii.empty:
        merged_data['Gender Inequality Index'] = float('nan')
    
    # Add additional indicator columns from instruments if available
    if not instruments.empty:
        merged_data['Poverty Rate $3.65'] = float('nan')
    
    # Add FI index columns
    if not fi_index.empty:
        for col in fi_index.columns:
            if col not in ('COUNTRY', 'year'):
                merged_data[col] = float('nan')

    # Add FII_instrument columns
    if not instruments_fii.empty:
        for col in instruments_fii.columns:
            if col not in ('Unnamed: 0', 'Country Name', 'year'):
                merged_data[col] = float('nan')

    print("Created empty dataset with target variables...")
    
    # Add data from WDI dataset (contains most of the economic indicators)
    if not wdi.empty:
        print("Adding WDI data...")
        for _, row in wdi.iterrows():
            country = row['Country Name']
            if country in target_countries:
                series_name = row['Series Name']
                # Only process rows with series names that match our target variables
                if series_name in merged_data.columns:
                    for year in years:
                        if str(year) in wdi.columns:
                            value = row[str(year)]
                            # Find the corresponding row in merged_data
                            mask = (merged_data['Country'] == country) & (merged_data['Year'] == year)
                            if not pd.isna(value):
                                merged_data.loc[mask, series_name] = value
    
    # Add data from WGI dataset (Rule of Law indicators)
    if not wgi.empty:
        print("Adding WGI data...")
        for _, row in wgi.iterrows():
            country = row['countryname'] if 'countryname' in wgi.columns else row.get('country', '')
            if country in target_countries:
                year = row['year']
                if year in years:
                    # Find the corresponding row in merged_data
                    mask = (merged_data['Country'] == country) & (merged_data['Year'] == year)
                    # Add Rule of Law estimate
                    if 'estimate' in wgi.columns and not pd.isna(row['estimate']):
                        merged_data.loc[mask, 'Rule of Law - estimate'] = row['estimate']
                    # Add Rule of Law percentile rank
                    if 'pctrank' in wgi.columns and not pd.isna(row['pctrank']):
                        merged_data.loc[mask, 'Rule of Law - pctrank'] = row['pctrank']
    
    # Add data from WID dataset (Inequality Index, Wealth Accumilation)
    if not wid.empty:
        print("Adding WID data...")
        # Convert index values to columns if needed
        if not isinstance(wid.index, pd.RangeIndex):
            wid = wid.reset_index()
        
        for _, row in wid.iterrows():
            # Get the country name from the appropriate column
            country = row['Country']
            if country in target_countries:
                for year in years:
                    year_str = str(year)
                    if year_str in wid.columns:
                        value = row[year_str]
                        # Find the corresponding row in merged_data
                        mask = (merged_data['Country'] == country) & (merged_data['Year'] == year)
                        if not pd.isna(value):
                            merged_data.loc[mask, 'Wealth'] = value
                            print(f"Added WID data for {country}, {year}: {value}")
    
    # Add data from FAS dataset (if relevant)
    if not fas.empty:
        print("Adding FAS data...")
        # Process if there are any variables from FAS that need to be added
    
    # Add data from GII dataset (if available)
    if not gii.empty:
        print("Adding GII data...")
        for _, row in gii.iterrows():
            country = row['country']
            if country in target_countries:
                year = row['year']
                if year in [str(y) for y in years]:
                    # Convert year to int for matching
                    try:
                        year_int = int(year)
                        # Find the corresponding row in merged_data
                        mask = (merged_data['Country'] == country) & (merged_data['Year'] == year_int)
                        # Add Gender Inequality Index value
                        if not pd.isna(row['gii_value']):
                            merged_data.loc[mask, 'Gender Inequality Index'] = row['gii_value']
                    except ValueError:
                        continue
    
    # Add data from Instruments (poverty data)
    if not instruments.empty:
        print("Adding Instruments poverty data...")
        if 'Instruments_Poverty_long.csv' in interim_files:
            try:
                # Use the long format for easier merging
                poverty_long = pd.read_csv('data/interim/Instruments_Poverty_long.csv')
                for _, row in poverty_long.iterrows():
                    country = row['Country Name']
                    if country in target_countries:
                        year = row['Year']
                        if year.isdigit() and int(year) in years:
                            year_int = int(year)
                            mask = (merged_data['Country'] == country) & (merged_data['Year'] == year_int)
                            if not pd.isna(row['Poverty_Rate']):
                                merged_data.loc[mask, 'Poverty Rate $3.65'] = row['Poverty_Rate']
            except Exception as e:
                print(f"Error adding poverty data: {e}")
    
    print("Merged data shape:", merged_data.shape)
    print("Merged data columns:", merged_data.columns.tolist())
    
    # Calculate Trade Openness as the sum of exports and imports (% of GDP)
    print("Calculating Trade Openness...")
    for index, row in merged_data.iterrows():
        exports = merged_data.loc[index, 'Exports of goods and services (% of GDP)']
        imports = merged_data.loc[index, 'Imports of goods and services (% of GDP)']
        if pd.notna(exports) and pd.notna(imports):
            try:
                # Try to convert to float if they are strings
                if isinstance(exports, str):
                    exports = float(exports)
                if isinstance(imports, str):
                    imports = float(imports)
                merged_data.loc[index, 'Trade Openness (% of GDP)'] = exports + imports
            except (ValueError, TypeError):
                # If conversion fails, leave as NaN
                pass

    # New: Add FI index data
    if not fi_index.empty:
        print("Adding FI Index data...")
        for _, row in fi_index.iterrows():
            country = row['COUNTRY']
            year = int(row['year']) if str(row['year']).isdigit() else None
            if country in target_countries and year in years:
                mask = (merged_data['Country'] == country) & (merged_data['Year'] == year)
                for col in fi_index.columns:
                    if col not in ('COUNTRY', 'year'):
                        val = row[col]
                        if pd.notna(val):
                            merged_data.loc[mask, col] = val
        
     # New: Add FII instruments data
    if not instruments_fii.empty:
        print("Adding Instruments_FII data...")
        for _, row in instruments_fii.iterrows():
            country = row['Country Name']
            year = int(row['year']) if str(row['year']).isdigit() else None
            if country in target_countries and year in years:
                mask = (merged_data['Country'] == country) & (merged_data['Year'] == year)
                for col in instruments_fii.columns:
                    if col not in ('Unnamed: 0', 'Country Name', 'year'):
                        val = row[col]
                        if pd.notna(val):
                            merged_data.loc[mask, col] = val
    
    # Save the merged dataset
    merged_data.to_csv('data/processed/merged_dataset_1975_2024.csv', index=False)
    print("Merged dataset saved to 'data/processed/merged_dataset_1975_2024.csv'")


def process_financial_inclusion_index():
    import pandas as pd
    import numpy as np
    #from ace_tools import display_dataframe_to_user

    """
    Compute the Financial Inclusion Index (FII) from the FAS dataset.
    Read from data/interim/FAS_cleaned.csv
    Save to data/interim/FI_index.csv
    """

    # 1) Load the long‐format dataset
    file_path = 'data/interim/FAS_cleaned.csv'
    df_long = pd.read_csv(file_path)

    # 2) Select only the six relevant indicators and focus on per‐adult metrics to avoid duplicates
    fii_vars = [
        'Number of automated teller machines (ATMs)',
        'Number of borrowers, Commercial banks',
        'Number of commercial bank branches',
        'Number of deposit accounts, Commercial banks',
        'Number of depositors, Commercial banks',
        'Number of loan accounts, Commercial banks'
    ]
    df_sel = df_long[
        df_long['INDICATOR'].isin(fii_vars) &
        df_long['TYPE_OF_TRANSFORMATION'].str.contains('adults')
    ]

    # 3) Pivot to wide format with one row per country‐year
    df_wide = (
        df_sel
        .pivot(index=['COUNTRY', 'year'], columns='INDICATOR', values='value')
        .reset_index()
    )

    # 4) Define the function to compute the Financial Inclusion Index
    def compute_fii(df):
        # Standardize and cap each indicator to [0,1] using its 95th percentile
        pct95 = df[fii_vars].quantile(0.95)
        d_mat = df[fii_vars].divide(pct95).clip(lower=0, upper=1)

        # Weight matrix: 1 if data present, 0 if missing
        w_mat = df[fii_vars].notna().astype(int)

        # Replace NaNs in standardized data with 0
        d0_mat = d_mat.fillna(0)

        # Compute per‐row normalizer: √(∑ w_i²)
        norm_w = np.sqrt((w_mat**2).sum(axis=1))

        # Distance from worst (origin)
        X1 = np.sqrt((d0_mat**2).sum(axis=1)) / norm_w

        # Inverse distance from ideal (all ones)
        X2 = 1 - np.sqrt(((w_mat - d0_mat)**2).sum(axis=1)) / norm_w

        # Combine into the final index
        df['fi_index'] = 0.5 * (X1 + X2)
        return df

    # 5) Compute the index
    df_fii = compute_fii(df_wide)

    # 6) Select only the relevant columns
    df_fii = df_fii[['COUNTRY', 'year', 'fi_index']].copy()

    # 7) Write the results to a CSV file
    df_fii.to_csv('data/interim/FI_index.csv', index=False)
    print("Financial Inclusion Index computed and saved to 'FI_index.csv'")



def clean_instruments_data(): 
    """
    Clean the Instruments data.
    Read from data/raw/Instruments_FII.csv
    Apply specified lags per indicator.
    Save to data/interim/Instruments_FII_lagged.csv
    """

    # 1. Load the raw data
    file_path = 'data/raw/Instruments_FII.csv'
    df = pd.read_csv(file_path)

    # 2. Melt to long format
    df_long = df.melt(
        id_vars=['Country Name', 'Country Code', 'Series Name', 'Series Code'],
        var_name='year_raw',
        value_name='value'
    )

    # 3. Clean value column
    df_long['value'] = pd.to_numeric(df_long['value'], errors='coerce')

    # 4. Extract year
    df_long['year'] = df_long['year_raw'].str.extract(r'(\d{4})').astype(int)

    # 5. Pivot to wide format
    df_wide = df_long.pivot_table(
        index=['Country Name', 'year'],
        columns='Series Name',
        values='value'
    )
    df_wide.columns.name = None
    df_wide = df_wide.reset_index()

    # 6. Define lags per indicator (years)
    lags = {
        "Access to electricity (% of population)": 5,
        "Account ownership at a financial institution or with a mobile-money-service provider (% of population ages 15+)": 5,
        "Commercial bank branches (per 100,000 adults)": 7,
        "Domestic credit to private sector (% of GDP)": 7,
        "Individuals using the Internet (% of population)": 4,
        "Mobile cellular subscriptions (per 100 people)": 4,
        "Urban population (% of total population)": 5
    }

    # 7. Apply lags
    for indicator, lag in lags.items():
        if indicator in df_wide.columns:
            df_wide[f"{indicator} (lag {lag})"] = (
                df_wide
                .groupby('Country Name')[indicator]
                .shift(lag)
            )

    # 8. Drop original (non-lagged) columns
    cols_to_drop = [col for col in lags if col in df_wide.columns]
    df_lagged = df_wide.drop(columns=cols_to_drop)

    # 9. Save result
    df_lagged.to_csv('data/interim/Instruments_FII_lagged.csv', index=False)
    print("Saved lagged instrument data to data/interim/Instruments_FII_lagged.csv")


def main():
    """
    Main function to execute all data cleaning processes
    """
    # Ensure directories exist
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/interim', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    
    # Check for required files in raw directory
    raw_files = os.listdir('data/raw')
    
    print("Files found in data/raw directory:")
    for f in raw_files:
        print(f"  - {f}")
    
    # Clean individual datasets
    if 'FAS.csv' in raw_files:
        try:
            clean_fas_data()
        except Exception as e:
            print(f"Error processing FAS data: {e}")
    else:
        print("FAS.csv not found in data/raw directory, skipping FAS processing")
    
    if 'WGI.csv' in raw_files:
        try:
            clean_wgi_data()
        except Exception as e:
            print(f"Error processing WGI data: {e}")
    else:
        print("WGI.csv not found in data/raw directory, skipping WGI processing")
    
    if 'WDI.csv' in raw_files:
        try:
            clean_wdi_data()
        except Exception as e:
            print(f"Error processing WDI data: {e}")
    else:
        print("WDI.csv not found in data/raw directory, skipping WDI processing")
    
    if 'WID.csv' in raw_files:
        try:
            clean_wid_data()
        except Exception as e:
            print(f"Error processing WID data: {e}")
    else:
        print("WID.csv not found in data/raw directory, skipping WID processing")
    
    if 'GII.csv' in raw_files:
        try:
            clean_gii_data()
        except Exception as e:
            print(f"Error processing GII data: {e}")
    else:
        print("GII.csv not found in data/raw directory, skipping GII processing")
    
    if 'Instruments_FII.csv' in raw_files:
        try:
            clean_instruments_data()
        except Exception as e:
            print(f"Error processing Instruments data: {e}")
    else:
        print("Instruments_FII.csv not found in data/raw directory, skipping Instruments processing")
    
    # Check which datasets were successfully cleaned
    interim_files = []
    try:
        interim_files = os.listdir('data/interim')
        print("\nFiles generated in data/interim directory:")
        for f in interim_files:
            print(f"  - {f}")
    except FileNotFoundError:
        print("No files found in data/interim directory")
    
    # Standardize country names for successful files
    if 'FAS_cleaned.csv' in interim_files:
        try:
            standardize_country_names('data/interim/FAS_cleaned.csv')
        except Exception as e:
            print(f"Error standardizing FAS country names: {e}")
    
    if 'WGI_cleaned.csv' in interim_files:
        try:
            standardize_country_names('data/interim/WGI_cleaned.csv')
        except Exception as e:
            print(f"Error standardizing WGI country names: {e}")
    
    if 'WDI_cleaned.csv' in interim_files:
        try:
            standardize_country_names('data/interim/WDI_cleaned.csv')
        except Exception as e:
            print(f"Error standardizing WDI country names: {e}")
    
    if 'WID_cleaned.csv' in interim_files:
        try:
            standardize_country_names('data/interim/WID_cleaned.csv')
        except Exception as e:
            print(f"Error standardizing WID country names: {e}")
    
    if 'GII_cleaned.csv' in interim_files:
        try:
            standardize_country_names('data/interim/GII_cleaned.csv')
        except Exception as e:
            print(f"Error standardizing GII country names: {e}")
    
    if 'Instruments_Poverty_cleaned.csv' in interim_files:
        try:
            standardize_country_names('data/interim/Instruments_Poverty_cleaned.csv')
        except Exception as e:
            print(f"Error standardizing Instruments country names: {e}")
    
    # Process financial inclusion index if required files exist
    try:
        process_financial_inclusion_index()
    except Exception as e:
        print(f"Error processing financial inclusion index: {e}")
    
    # Merge all available datasets
    try:
        if any(f.endswith('_cleaned.csv') for f in interim_files):
            merge_datasets()
        else:
            print("No cleaned datasets available for merging.")
    except Exception as e:
        print(f"Error merging datasets: {e}")
    
    print("\nData cleaning process completed!")


if __name__ == "__main__":
    main() 