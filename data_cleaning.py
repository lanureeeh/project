import pandas as pd
import re

def clean_fas_data():
    # Read the FAS.csv file
    file_path = 'raw_data/FAS.csv'
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
    df_long.to_csv('clean_data/FAS_cleaned.csv', index=False)
    print("FAS.csv cleaning complete. 'FAS_cleaned.csv' has been created.")




def clean_wgi_data():
    # Read the WGI.csv file
    file_path = 'raw_data/WGI.csv'
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
    data_cleaned.to_csv('clean_data/WGI_cleaned.csv', index=False)

    print("WGI.csv cleaning complete. 'WGI_cleaned.csv' has been created.")


def clean_wdi_data():
    # Read the WDI.csv file
    file_path = 'raw_data/WDI.csv'
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
    data.to_csv('clean_data/WDI_cleaned.csv', index=False)

    print("WDI.csv cleaning complete. 'WDI_cleaned.csv' has been created.")


def clean_wid_data():
    # Read the WID.csv file
    file_path = 'raw_data/WID.csv'
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
    data_pivoted.to_csv('clean_data/WID_cleaned.csv')

    print("WID.csv cleaning complete. 'WID_cleaned.csv' has been created.")


def standardize_country_names(file_path):
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
    else:
        raise KeyError("No country column found in the dataset.")

    # Print the column names for debugging
    print(f"Columns in {file_path}: {data.columns.tolist()}")
    
    # Print the first few rows before standardization for debugging
    print(f"Data before standardization in {file_path}:")
    print(data.head())

    # Handle countries in WGI dataset differently
    if country_column == 'countryname':
        # Only standardize countries that match our list
        filtered_countries = ['Mexico', 'Peru', 'Belize', 'Costa Rica', 'El Salvador', 
                           'Guatemala', 'Honduras', 'Nicaragua', 'Panama', 'Cuba', 
                           'Dominican Republic', 'Haiti', 'Puerto Rico', 'Argentina', 
                           'Bolivia', 'Brazil', 'Chile', 'Colombia', 'Ecuador', 
                           'Guyana', 'Paraguay', 'Suriname', 'Uruguay', 'Venezuela']
        
        # Create a mask for rows that have countries in our list
        mask = data[country_column].isin(filtered_countries)
        
        # Apply mapping only to those rows
        data.loc[mask, country_column] = data.loc[mask, country_column].map(country_name_mapping)
    else:
        # Standard mapping for other datasets
        data[country_column] = data[country_column].map(country_name_mapping)

    # Print the first few rows after standardization for debugging
    print(f"Data after standardization in {file_path}:")
    print(data.head())

    # Save the updated data back to the CSV file
    data.to_csv(file_path, index=False)

    print(f"Country names standardized in {file_path}.")


def merge_datasets():
    # Read all cleaned datasets
    fas = pd.read_csv('clean_data/FAS_cleaned.csv')
    wgi = pd.read_csv('clean_data/WGI_cleaned.csv')
    wdi = pd.read_csv('clean_data/WDI_cleaned.csv')
    wid = pd.read_csv('clean_data/WID_cleaned.csv')
    
    print("Reading all cleaned datasets...")
    
    # Define the years range from 1975 to 2024
    years = list(range(1975, 2025))
    
    # Get the list of target countries
    target_countries = [
        'Mexico', 'Peru', 'Belize', 'Costa Rica', 'El Salvador', 'Guatemala',
        'Honduras', 'Nicaragua', 'Panama', 'Cuba', 'Dominican Republic', 'Haiti',
        'Puerto Rico', 'Argentina', 'Bolivia', 'Brazil', 'Chile', 'Colombia',
        'Ecuador', 'Guyana', 'Paraguay', 'Suriname', 'Uruguay', 'Venezuela'
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
    merged_data['Inequality Index'] = float('nan')
    
    print("Created empty dataset with target variables...")
    
    # Add data from WDI dataset (contains most of the economic indicators)
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
    print("Adding WGI data...")
    for _, row in wgi.iterrows():
        country = row['countryname']
        if country in target_countries:
            year = row['year']
            if year in years:
                # Find the corresponding row in merged_data
                mask = (merged_data['Country'] == country) & (merged_data['Year'] == year)
                # Add Rule of Law estimate
                if not pd.isna(row['estimate']):
                    merged_data.loc[mask, 'Rule of Law - estimate'] = row['estimate']
                # Add Rule of Law percentile rank
                if not pd.isna(row['pctrank']):
                    merged_data.loc[mask, 'Rule of Law - pctrank'] = row['pctrank']
    
    # Add data from WID dataset (Inequality Index)
    print("Adding WID data...")
    for idx, row in wid.iterrows():
        country = idx
        if country in target_countries:
            for year in years:
                if str(year) in wid.columns:
                    value = row[str(year)]
                    # Find the corresponding row in merged_data
                    mask = (merged_data['Country'] == country) & (merged_data['Year'] == year)
                    if not pd.isna(value):
                        merged_data.loc[mask, 'Inequality Index'] = value
    
    # Add any relevant data from FAS dataset
    print("Adding FAS data...")
    # Process if there are any variables from FAS that need to be added
    
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
    
    # Save the merged dataset
    merged_data.to_csv('merged_dataset_1975_2024.csv', index=False)
    print("Merged dataset saved to 'merged_dataset_1975_2024.csv'")


def main():
    clean_fas_data()
    clean_wgi_data()
    clean_wdi_data()
    clean_wid_data()
    
    # Standardize country names in all cleaned datasets
    standardize_country_names('clean_data/FAS_cleaned.csv')
    standardize_country_names('clean_data/WGI_cleaned.csv')
    standardize_country_names('clean_data/WDI_cleaned.csv')
    standardize_country_names('clean_data/WID_cleaned.csv')
    
    # Merge all datasets
    merge_datasets()


if __name__ == "__main__":
    main() 