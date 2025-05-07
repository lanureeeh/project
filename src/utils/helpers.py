"""
Helper functions for data analysis and visualization.
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno

def plot_missing(df, out_path):
    """
    Visualize missing data in the DataFrame with a heatmap showing data availability by country and year.
    
    If the DataFrame contains panel data with 'Country' and 'Year' columns, it will create a heatmap
    showing data availability for each country across years for the selected variables.
    
    Args:
        df (pd.DataFrame): DataFrame to visualize missing values, should contain 'Country' and 'Year' columns
        out_path (str): Path to save the figure
        
    Returns:
        None
    """
    # Check if the DataFrame has Country and Year columns for panel data
    if 'Country' in df.columns and 'Year' in df.columns:
        # Identify the variables to visualize (exclude Country and Year)
        data_vars = [col for col in df.columns if col not in ['Country', 'Year']]
        
        if len(data_vars) == 0:
            print("No data variables found besides Country and Year")
            return None
        
        # For each variable, create a heatmap showing data availability
        for var in data_vars:
            # Create a pivot table with countries as rows and years as columns
            pivot_df = df.pivot_table(
                index='Country', 
                columns='Year', 
                values=var,
                aggfunc=lambda x: 1.0 if pd.notnull(x).any() else 0.0
            )
            
            # Create the figure
            plt.figure(figsize=(15, 10))
            
            # Create the heatmap
            sns.heatmap(
                pivot_df, 
                cmap='Blues', 
                cbar_kws={'label': 'Data Available'},
                vmin=0, 
                vmax=1
            )
            
            # Set title and labels
            title = f"Data Availability for {var.replace('lnpovhead215', 'Poverty headcount ratio at $2.15 a day').replace('lnpovhead', 'Poverty headcount ratio at $3.65 a day')} ({df['Year'].min()}-{df['Year'].max()})"
            plt.title(title, fontsize=14)
            plt.xlabel('Year', fontsize=12)
            plt.ylabel('Country', fontsize=12)
            plt.tight_layout()
            
            # Ensure the directory exists
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            
            # Create a variable-specific output path
            var_out_path = out_path.replace('.png', f'_{var}.png')
            
            # Save the figure
            plt.savefig(var_out_path, bbox_inches='tight', dpi=300)
            plt.close()
        
        return None
    
    # If not a panel dataset, use the default missingno matrix
    plt.figure(figsize=(12, 8))
    msno.matrix(df, figsize=(12, 8))
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    # Save the figure
    plt.savefig(out_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    return None

def vif_table(df, cols, out_path="reports/tables/vif.xlsx"):
    """
    Calculate Variance Inflation Factors for specified columns and export to Excel.
    
    Args:
        df (pd.DataFrame): Input data
        cols (list): List of column names to calculate VIF for
        out_path (str, optional): Path to save the Excel file. Defaults to "reports/tables/vif.xlsx".
    
    Returns:
        pd.DataFrame: DataFrame containing VIF values
    """
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    
    # Create a DataFrame with only the desired columns
    X = df[cols].dropna()
    
    # Calculate VIF for each feature
    vif_data = pd.DataFrame()
    vif_data["Variable"] = cols
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    
    # Sort by VIF (highest first)
    vif_data = vif_data.sort_values("VIF", ascending=False).reset_index(drop=True)
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    # Export to Excel
    vif_data.to_excel(out_path, index=False)
    
    return vif_data

def export_reg_table(res_dict, tag, folder="reports/tables"):
    """
    Export regression results to tables in Excel and LaTeX formats.
    
    Args:
        res_dict (dict): Dictionary of regression results
            Keys are model names, values are statsmodels results objects
        tag (str): Identifier for the output files
        folder (str, optional): Folder to save output files. Defaults to "reports/tables".
    
    Returns:
        None
    """
    import statsmodels.api as sm
    
    # Ensure the output directory exists
    os.makedirs(folder, exist_ok=True)
    
    # Combine results into a summary table
    results_table = sm.summary2.summary_col(
        results=list(res_dict.values()),
        float_format='%0.3f',
        stars=True,
        model_names=list(res_dict.keys()),
        regressor_order=[v.model.exog_names for v in res_dict.values()][0]
    )
    
    # Get the summary as a DataFrame
    results_df = results_table.tables[0]
    
    # Export to Excel
    excel_path = os.path.join(folder, f"regression_{tag}.xlsx")
    results_df.to_excel(excel_path)
    
    # Export to LaTeX
    latex_path = os.path.join(folder, f"regression_{tag}.tex")
    
    # Create a more publication-ready LaTeX table
    latex_table = results_table.as_latex()
    
    # Replace some LaTeX formatting for better output
    latex_table = latex_table.replace('\\begin{center}', '\\begin{table}[htbp]\n\\centering')
    latex_table = latex_table.replace('\\end{center}', '\\caption{Regression Results}\n\\label{tab:reg_' + tag + '}\n\\end{table}')
    
    # Write the LaTeX table to a file
    with open(latex_path, 'w') as f:
        f.write(latex_table)
    
    return None 