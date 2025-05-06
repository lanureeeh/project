"""
Helper functions for data analysis and visualization.
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import missingno as msno

def plot_missing(df, out_path):
    """
    Visualize missing data in the DataFrame, with special handling for poverty line variables.
    
    When both 'lnpovhead' & 'lnpovhead215' exist, they are colored differently in 
    the missingno matrix and a legend titled "Poverty Lines" is added.
    
    Args:
        df (pd.DataFrame): DataFrame to visualize missing values
        out_path (str): Path to save the figure
        
    Returns:
        None
    """
    # Make a copy of the DataFrame to avoid modifying the original
    df_copy = df.copy()
    
    # Create a figure with appropriate size
    plt.figure(figsize=(12, 8))
    
    # Check if both poverty variables exist
    has_both_pov_vars = 'lnpovhead' in df.columns and 'lnpovhead215' in df.columns
    
    if has_both_pov_vars:
        # Get the column positions in the DataFrame
        cols = list(df.columns)
        pov365_pos = cols.index('lnpovhead')
        pov215_pos = cols.index('lnpovhead215')
        
        # Create a custom color map
        # Default columns will be colored using missingno's default (white/gray)
        # Poverty variables will use custom colors
        color_dict = {}
        
        # The rest of the columns will use missingno's default coloring
        
        # Create the matrix visualization with custom colors
        matrix = msno.matrix(df_copy, figsize=(12, 8), color_dict=color_dict)
        
        # Customize the plot after it's created
        # Get the matrix heatmap from the returned object
        heatmap = matrix.findobj(lambda x: hasattr(x, 'get_facecolors'))[0]
        
        # Get the face colors
        fc = heatmap.get_facecolors()
        
        # Change colors for the poverty variables
        for i in range(len(df)): 
            # Use blue for lnpovhead (3.65 poverty line)
            if not pd.isna(df.iloc[i, pov365_pos]):
                fc[i * len(cols) + pov365_pos] = [0, 0.447, 0.698, 1]  # Blue
            
            # Use red for lnpovhead215 (2.15 poverty line)
            if not pd.isna(df.iloc[i, pov215_pos]):
                fc[i * len(cols) + pov215_pos] = [0.835, 0.369, 0, 1]  # Red
        
        # Set the new face colors
        heatmap.set_facecolors(fc)
        
        # Add a custom legend
        import matplotlib.patches as mpatches
        blue_patch = mpatches.Patch(color=[0, 0.447, 0.698, 1], label='$3.65/day poverty line')
        red_patch = mpatches.Patch(color=[0.835, 0.369, 0, 1], label='$2.15/day poverty line')
        plt.legend(handles=[blue_patch, red_patch], title="Poverty Lines", 
                  loc='upper right', bbox_to_anchor=(1.15, 1))
        
        plt.tight_layout()
    else:
        # If both poverty variables don't exist, use the default missingno matrix
        msno.matrix(df_copy, figsize=(12, 8))
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    # Save the figure
    plt.savefig(out_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    # Return None as specified
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