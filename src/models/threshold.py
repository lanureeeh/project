"""
Threshold regression for poverty reduction analysis.

This module implements threshold regression with grid search over
percentiles of inequality (lngini) to identify potential nonlinear
relationships in poverty reduction.

$2.15 line dropped due to missingness (see missing_poverty.png).
"""
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from linearmodels import PanelOLS
from tqdm import tqdm

def find_threshold(df, threshold_var='lngini', y='lnpovhead', controls=None, 
                   percentiles=range(10, 91, 1), bootstrap_reps=500, 
                   save_dir='reports'):
    """
    Find threshold in panel data using grid search over percentiles.
    
    Implements a two-regime threshold regression model by searching over
    percentiles of the threshold variable (lngini) and selecting the 
    threshold that minimizes the sum of squared residuals.
    
    Args:
        df (pd.DataFrame): Panel dataset with MultiIndex (entity, time)
        threshold_var (str): Variable to use for threshold (default: 'lngini')
        y (str): Dependent variable (default: 'lnpovhead')
        controls (list): Control variables (default: lngovt, lntradeopen, ruleoflaw)
        percentiles (range): Percentiles to search over (default: 10-90 by 1)
        bootstrap_reps (int): Number of bootstrap replications (default: 500)
        save_dir (str): Directory to save results (default: 'reports')
        
    Returns:
        dict: Dictionary of results including optimal threshold, coefficients,
              bootstrap confidence intervals, and SSR values.
    """
    # Set default controls if not provided
    if controls is None:
        controls = ['lngovt', 'lntradeopen', 'ruleoflaw']
    
    # Ensure the DataFrame has proper MultiIndex
    if not isinstance(df.index, pd.MultiIndex):
        try:
            # Assume Country/ISO3 and Year columns exist
            entity_col = 'Country' if 'Country' in df.columns else 'ISO3'
            df = df.set_index([entity_col, 'Year'])
        except Exception as e:
            raise ValueError(f"DataFrame must have MultiIndex or Country/ISO3 and Year columns: {e}")
    
    # Make a copy of the dataframe to avoid modifying the original
    data = df.copy()
    
    # Drop missing values in the relevant variables
    mask = ~data[([y, threshold_var] + controls)].isna().any(axis=1)
    data = data.loc[mask]
    
    # Set default financial inclusion variable
    fi_var = None
    for col in data.columns:
        if any(term in col.lower() for term in ['fi_index', 'financial', 'inclusion']):
            fi_var = col
            break
    
    if fi_var is None:
        raise ValueError("No financial inclusion variable found in data.")
    
    print(f"Using '{fi_var}' as financial inclusion variable")
    print(f"Using '{threshold_var}' as threshold variable")
    print(f"Using '{y}' as outcome variable")
    
    # Calculate threshold values for each percentile
    thresholds = [np.percentile(data[threshold_var].dropna(), p) for p in percentiles]
    
    # Dictionary to store SSR for each threshold
    ssr_results = {}
    models = {}
    
    # Grid search over thresholds
    print(f"Grid searching over {len(thresholds)} thresholds...")
    for i, threshold in enumerate(thresholds):
        # Create indicator for threshold
        data['below_threshold'] = (data[threshold_var] <= threshold).astype(float)
        data['above_threshold'] = (data[threshold_var] > threshold).astype(float)
        
        # Create interaction terms
        data[f'{fi_var}_below'] = data[fi_var] * data['below_threshold']
        data[f'{fi_var}_above'] = data[fi_var] * data['above_threshold']
        
        # Create model formula with two regimes
        exog_vars = [f'{fi_var}_below', f'{fi_var}_above'] + controls
        
        # Create model
        model = PanelOLS(data[y], data[exog_vars], entity_effects=True, time_effects=True)
        results = model.fit(cov_type='clustered', cluster_entity=True)
        
        # Store SSR and model
        ssr = results.resid.T @ results.resid
        ssr_results[threshold] = ssr
        models[threshold] = results
        
        # Print progress
        if (i + 1) % 10 == 0:
            print(f"Completed {i + 1}/{len(thresholds)} thresholds")
    
    # Find threshold with minimum SSR
    optimal_threshold = min(ssr_results, key=ssr_results.get)
    optimal_percentile = percentiles[thresholds.index(optimal_threshold)]
    
    # Get the results for the optimal threshold
    optimal_model = models[optimal_threshold]
    
    print("\nOptimal threshold found:")
    print(f"Threshold value: {optimal_threshold:.4f}")
    print(f"Corresponds to {optimal_percentile}th percentile of {threshold_var}")
    print("\nModel results:")
    print(optimal_model.summary)
    
    # Bootstrap confidence intervals for the threshold
    print(f"\nRunning bootstrap with {bootstrap_reps} replications...")
    bootstrap_thresholds = []
    
    # Entity-block bootstrap
    entities = data.index.get_level_values(0).unique()
    
    # Use tqdm for a progress bar
    for _ in tqdm(range(bootstrap_reps)):
        # Sample entities with replacement
        sampled_entities = np.random.choice(entities, size=len(entities), replace=True)
        bootstrap_sample = pd.concat([data.loc[entity] for entity in sampled_entities])
        
        # Set proper index
        index_names = data.index.names
        bootstrap_sample.index.names = index_names
        
        # Run grid search on the bootstrap sample
        bootstrap_ssr = {}
        for threshold in thresholds:
            bootstrap_sample['below_threshold'] = (bootstrap_sample[threshold_var] <= threshold).astype(float)
            bootstrap_sample['above_threshold'] = (bootstrap_sample[threshold_var] > threshold).astype(float)
            
            bootstrap_sample[f'{fi_var}_below'] = bootstrap_sample[fi_var] * bootstrap_sample['below_threshold']
            bootstrap_sample[f'{fi_var}_above'] = bootstrap_sample[fi_var] * bootstrap_sample['above_threshold']
            
            exog_vars = [f'{fi_var}_below', f'{fi_var}_above'] + controls
            
            model = PanelOLS(bootstrap_sample[y], bootstrap_sample[exog_vars], 
                            entity_effects=True, time_effects=True)
            results = model.fit(cov_type='clustered', cluster_entity=True)
            
            bootstrap_ssr[threshold] = results.resid.T @ results.resid
        
        # Find threshold with minimum SSR in this bootstrap
        bootstrap_optimal = min(bootstrap_ssr, key=bootstrap_ssr.get)
        bootstrap_thresholds.append(bootstrap_optimal)
    
    # Calculate bootstrap confidence intervals
    bootstrap_thresholds.sort()
    ci_lower = bootstrap_thresholds[int(0.025 * bootstrap_reps)]
    ci_upper = bootstrap_thresholds[int(0.975 * bootstrap_reps)]
    
    print(f"Bootstrap 95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
    
    # Prepare results dictionary
    results = {
        'optimal_threshold': float(optimal_threshold),
        'optimal_percentile': int(optimal_percentile),
        'bootstrap_ci': [float(ci_lower), float(ci_upper)],
        'coefficients': {
            f'{fi_var}_below': float(optimal_model.params[f'{fi_var}_below']),
            f'{fi_var}_above': float(optimal_model.params[f'{fi_var}_above']),
        },
        'ssr_values': {str(float(t)): float(ssr) for t, ssr in ssr_results.items()},
        'threshold_variable': threshold_var,
        'outcome_variable': y,
        'controls': controls,
        'bootstrap_reps': bootstrap_reps
    }
    
    # Create the directory if it doesn't exist
    os.makedirs(f'{save_dir}/tables', exist_ok=True)
    os.makedirs(f'{save_dir}/figures', exist_ok=True)
    
    # Save results to JSON
    with open(f'{save_dir}/tables/threshold_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    # Create figure
    plt.figure(figsize=(12, 6))
    
    # Convert to list of tuples for sorting
    ssr_items = [(float(threshold), float(ssr)) for threshold, ssr in ssr_results.items()]
    ssr_items.sort(key=lambda x: x[0])
    
    # Extract sorted values
    thresholds_sorted = [t for t, _ in ssr_items]
    ssr_sorted = [s for _, s in ssr_items]
    
    # Plot SSR values
    plt.plot(thresholds_sorted, ssr_sorted, 'b-', linewidth=2)
    plt.axvline(x=optimal_threshold, color='r', linestyle='--', 
                label=f'Optimal threshold: {optimal_threshold:.4f}')
    
    # Add shaded area for bootstrap CI
    plt.axvspan(ci_lower, ci_upper, alpha=0.2, color='gray', label='95% Bootstrap CI')
    
    # Labels and title
    plt.xlabel(f'Threshold Value of {threshold_var}')
    plt.ylabel('Sum of Squared Residuals')
    plt.title(f'Grid Search Results for Threshold in {threshold_var}\nOutcome: {y}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save figure
    plt.savefig(f'{save_dir}/figures/threshold_search_{threshold_var}_{y}.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Results saved to {save_dir}/tables/threshold_results.json")
    print(f"Figure saved to {save_dir}/figures/threshold_search_{threshold_var}_{y}.png")
    
    return results

if __name__ == "__main__":
    # Example usage
    print("Loading data...")
    try:
        # Attempt to load the panel dataset
        panel = pd.read_csv("data/processed/panel_2003_2023.csv")
        
        # Run threshold regression
        find_threshold(panel, threshold_var='lngini', y='lnpovhead', 
                      bootstrap_reps=500, save_dir='reports')
        
    except Exception as e:
        print(f"Error running threshold regression: {e}") 