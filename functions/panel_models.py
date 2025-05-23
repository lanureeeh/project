"""
Panel data econometric models for Latin American economic indicators analysis.

This module provides functions to run various panel data models:
- Fixed effects models
- IV fixed effects models
- Interaction models
- Time trend models

All models are implemented using the linearmodels package.
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from linearmodels.panel import PanelOLS
from linearmodels.iv import IV2SLS
from linearmodels.panel import compare
import statsmodels.api as sm
from statsmodels.tools import add_constant

def run_fe(df, y='lnpovhead', controls=None):
    """
    Run a fixed effects panel regression model.
    
    Args:
        df (pd.DataFrame): Panel dataset with MultiIndex (entity, time)
        y (str): Dependent variable name
        controls (list): List of control variables
        
    Returns:
        PanelOLSResults: Regression results object
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
    
    # Prepare the data
    exog_vars = sm.add_constant(df[controls])
    mod = PanelOLS(df[y], exog_vars, entity_effects=True, time_effects=True)
    
    # Run the model
    res = mod.fit(cov_type='clustered', cluster_entity=True)
    
    # Print summary
    print(f"Fixed Effects Model Results (Dependent Variable: {y})")
    print("=" * 80)
    print(res.summary)
    
    return res

def run_iv_fe(df, y='lnpovhead', instruments=None):
    """
    Run an IV two‐stage least squares with entity and time fixed effects,
    using linearmodels.iv.IV2SLS.

    Args:
        df (pd.DataFrame): Panel dataset with MultiIndex (entity, time)
        y (str): Dependent variable name
        instruments (list): List of instrumental variables

    Returns:
        IVResults: linearmodels IVResults object
    """
    # 1) Default instruments: any lagged financial-inclusion proxies ending in "lag1"
    if instruments is None:
        potential = [
            col for col in df.columns
            if any(x in col.lower() for x in ['mobile', 'atm', 'branch', 'account'])
            and 'lag1' in col.lower()
        ]
        if not potential:
            raise ValueError("No default instruments found. Please specify.")
        instruments = potential[:3]

    # 2) Endogenous and exogenous regressors
    endog_var = 'fi_index'
    if endog_var not in df.columns:
        cand = [c for c in df.columns if 'fii' in c.lower()]
        if not cand:
            raise ValueError(f"Endog {endog_var} not in data.")
        endog_var = cand[0]
    exog_vars = ['lngovt', 'lntradeopen', 'ruleoflaw']
    exog_vars = [x for x in exog_vars if x in df.columns]

    # 3) Bring entity & time back into columns
    if not isinstance(df.index, pd.MultiIndex):
        ent = 'Country' if 'Country' in df.columns else 'ISO3'
        df = df.set_index([ent, 'Year'])
    df_iv = df.reset_index().copy()
    ent_col, time_col = df_iv.columns[0], df_iv.columns[1]

    # 4) Build formula, adding C(entity)+C(time) for FE, plus IV block [endog ~ instruments]
    fe_terms = f" + C({ent_col}) + C({time_col})"
    formula = (
        f"{y} ~ 1 + {' + '.join(exog_vars)}"
        f"{fe_terms} + [{endog_var} ~ {' + '.join(instruments)}]"
    )

    # 5) Fit IV2SLS
    mod = IV2SLS.from_formula(formula, data=df_iv)
    # cluster standard errors by entity
    res = mod.fit(
        cov_type='clustered',
        clusters=df_iv[ent_col].values,  # Use clusters argument instead of cov_config
        debiased=True  # Small‐sample adjustment
    )

    # 6) Output
    print(f"IV2SLS FE‐IV Model Results ({y})")
    print("=" * 80)
    print(res.summary)
    return res

def run_interaction(df, modifier, y='lnpovhead'):
    """
    Run a fixed effects panel regression model with interaction terms.
    
    Args:
        df (pd.DataFrame): Panel dataset with MultiIndex (entity, time)
        modifier (str): Variable to interact with financial inclusion
        y (str): Dependent variable name
        
    Returns:
        PanelOLSResults: Regression results object
    """
    # Set default financial inclusion variable
    fi_var = 'fi_index'  # Financial inclusion index (assumed to exist)
    
    # Check if the financial inclusion variable exists
    if fi_var not in df.columns:
        potential_fi = [col for col in df.columns if any(x in col.lower() for x in 
                                                      ['fii', 'financial', 'inclusion'])]
        if potential_fi:
            fi_var = potential_fi[0]
        else:
            raise ValueError(f"Financial inclusion variable {fi_var} not found in data")
    
    # Check if the modifier variable exists
    if modifier not in df.columns:
        # Try to find variable by partial name match
        name_map = {
            'lngini': ['gini', 'inequality'],
            'gii': ['gender', 'gii', 'gender_inequality'],
            'rulelaw': ['rule', 'law', 'governance', 'wgi']
        }
        
        if modifier in name_map:
            for term in name_map[modifier]:
                matches = [col for col in df.columns if term.lower() in col.lower()]
                if matches:
                    modifier = matches[0]
                    break
            
            if modifier not in df.columns:
                raise ValueError(f"Modifier variable {modifier} not found in data")
    
    # Ensure the DataFrame has proper MultiIndex
    if not isinstance(df.index, pd.MultiIndex):
        try:
            # Assume Country/ISO3 and Year columns exist
            entity_col = 'Country' if 'Country' in df.columns else 'ISO3'
            df = df.set_index([entity_col, 'Year'])
        except Exception as e:
            raise ValueError(f"DataFrame must have MultiIndex or Country/ISO3 and Year columns: {e}")
    
    # Create a copy to avoid modifying the original
    df_mod = df.copy()
    
    # Create interaction term
    interaction_var = f"{fi_var}_{modifier}"
    df_mod[interaction_var] = df_mod[fi_var] * df_mod[modifier]
    
    # Define control variables
    controls = ['lngovt', 'lntradeopen', 'ruleoflaw']
    
    # Filter out any controls that don't exist in the data
    controls = [c for c in controls if c in df_mod.columns]
    
    # Prepare the data for the model
    exog_vars = [fi_var, modifier, interaction_var] + controls
    
    # Create the model
    mod = PanelOLS(df_mod[y], df_mod[exog_vars], entity_effects=True, time_effects=True)
    res = mod.fit(cov_type='clustered', cluster_entity=True)
    
    # Print summary
    print(f"Interaction Model Results (Dependent Variable: {y}, Modifier: {modifier})")
    print("=" * 80)
    print(res.summary)
    
    # Create and save marginal effects plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Get the range of the modifier variable
    modifier_min = df_mod[modifier].min()
    modifier_max = df_mod[modifier].max()
    modifier_range = np.linspace(modifier_min, modifier_max, 100)
    
    # Calculate marginal effect and standard errors
    beta_fi = res.params[fi_var]
    beta_int = res.params[interaction_var]
    se_fi = res.std_errors[fi_var]
    se_int = res.std_errors[interaction_var]
    
    # Calculate marginal effects and confidence intervals
    marg_effect = beta_fi + beta_int * modifier_range
    marg_effect_se = np.sqrt(se_fi**2 + (modifier_range**2) * (se_int**2) + 2 * modifier_range * 0)  # Covariance assumed 0 for simplicity
    
    # Upper and lower 95% confidence intervals
    upper_ci = marg_effect + 1.96 * marg_effect_se
    lower_ci = marg_effect - 1.96 * marg_effect_se
    
    # Plot the marginal effects
    ax.plot(modifier_range, marg_effect, 'b-', label='Marginal Effect')
    ax.fill_between(modifier_range, lower_ci, upper_ci, color='b', alpha=0.2, label='95% CI')
    ax.axhline(y=0, color='r', linestyle='--')
    
    # Add labels and title
    ax.set_xlabel(modifier)
    ax.set_ylabel(f'Marginal Effect of {fi_var} on {y}')
    ax.set_title(f'Marginal Effect of {fi_var} on {y} as {modifier} Changes')
    ax.legend()
    
    # Create directory if it doesn't exist
    os.makedirs('reports/figures', exist_ok=True)
    
    # Save the figure
    output_path = f'reports/figures/marginal_effect_{fi_var}_{modifier}_{y}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Marginal effects plot saved to {output_path}")
    
    return res

def run_fe_trend(df, y='lnpovhead'):
    """
    Run a fixed effects panel regression model with time trend.
    
    Args:
        df (pd.DataFrame): Panel dataset with MultiIndex (entity, time)
        y (str): Dependent variable name
        
    Returns:
        PanelOLSResults: Regression results object
    """
    # Ensure the DataFrame has proper MultiIndex
    if not isinstance(df.index, pd.MultiIndex):
        try:
            # Assume Country/ISO3 and Year columns exist
            entity_col = 'Country' if 'Country' in df.columns else 'ISO3'
            df = df.set_index([entity_col, 'Year'])
        except Exception as e:
            raise ValueError(f"DataFrame must have MultiIndex or Country/ISO3 and Year columns: {e}")
    
    # Create a copy to avoid modifying the original
    df_trend = df.copy()
    
    # Set default financial inclusion variable
    fi_var = 'fi_index'  # Financial inclusion index (assumed to exist)
    
    # Check if the financial inclusion variable exists
    if fi_var not in df_trend.columns:
        potential_fi = [col for col in df_trend.columns if any(x in col.lower() for x in 
                                                            ['fii', 'financial', 'inclusion'])]
        if potential_fi:
            fi_var = potential_fi[0]
        else:
            raise ValueError(f"Financial inclusion variable {fi_var} not found in data")
    
    # Extract the years and normalize to start at 0
    years = df_trend.index.get_level_values('Year').astype(int)
    trend = years - years.min()
    df_trend['trend'] = trend
    
    # Create interaction term
    df_trend[f'{fi_var}_trend'] = df_trend[fi_var] * df_trend['trend']
    
    # Define control variables
    controls = ['lngovt', 'lntradeopen', 'ruleoflaw']
    
    # Filter out any controls that don't exist in the data
    controls = [c for c in controls if c in df_trend.columns]
    
    # Prepare the data for the model
    exog_vars = [fi_var, 'trend', f'{fi_var}_trend'] + controls
    
    # Create the model
    mod = PanelOLS(df_trend[y], df_trend[exog_vars], entity_effects=True)
    res = mod.fit(cov_type='clustered', cluster_entity=True)
    
    # Print summary
    print(f"Fixed Effects with Time Trend Model Results (Dependent Variable: {y})")
    print("=" * 80)
    print(res.summary)
    
    return res

def compare_models(results_dict):
    """
    Compare multiple models side by side.
    
    Args:
        results_dict (dict): Dictionary of model results with format {model_name: results_object}
        
    Returns:
        linearmodels.panel.results.Comparison: Comparison results
    """
    comparison = compare(results_dict)
    print(comparison)
    return comparison 