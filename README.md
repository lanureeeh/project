# Financial Inclusion and Poverty Reduction in Latin America

## Business Problem Statement

This project investigates the relationship between financial inclusion and poverty reduction across Latin American countries, with a particular focus on how this relationship is moderated by inequality, gender disparities, and institutional quality.

The key research questions we address are:

1. Does increased financial inclusion lead to a reduction in poverty levels in Latin American countries?
2. How do factors such as inequality, gender inequality, and rule of law affect the impact of financial inclusion on poverty reduction?
3. Are there threshold effects in inequality beyond which financial inclusion becomes less effective at reducing poverty?

These questions are crucial for policymakers and development organizations to design more effective strategies that maximize the poverty-reduction impact of financial inclusion initiatives in diverse socioeconomic contexts.

## Data Sources

The analysis integrates data from multiple international sources covering Latin American countries from 2003 to 2023:

| Dataset                            | Source     | Description                               | Variables Used                                                                                 |
| ---------------------------------- | ---------- | ----------------------------------------- | ---------------------------------------------------------------------------------------------- |
| Financial Access Survey (FAS)      | IMF        | Comprehensive data on financial inclusion | ATM density, bank branches, account ownership                                                  |
| World Development Indicators (WDI) | World Bank | Socioeconomic indicators                  | Poverty headcount ratios ($3.65/day and $2.15/day*), GDP growth, government expenditure, trade |
| World Governance Indicators (WGI)  | World Bank | Institutional quality measures            | Rule of law estimates                                                                          |
| Gender Inequality Index (GII)      | UNDP       | Gender-based inequalities                 | Gender inequality indices                                                                      |
| World Inequality Database (WID)    | WID.world  | Income and wealth inequality data         | Gini coefficients, income shares                                                               |

## Project Structure

```
├── data/               # All data files
│   ├── raw/            # Original, immutable data
│   ├── interim/        # Intermediate data (cleaned individual datasets)
│   └── processed/      # Final, canonical data sets (merged and panel datasets)
│
├── functions/          # Core functionality modules
│   ├── data_cleaning.py    # Data cleaning and processing functions
│   ├── build_panel.py      # Panel dataset creation with missing value imputation
│   ├── panel_models.py     # Econometric models for panel data analysis
│   └── helpers.py          # Utility functions for visualization and analysis
│
├── reports/            # Generated analysis
│   └── figures/        # Generated graphics and figures
│
├── complete_project_notebook.ipynb  # Consolidated analysis notebook
├── README.md           # Project description
└── environment.yml     # Environment configuration
```

## Data Processing Pipeline

The data processing workflow follows these steps:

1. **Raw Data Cleaning**: `data_cleaning.py` cleans individual datasets from different sources and saves them to the interim directory.
2. **Dataset Merging**: The cleaned datasets are merged into a comprehensive dataset spanning 1975-2024.
3. **Panel Construction**: `build_panel.py` creates a filtered panel dataset for 2003-2023 with:

   - Variable transformations (log transforms, renamed variables)
   - Country and time period filtering
   - Missing value imputation using:
     - Linear interpolation within country time series
     - KNN imputation based on similar country clusters
     - Global median imputation as a last resort
4. **Model Estimation**: `panel_models.py` provides functions for:

   - Fixed Effects (FE) panel models
   - Instrumental Variable (IV) estimation
   - Interaction models with moderator variables
   - Fixed effects models with time trends

## Quick-Start Guide

Follow these steps to set up the environment and run the analysis:

```bash
# Create and activate the conda environment
conda env create -f environment.yml
conda activate project-env

# Process the data pipeline (runs cleaning, merging, and panel creation)
python -c "from functions.data_cleaning import main; main()"
python -c "from functions.build_panel import build_panel; build_panel()"

# Open the consolidated notebook
jupyter lab complete_project_notebook.ipynb
```

## Key Findings

Our analysis reveals several important insights:

1. Financial inclusion generally has a positive impact on poverty reduction across Latin American countries.
2. The effect of financial inclusion on poverty reduction is moderated by:

   - Income inequality (Gini coefficient)
   - Gender inequality (GII)
   - Rule of law
3. We identified a threshold level of inequality beyond which the poverty-reduction benefits of financial inclusion diminish significantly.
4. The effectiveness of financial inclusion policies varies over time and across countries, suggesting the need for context-specific approaches.

## Methodology

The project employs several econometric approaches:

- Fixed Effects panel models
- Instrumental Variable estimation with lagged variables
- Interaction models to test moderating effects
- Time trend analysis to account for temporal dynamics

For detailed methodology and results, please refer to the `complete_project_notebook.ipynb` notebook which contains the consolidated analysis workflow.
