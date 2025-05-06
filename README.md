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

| Dataset | Source | Description | Variables Used |
|---------|--------|-------------|----------------|
| Financial Access Survey (FAS) | IMF | Comprehensive data on financial inclusion | ATM density, bank branches, account ownership |
| World Development Indicators (WDI) | World Bank | Socioeconomic indicators | Poverty headcount ratios ($3.65/day and $2.15/day*), GDP growth, government expenditure, trade |
| World Governance Indicators (WGI) | World Bank | Institutional quality measures | Rule of law estimates |
| Gender Inequality Index (GII) | UNDP | Gender-based inequalities | Gender inequality indices |
| World Inequality Database (WID) | WID.world | Income and wealth inequality data | Gini coefficients, income shares |

*The $2.15/day poverty line was not used in the final analysis due to significant missing data (see [missing_poverty.png](reports/figures/missing_poverty.png))

## Project Structure

```
├── data/               # All data files
│   ├── raw/            # Original, immutable data
│   ├── interim/        # Intermediate data
│   └── processed/      # Final, canonical data sets
│
├── notebooks/          # Jupyter notebooks
│   ├── eda/            # Exploratory data analysis
│   └── modeling/       # Modeling notebooks
│
├── src/                # Source code
│   ├── data/           # Data processing scripts
│   ├── features/       # Feature engineering scripts
│   ├── models/         # Model training and prediction
│   └── utils/          # Utility functions
│
├── reports/            # Generated analysis
│   ├── figures/        # Generated graphics and figures
│   ├── slides/         # Presentation materials
│   └── logs/           # Logs and outputs
│
├── time_log.md         # Time tracking log
├── README.md           # Project description
├── environment.yml     # Environment configuration
└── .gitignore          # Git ignore file
```

## Quick-Start Guide

Follow these steps to set up the environment and run the analysis:

```bash
# Create and activate the conda environment
conda env create -f environment.yml
conda activate latam-fininc

# Process data and build panel dataset
python -m src.data.build_panel

# Build the Financial Inclusion Index
python -m src.features.build_fii

# Start Jupyter Lab to explore notebooks
jupyter lab
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
- Instrumental Variable estimation
- Interaction models
- Threshold regression with bootstrap confidence intervals

For detailed methodology and results, please refer to the notebooks in the `notebooks/modeling/` directory.
