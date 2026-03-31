# Ames Housing Data Analysis
ML Foundations Bootcamp — Capstone Project

## Overview
Exploratory data analysis pipeline built with Python and Jupyter Notebooks.
Covers data cleaning, feature engineering, EDA visualisation, and math basics
on the Ames Housing dataset (2,930 house sales, 80+ features).

## Project Structure
```
ames-housing-capstone/
├── 01_cleaning.ipynb      # Phase 1 — Load, explore & clean
├── 02_features.ipynb      # Phase 2 — Feature engineering
├── 03_eda.ipynb           # Phase 3 — EDA & visualisation
├── 04_math.ipynb          # Phase 3 — Math basics
├── AmesHousing.csv        # Raw dataset
├── ames_cleaned.csv       # Cleaned dataset
├── ames_features.csv      # Engineered features
├── report.pdf             # Written report (3 pages)
└── requirements.txt       # Python dependencies
```

## Phases

### Phase 1 — Data Cleaning
- Fixed wrong data types (MS SubClass, Mo Sold)
- Handled 15 columns with missing values
- Detected and capped 137 outliers using IQR method
- Final shape: 2,929 rows × 78 columns

### Phase 2 — Feature Engineering
- One-hot encoded MS Zoning and Neighborhood
- Ordinal encoded Kitchen Qual (1–5)
- Scaled Gr Liv Area and Lot Area with StandardScaler
- Created ratio features: price_per_sqft, bath_per_bed
- Created interaction feature: qual_x_area (r = 0.84 with SalePrice)
- Log-transformed Lot Area to reduce right skew
- Binned Year Built into 5 age groups
- Final shape: 2,929 rows × 79 columns

### Phase 3 — EDA & Math
- Histograms, boxplots, correlation heatmap, scatter plot
- Groupby analysis by house age group
- Manual mean, std, z-score, cosine similarity, probability estimation

## Key Findings
1. Quality is the strongest price driver — quality 10 houses cost ~10x quality 1
2. New houses average $283k vs $129k for very old houses (2.2x difference)
3. 91.8% of high-quality houses sell above the median price

## Setup
```bash
pip install -r requirements.txt
jupyter notebook
```

## Dataset
[Ames Housing Dataset — Kaggle](https://www.kaggle.com/datasets/prevek18/ames-housing-dataset)
