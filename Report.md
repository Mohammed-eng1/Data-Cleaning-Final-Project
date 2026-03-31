# Ames Housing Data Analysis
## ML Foundations Bootcamp — Capstone Project
**Student:** Mohammed Mazen Alsharif
**Dataset:** Ames Housing | **Tools:** Python, Pandas, NumPy, Seaborn, Scikit-learn

---

## 1. Introduction

This project analyses the Ames Housing dataset, which contains 2,930 house sales
in Ames, Iowa with over 80 features covering lot size, quality, neighbourhood,
year built, and sale price. The dataset was chosen because it offers a realistic
mix of numerical and categorical columns along with meaningful messiness —
making it ideal for practising the full data analysis pipeline.

**Questions we aimed to answer:**

- What features most strongly predict house sale price?
- Does the age and quality of a house significantly affect its price?
- Can we engineer features that are more informative than the raw columns?

The project was structured into three phases: data cleaning, feature engineering,
and exploratory data analysis with math basics.

---

## 2. Cleaning Summary

The raw dataset had several problems that needed to be addressed before any
meaningful analysis could begin.

**Problem 1 — Wrong Data Types**
`MS SubClass` was stored as an integer but is actually a category code
(e.g. 20 = 1-story house built after 1946, 60 = 2-story house). Treating it
as a number would imply a false numeric relationship. Similarly, `Mo Sold`
represented a month label (1–12) stored as an integer. Both columns were
converted to their correct types (`str` and `category`).

**Problem 2 — Missing Values**
15 columns had missing data. The strategy differed based on what the missing
value actually meant:

| Column Group | % Missing | Action | Reason |
|---|---|---|---|
| Pool QC, Misc Feature, Alley, Fence | > 80% | Dropped | No useful signal remaining |
| Garage/Bsmt categorical columns | 3–48% | Filled 'None' | NaN means the feature does not exist |
| Garage/Bsmt numeric columns | ~5% | Filled 0 | NaN means no garage or basement |
| Lot Frontage | 17% | Filled median | Numeric and skewed — median is robust |
| Electrical | < 1% | Row dropped | Only 1 row affected |

**Problem 3 — Outliers**
Using the IQR method, 137 extreme `SalePrice` values were detected
(IQR lower: $3,500 / upper: $339,500). Rather than removing rows, values
were capped at the 99th percentile ($456,690) to preserve all records while
limiting the influence of extremes on any future analysis.

**Problem 4 — Duplicates**
No duplicate rows were found in the dataset.

**Validation checks passed:**
- No nulls in key columns (SalePrice, Gr Liv Area, Overall Qual, Lot Area, Year Built)
- All SalePrice values > 0
- Final column count = 78

**Final shape after cleaning: 2,929 rows × 78 columns**

---

## 3. Feature Engineering Summary

Eight new features were created to make the data more useful for analysis.
Each feature was motivated by domain knowledge or statistical reasoning.

**Encoding:**
- `MS Zoning` and `Neighborhood` were one-hot encoded using `pd.get_dummies(drop_first=True)`,
  adding 31 binary columns. This allows models to treat each category independently
  without implying any numeric order.
- `Kitchen Qual` was ordinal encoded (Poor=1, Fair=2, Typical=3, Good=4, Excellent=5)
  to preserve its meaningful rank order.

**Scaling:**
- `Gr Liv Area` and `Lot Area` were standardised using `StandardScaler` (mean=0, std=1)
  so that their large numeric ranges do not dominate distance-based calculations.

**Ratio Features (safe division used to avoid dividing by zero):**
- `price_per_sqft` = SalePrice / Gr Liv Area — buyers compare cost per unit area,
  not absolute price. This ratio normalises for house size.
- `bath_per_bed` = (Full Bath + Half Bath × 0.5) / Bedroom AbvGr — measures the
  comfort level of the house layout, independent of total size.

**Interaction Feature:**
- `qual_x_area` = Overall Qual × Gr Liv Area — captures the joint effect that a
  bigger AND better-quality house is worth significantly more. This became the
  strongest predictor with r = **0.84** against SalePrice.

**Transformation:**
- `log1p(Lot Area)` — Lot Area has a heavy right tail. Applying log1p compresses
  large values and makes the distribution more symmetric, which helps linear models.

**Binning:**
- `House Age Group` — Year Built was binned into five meaningful groups:
  Very Old (before 1940), Old (1940–1970), Mid (1970–1990), Recent (1990–2010), New (after 2010).
  Houses from different eras have different characteristics that the raw year does not capture clearly.

**Redundancy removal:**
- `Yr Sold` was dropped after being identified as having correlation > 0.95 with another feature.

**Final shape after engineering: 2,929 rows × 79 columns**

---

## 4. Key Findings

**Finding 1 — Quality is the strongest price driver**

The boxplot below shows that `SalePrice` rises consistently with `Overall Qual`.
Houses rated quality 10 have a median price nearly 10 times higher than quality 1
houses. The correlation between `Overall Qual` and `SalePrice` is **0.81**, and our
engineered `qual_x_area` feature pushes this to **0.84** — confirming that combining
quality and size captures more signal than either alone.

![SalePrice by Overall Quality and House Age Group](report_chart3.png)

**Finding 2 — New houses cost 2.2x more than very old houses**

The groupby analysis revealed a clear and consistent pricing pattern across age groups:

| House Age Group | Mean Price | Median Price |
|---|---|---|
| New (after 2010) | $283,116 | $267,916 |
| Recent (1990–2010) | $238,730 | $218,689 |
| Mid (1970–1990) | $160,626 | $150,000 |
| Old (1940–1970) | $142,085 | $137,500 |
| Very Old (before 1940) | $129,563 | $122,250 |

New houses average $283k versus $129k for Very Old — a 2.2x difference.
This confirms that the binning decision in Phase 2 was meaningful.

![Mean SalePrice by House Age Group](report_chart2.png)

**Finding 3 — Size and quality jointly drive price (Best Chart)**

The scatter plot below encodes three variables at once: living area (x-axis),
sale price (y-axis), and overall quality (colour). High-quality houses (green)
consistently appear at the top of the price range for any given size, while
low-quality houses (red) cluster at the bottom. This demonstrates that neither
size nor quality alone tells the full story — their interaction is what matters most.

Additionally, 91.8% of high-quality houses (Overall Qual >= 7) sell above the
dataset median of $160,000 — making quality a near-reliable indicator of
above-median performance.

![SalePrice vs Gr Liv Area coloured by Quality](report_chart1.png)

---

## 5. What I Would Do Next

Given more time, I would explore the following directions:

**1 — Neighbourhood deep-dive**
Some neighbourhoods show significantly higher average prices than others.
I would analyse which specific areas drive price the most and whether
location explains price differences beyond quality alone.

**2 — Deeper feature analysis**
I would investigate which combination of rooms, bathrooms, and quality
gives the best value for money — essentially looking for underpriced houses
in the dataset.

**3 — Seasonal analysis**
The dataset includes month and year of sale. I would explore whether there
are seasonal pricing patterns — for example, do houses sell for more in
summer than winter?

**4 — House style comparison**
I would compare 1-story versus 2-story houses of similar size and quality
to see if buyers pay a premium for a specific layout, which could reveal
hidden preferences in the Ames housing market.
