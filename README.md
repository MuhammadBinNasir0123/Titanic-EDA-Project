# Titanic EDA Project

An exploratory data analysis of the Titanic passenger dataset to uncover survival patterns across gender, class, and age.

---

## Project Overview

This project focuses on cleaning and analyzing the Titanic dataset to answer one core question: **who was more likely to survive, and why?**

The analysis covers:
- Data cleaning and preparation
- Survival rate breakdowns by gender, class, and age
- Statistical correlations with survival
- Auto-generated summary report with key findings

---

## Key Insights

| Finding | Value |
|---|---|
| Overall Survival Rate | 38.4% |
| Female Survival Rate | 74.2% |
| Male Survival Rate | 18.9% |
| Gender Survival Gap | 55.3% |
| 1st Class Survival Rate | 63.0% |
| 3rd Class Survival Rate | 24.2% |
| Class Survival Gap | 38.7% |
| Highest Survival Group | Women in Class 1 (96.8%) |
| Lowest Survival Group | Men in Class 3 (13.5%) |

---

## Dataset

- **Source:** Kaggle Titanic Passenger Dataset
- **File:** `Titanic-cleaned.csv` — cleaned version used in analysis
- **Size:** 891 passengers × 10 columns

---

## Tools & Libraries

| Tool | Purpose |
|---|---|
| Python 3.8+ | Core language |
| pandas | Data cleaning and analysis |
| numpy | Numerical operations |
| matplotlib | Visualizations |
| seaborn | Statistical charts |
| PyCharm | Development environment |

---

## Statistical Findings

**Strongest correlations with survival:**
- Passenger Class: −0.338 (strong negative — lower class, lower survival)
- Family size (Parch): +0.082 (weak positive)
- Age: −0.070 (weak negative)

**Passenger breakdown:**
- 64.7% male, 35.2% female
- 55% were 3rd class passengers
- Age range: 0.4 to 80 years, average 29.7 years

---

## Project Structure

```
├── Titanic.py                  # Main EDA script
├── Titanic-cleaned.csv         # Cleaned dataset
├── titanic_summary_report.txt  # Auto-generated analysis report
└── README.md                   # Project documentation
```

---

## Business Relevance

While historical, the Titanic dataset demonstrates real analytical skills:
- Identifying which variables drive outcomes
- Quantifying gaps between demographic groups
- Communicating findings clearly through structured reporting
- Laying the groundwork for predictive modeling

---
