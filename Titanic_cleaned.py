import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import datetime
from tabulate import tabulate

# Redirect output to file
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = f"C:/Users/ABC/Documents/Titanic project/titanic_report_{timestamp}.txt"
original_stdout = sys.stdout
sys.stdout = open(output_file, 'w', encoding='utf-8')


def main():
    df = pd.read_csv('C:/Users/ABC/Documents/Titanic project/titanic-cleaned.csv')
    print("Dataset loaded successfully!")
    print(f"Total passengers: {len(df)}")
    print(f"Columns available: {df.columns.tolist()}")

    pclass_col = 'Pclass' if 'Pclass' in df.columns else 'P class' if 'P class' in df.columns else None
    gender_col = 'Sex' if 'Sex' in df.columns else 'Gender' if 'Gender' in df.columns else None
    survived_col = 'Survived'

    print(f"Using '{pclass_col}' for passenger class")
    print(f"Using '{gender_col}' for gender")
    print(f"Overall survival rate: {df[survived_col].mean() * 100:.1f}%")

    print("\n=== KEY STATISTICS ===")
    if gender_col:
        gender_survival = df.groupby(gender_col)[survived_col].mean() * 100
        for gender, rate in gender_survival.items():
            print(f"{gender} survival rate: {rate:.1f}%")

    if pclass_col:
        class_survival = df.groupby(pclass_col)[survived_col].mean() * 100
        for pclass, rate in class_survival.items():
            print(f"Class {pclass} survival rate: {rate:.1f}%")

    available_numeric = []
    for col in ['Survived', 'Pclass', 'P class', 'Age', 'Fare', 'SibSp', 'Parch']:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            available_numeric.append(col)

    if 'Survived' in df.columns and len(available_numeric) > 1:
        numeric_df = df[available_numeric].corr()
        print("\nCORRELATIONS WITH SURVIVAL:")
        survival_corr = numeric_df['Survived'].sort_values(key=abs, ascending=False)
        for feature, corr in survival_corr.items():
            if feature != 'Survived':
                print(f"{feature}: {corr:+.3f}")


if __name__ == "__main__":
    main()
    sys.stdout.close()
    sys.stdout = original_stdout
    print(f"Report saved to: {output_file}")


def generate_titanic_summary_report(df, output_file=None):
    pclass_col = 'Pclass' if 'Pclass' in df.columns else 'P class'
    gender_col = 'Sex' if 'Sex' in df.columns else 'Gender'
    survived_col = 'Survived'

    report = []
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    report.append("TITANIC DATASET ANALYSIS REPORT")
    report.append("=" * 50)
    report.append(f"Report Generated: {timestamp}")
    report.append(f"Total Passengers: {len(df):,}")
    report.append(f"Overall Survival Rate: {df[survived_col].mean() * 100:.1f}%")
    report.append("")

    report.append("1. DATASET OVERVIEW")
    report.append("-" * 30)
    report.append(f"• Dimensions: {df.shape[0]} rows × {df.shape[1]} columns")
    report.append(f"• Memory Usage: {df.memory_usage(deep=True).sum() / 1024:.1f} KB")
    report.append("")

    report.append("2. DATA QUALITY SUMMARY")
    report.append("-" * 30)
    missing_data = df.isnull().sum()
    missing_percent = (missing_data / len(df)) * 100
    for col, count, percent in zip(df.columns, missing_data, missing_percent):
        if count > 0:
            report.append(f"• {col}: {count} missing values ({percent:.1f}%)")
    if missing_data.sum() == 0:
        report.append("• No missing values detected")
    report.append("")

    report.append("3. PASSENGER DEMOGRAPHICS")
    report.append("-" * 30)
    if 'Age' in df.columns:
        age_stats = df['Age'].describe()
        report.append(f"• Age Range: {age_stats['min']:.1f} - {age_stats['max']:.1f} years")
        report.append(f"• Average Age: {age_stats['mean']:.1f} years")
        report.append(f"• Median Age: {age_stats['50%']:.1f} years")
    if gender_col in df.columns:
        gender_counts = df[gender_col].value_counts()
        for gender, count in gender_counts.items():
            percentage = (count / len(df)) * 100
            report.append(f"• {gender}: {count} passengers ({percentage:.1f}%)")
    if pclass_col in df.columns:
        class_counts = df[pclass_col].value_counts().sort_index()
        for pclass, count in class_counts.items():
            percentage = (count / len(df)) * 100
            report.append(f"• Class {pclass}: {count} passengers ({percentage:.1f}%)")
    report.append("")

    report.append("4. SURVIVAL ANALYSIS SUMMARY")
    report.append("-" * 30)
    report.append(f"Overall Survival Rate: {df[survived_col].mean() * 100:.1f}%")
    report.append(f"Survived: {df[survived_col].sum():,} passengers")
    report.append(f"Did Not Survive: {len(df) - df[survived_col].sum():,} passengers")
    report.append("")

    report.append("5. SURVIVAL RATES BY DEMOGRAPHICS")
    report.append("-" * 30)
    if gender_col in df.columns:
        gender_survival = df.groupby(gender_col)[survived_col].agg(['mean', 'count'])
        gender_survival['mean'] = gender_survival['mean'] * 100
        report.append("BY GENDER:")
        for gender, row in gender_survival.iterrows():
            report.append(f"  • {gender}: {row['mean']:.1f}% ({int(row['count'])} passengers)")
        report.append("")
    if pclass_col in df.columns:
        class_survival = df.groupby(pclass_col)[survived_col].agg(['mean', 'count'])
        class_survival['mean'] = class_survival['mean'] * 100
        report.append("BY PASSENGER CLASS:")
        for pclass, row in class_survival.iterrows():
            report.append(f"  • Class {pclass}: {row['mean']:.1f}% ({int(row['count'])} passengers)")
        report.append("")

    report.append("6. KEY STATISTICAL FINDINGS")
    report.append("-" * 30)
    if gender_col in df.columns:
        gender_rates = df.groupby(gender_col)[survived_col].mean() * 100
        if len(gender_rates) == 2:
            diff = abs(gender_rates.iloc[0] - gender_rates.iloc[1])
            report.append(f"• Gender Survival Gap: {diff:.1f}% difference")
    if pclass_col in df.columns:
        class_rates = df.groupby(pclass_col)[survived_col].mean() * 100
        if len(class_rates) >= 2:
            max_diff = abs(class_rates.max() - class_rates.min())
            report.append(f"• Class Survival Gap: {max_diff:.1f}% between highest and lowest class")

    report.append("")
    report.append("7. CORRELATION INSIGHTS")
    report.append("-" * 30)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 1 and survived_col in numeric_cols:
        correlations = df[numeric_cols].corr()[survived_col].sort_values(key=abs, ascending=False)
        report.append("Correlation with Survival (absolute values):")
        for feature, corr in correlations.items():
            if feature != survived_col:
                strength = "Strong" if abs(corr) > 0.3 else "Moderate" if abs(corr) > 0.1 else "Weak"
                direction = "positive" if corr > 0 else "negative"
                report.append(f"  • {feature}: {corr:+.3f} ({strength} {direction} correlation)")

    report.append("")
    report.append("8. EXECUTIVE SUMMARY")
    report.append("-" * 30)
    if gender_col in df.columns and pclass_col in df.columns:
        highest_survival = df.groupby([gender_col, pclass_col])[survived_col].mean().idxmax()
        lowest_survival = df.groupby([gender_col, pclass_col])[survived_col].mean().idxmin()
        highest_rate = df.groupby([gender_col, pclass_col])[survived_col].mean().max() * 100
        lowest_rate = df.groupby([gender_col, pclass_col])[survived_col].mean().min() * 100
        report.append(f"• Highest survival: {highest_survival[0]} in Class {highest_survival[1]} ({highest_rate:.1f}%)")
        report.append(f"• Lowest survival: {lowest_survival[0]} in Class {lowest_survival[1]} ({lowest_rate:.1f}%)")
        report.append(f"• Survival range: {lowest_rate:.1f}% to {highest_rate:.1f}%")

    report.append("")
    report.append("END OF REPORT")
    report.append("=" * 50)

    report_text = "\n".join(report)
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report_text)
    return report_text


def main():
    df = pd.read_csv('C:/Users/ABC/Documents/Titanic project/titanic-cleaned.csv')
    report = generate_titanic_summary_report(
        df,
        output_file='C:/Users/ABC/Documents/Titanic project/titanic_summary_report.txt'
    )
    print(report)


if __name__ == "__main__":
    main()
