import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load fully processed data
df = pd.read_csv('analysis_data_fully_processed.csv')

# --- Graph 1: Boxplot of Engagement by Category ---
plt.figure(figsize=(8,6))
sns.boxplot(x='category', y='engagement', data=df, palette=['#6BAED6', '#FD8D3C'])
plt.title('Engagement by Category')
plt.ylabel('Engagement Score')
plt.xlabel('Category')
plt.savefig('graph1_engagement_boxplot.png', dpi=300)
plt.close()

# --- Graph 2: Scatterplot Engagement vs Polarity ---
plt.figure(figsize=(8,6))
sns.scatterplot(x='polarity', y='engagement', hue='category', data=df, palette=['#6BAED6', '#FD8D3C'])
plt.title('Engagement vs Polarity')
plt.ylabel('Engagement Score')
plt.xlabel('Polarity')
plt.savefig('graph2_engagement_polarity.png', dpi=300)
plt.close()

# --- Graph 3: Scatterplot Engagement vs Flesch-Kincaid Grade ---
plt.figure(figsize=(8,6))
sns.scatterplot(x='flesch_kincaid_grade', y='engagement', hue='category', data=df, palette=['#6BAED6', '#FD8D3C'])
plt.title('Engagement vs Flesch-Kincaid Grade')
plt.ylabel('Engagement Score')
plt.xlabel('Flesch-Kincaid Grade')
plt.savefig('graph3_engagement_readability.png', dpi=300)
plt.close()

# --- Graph 4 & 5: Top 10 Keywords Political & Entertainment ---
# Load ML report
with open('results_C_ml_report.txt', 'r') as f:
    lines = f.readlines()

# Function to parse top keywords section
def parse_top_keywords(lines, section_name):
    start_idx = None
    for i, line in enumerate(lines):
        if section_name in line:
            start_idx = i + 1
            break
    keywords = {}
    if start_idx:
        for line in lines[start_idx:start_idx+10]:
            parts = line.strip().split()
            if len(parts) == 2:
                word, score = parts
                try:
                    keywords[word] = float(score)
                except:
                    continue
    return keywords

political_keywords = parse_top_keywords(lines, 'Top 10 Keywords that predict the Political Category')
entertainment_keywords = parse_top_keywords(lines, 'Top 10 Keywords that predict the Entertainment Category')

# --- Graph 4: Top Political Keywords ---
plt.figure(figsize=(8,6))
sns.barplot(x=list(political_keywords.values()), y=list(political_keywords.keys()), color='#FD8D3C')
plt.title('Top 10 Keywords Predicting Political Category')
plt.xlabel('Coefficient Value')
plt.ylabel('Keyword')
plt.tight_layout()
plt.savefig('graph4_top_political_keywords.png', dpi=300)
plt.close()

# --- Graph 5: Top Entertainment Keywords ---
plt.figure(figsize=(8,6))
sns.barplot(x=list(entertainment_keywords.values()), y=list(entertainment_keywords.keys()), color='#6BAED6')
plt.title('Top 10 Keywords Predicting Entertainment Category')
plt.xlabel('Coefficient Value')
plt.ylabel('Keyword')
plt.tight_layout()
plt.savefig('graph5_top_entertainment_keywords.png', dpi=300)
plt.close()

print("All 5 graphs generated and saved successfully.")
