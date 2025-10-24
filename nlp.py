import pandas as pd
from textblob import TextBlob
from textstat import textstat
import numpy as np

df_master = pd.read_csv('consolidated_data.csv')
def get_sentiment(text):
    text = str(text) if pd.notna(text) else ""
    if not text:
        return 0.0, 0.0
    try:
        blob = TextBlob(text)
        return blob.sentiment.polarity, blob.sentiment.subjectivity
    except:
        return 0.0, 0.0

df_master[['polarity', 'subjectivity']] = df_master['caption'].apply(
    lambda x: pd.Series(get_sentiment(x))
)
def get_readability(text):
    text = str(text) if pd.notna(text) else ""
    if not text or len(text.split()) < 2:
        return 0.0
    try:
        return textstat.flesch_kincaid_grade(text)
    except:
        return 0.0

df_master['flesch_kincaid_grade'] = df_master['caption'].apply(get_readability)

print("\nnlp done")
print(df_master.info())

df_master.to_csv('analysis_data_fully_processed.csv', index=False)
print("\nsaved")
