import pandas as pd
import numpy as np
import os
from textblob import TextBlob
from textstat import textstat



all_files = [
    r'C:\Users\jasra\OneDrive\Desktop\Research Paper\Data\MensXP.csv',
    r'C:\Users\jasra\OneDrive\Desktop\Research Paper\Data\Netflix.csv',
    r'C:\Users\jasra\OneDrive\Desktop\Research Paper\Data\PrintMedia(images).csv',
    r'C:\Users\jasra\OneDrive\Desktop\Research Paper\Data\Quint(images).csv',
    r'C:\Users\jasra\OneDrive\Desktop\Research Paper\Data\ScoopWhoop.csv',
    r'C:\Users\jasra\OneDrive\Desktop\Research Paper\Data\TheHumansOfBombay.csv',
    r'C:\Users\jasra\OneDrive\Desktop\Research Paper\Data\TheMojoStory.csv',
    r'C:\Users\jasra\OneDrive\Desktop\Research Paper\Data\TheViralFever.csv',
    r'C:\Users\jasra\OneDrive\Desktop\Research Paper\Data\YouthKiAwaaz(images).csv',
    r'C:\Users\jasra\OneDrive\Desktop\Research Paper\Data\FilterCopy.csv',
    r'C:\Users\jasra\OneDrive\Desktop\Research Paper\Data\LaughingColors.csv'
    r'C:\Users\jasra\OneDrive\Desktop\Research Paper\Data\thewirein.csv'
    r'C:\Users\jasra\OneDrive\Desktop\Research Paper\Data\opindia_com.csv'
    r'C:\Users\jasra\OneDrive\Desktop\Research Paper\Data\missmalini.csv'
    r'C:\Users\jasra\OneDrive\Desktop\Research Paper\Data\bollywoodbubble.csv'
]

category_map = {
    'thewirein.csv': 'Political',
    'opindia_com.csv': 'Political',
    'missmalini.csv': 'Entertainment',
    'bollywoodbubble.csv': 'Entertainment',
    'PrintMedia(images).csv': 'Political',
    'Quint(images).csv': 'Political',
    'TheMojoStory.csv': 'Political',
    'YouthKiAwaaz(images).csv': 'Political',
    'Netflix.csv': 'Entertainment',
    'TheViralFever.csv': 'Entertainment',
    'ScoopWhoop.csv': 'Entertainment',
    'FilterCopy.csv': 'Entertainment',
    'LaughingColors.csv': 'Entertainment',
    'MensXP.csv': 'Entertainment',
    'TheHumansOfBombay.csv': 'Entertainment'
    ''
}


TARGET_COLS = ['account_name', 'category', 'caption',
               'likes_count', 'comments_count', 'video_view_count']

li = []
for full_path in all_files:
    base_filename = os.path.basename(full_path)

    try:
        df = pd.read_csv(full_path, low_memory=False)


        account_name = base_filename.split('.')[0]
        df['account_name'] = account_name
        df['category'] = category_map.get(base_filename, 'Unknown')


        df = df.rename(columns={
            'caption': 'caption',
            'commentsCount': 'comments_count',
            'likesCount': 'likes_count',
            'videoViewCount': 'video_view_count'

        })


        df = df.reindex(columns=TARGET_COLS)

        li.append(df)

    except Exception as e:
        print(f"Skipping {base_filename} due to error: {e}")

df_master = pd.concat(li, ignore_index=True)


cols_to_convert = ['likes_count', 'comments_count', 'video_view_count']

for col in cols_to_convert:

    df_master[col] = pd.to_numeric(df_master[col], errors='coerce').fillna(0).astype(int)

df_master['caption'] = df_master['caption'].fillna('')


df_master['engagement'] = (
    df_master['likes_count'] +
    (2 * df_master['comments_count']) +
    (df_master['video_view_count'] / 100)
)

print("\n--- Data Consolidation Complete ---")
print(df_master.info())
print("\n--- Sample Data ---")
print(df_master.head())


df_master.to_csv('consolidated_data.csv', index=False)
print("\nconsolidation")
print(df_master.info())
print("\nsaved")
