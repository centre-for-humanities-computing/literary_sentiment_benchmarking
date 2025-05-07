# %%

import pandas as pd
from datasets import load_dataset
from transformers import pipeline, AutoTokenizer
from loguru import logger

# set to current directory
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
#check dir
print(os.getcwd())
# set dir to parent directory
os.chdir("..")
# check dir
print(os.getcwd())

from utils import get_sentiment

# %%

# TEST SENTIMENT MODELS OUT

models = {"roberta_base_multilingual": "cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual",
          "roberta_base_xlm": "cardiffnlp/xlm-roberta-base-sentiment-multilingual",
          "mimememo_sentiment": "MiMe-MeMo/MeMo-BERT-SA",
          }

# test getting sentiment with memo
model = models["roberta_base_xlm"]
tokenizer = AutoTokenizer.from_pretrained(model)
sentiment = pipeline("text-classification", model=model)

text = "Min moster hader mig."
print(get_sentiment(text, sentiment, tokenizer))


# %%

# CHECK OUT THE SCORES (RESULTS)

df = pd.read_csv("results/sentiment_benchmark_results.csv")
df
# %%
for i, row in df.iterrows():
    print(row['text'])
    print(row['tr_xlm_roberta'], "old roberta")
    print(row['label'], "human")
    print(row['twitter_xlm_roberta_base_sentiment_multilingual'], "roberta multilingual")
    print()


# %%

# IRR

# get interrater reliability
from scipy.stats import spearmanr

# load the annotated dataset
ds = load_dataset("chcaa/fiction4sentiment")
# make df
df = pd.DataFrame(ds['train'])
# make "org_lang" column, if prose/poetry, then "en", else "dk"
df['org_lang'] = df['category'].apply(lambda x: 'en' if x in ['prose', 'poetry'] else 'dk')
df.head()

# get the columns to compare
cols = ['annotator_1', 'annotator_2', 'annotator_3']

mean_spearman = []

for col in cols:
    # compare each column to the other and get mean spearmanr
    for col2 in cols:
        if col != col2:
            # drop nans
            dt = df.dropna(subset=[col, col2])
            # get the spearmanr
            spearman = spearmanr(dt[col], dt[col2])
            print(f"Spearman correlation between {col} and {col2}: {spearman.correlation}")
            print(f"p-value: {spearman.pvalue}")
            mean_spearman.append(spearman.correlation)
# get the mean spearman
mean_spearman = sum(mean_spearman) / len(mean_spearman)
print(f"Mean spearman correlation: {mean_spearman}")


# %%

# TRY OUT GOOGLE TRANSLATE
#%pip install googletrans==4.0.0-rc1
from googletrans import Translator

df_dk = df[df['org_lang'] == 'dk']
# get the first 10 rows
df_dk = df_dk.head(10)

# make a translator object
translator = Translator()
# translate the text

for text in df_dk['text']:
    # translate the text
    translated = translator.translate(text, src='da', dest='en')
    # print the translated text
    print(translated.text)
# %%
df_dk
# %%
