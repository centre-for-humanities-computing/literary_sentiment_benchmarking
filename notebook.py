# %%

import pandas as pd
from datasets import load_dataset
from transformers import pipeline, AutoTokenizer

from utils import get_sentiment

# %%
# load the annotated dataset
ds = load_dataset("chcaa/fiction4sentiment")
# make df
df = pd.DataFrame(ds['train'], columns=['text', 'label', 'category', 'tr_xlm_roberta', 'vader'])
# make "org_lang" column, if prose/poetry, then "en", else "dk"
df['org_lang'] = df['category'].apply(lambda x: 'en' if x in ['prose', 'poetry'] else 'dk')
df.head()
# %%

models = {"roberta_base_multilingual": "cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual",
          "mimememo_sentiment_bert": "MiMe-MeMo/MeMo-BERT-SA",
          }


# %%
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
