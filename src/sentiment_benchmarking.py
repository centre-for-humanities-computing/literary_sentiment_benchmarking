# sentiment_benchmark.py

import pandas as pd
from datasets import load_dataset
from transformers import pipeline, AutoTokenizer
import typer
from loguru import logger
from datetime import datetime
from pathlib import Path

from scipy.stats import spearmanr

from typing import List, Optional
from utils import get_sentiment
import re
from tqdm import tqdm

# google translate
from googletrans import Translator

app = typer.Typer()
logger.add("sentiment.log", format="{time} {message}")


def clean_whitespace(text: str) -> str:
    # rm newline characters
    text = text.replace('\n', ' ')
    # multiple spaces -> single space
    text = re.sub(r'\s+', ' ', text)
    # rm spaces before punctuation
    text = re.sub(r'\s+([.,!?;:])', r'\1', text)
    # rm excess spaces after punctuation (.,!? etc.)
    text = re.sub(r'([.,!?;:])\s+', r'\1 ', text)
    # leading and trailing spaces
    text = text.strip()
    return text


@app.command()
def main(
    model_names: List[str] = typer.Option(..., help="List of HuggingFace model names"),
    dataset_name: str = typer.Option("chcaa/fiction4sentiment", help="HF Dataset name, must contain 'text' and 'label' columns"),
    n_rows: Optional[int] = typer.Option(None, help="Limit to first N rows"),
    output_dir: Path = typer.Option("results", help="Directory where the results CSV will be saved"),
    translate: bool = typer.Option(False, help="Translate Danish sentences to English using Google Translate"),
):
    # get model names
    #model_names = model_names.split(",")
    print(f"Model names: {model_names}")
    logger.info(f"Model names: {model_names}")

    # load dataset
    ds = load_dataset(dataset_name)
    df = pd.DataFrame(ds['train'], columns=['text', 'label', 'category', 'tr_xlm_roberta', 'vader', 'org_lang'])
    # clean text
    df['text'] = df['text'].apply(clean_whitespace)

    # option to translate all sentences marked "dk" in org_lang to english w / google translate
    if translate:
        if 'org_lang' not in df.columns:
            logger.warning("org_lang column not found. Continuing without translation.")

        else:
            translator = Translator()
            # Translate only the rows where org_lang is 'dk'
            logger.info("Translating Danish ('dk') sentences to English...")
            tqdm.pandas(desc="Translating")
            df.loc[df['org_lang'] == 'dk', 'text'] = df.loc[df['org_lang'] == 'dk', 'text'].progress_apply(
                lambda x: translator.translate(x, src='da', dest='en').text)
            logger.info("Translation completed.")
        output_dir = Path("results/translated_sents")

    # TESTING PURPOSES (number of rows)
    if n_rows:
        df = df.head(n_rows)
        logger.info(f"Limiting to first {n_rows} rows for testing.")

    # save column names for later
    colnames = []

    for model_name in model_names:
        print(f"\nRunning model: {model_name.upper()}")
        model = pipeline("text-classification", model=model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        col = model_name.split("/")[-1].replace("-", "_").lower()
        colnames.append(col) # save for later
        
        # Apply sentiment analysis with tqdm for progress bar
        tqdm.pandas(desc=f"Processing {model_name}")  # Set the description for the progress bar
        df[col] = df['text'].progress_apply(lambda x: get_sentiment(x, model=model, tokenizer=tokenizer))

        print(df[col].describe())
        logger.info(f"Model {model_name} completed.")

    output_dir.mkdir(parents=True, exist_ok=True)  # create directory if it doesn't exist
    output_path = output_dir / f"sentiment_benchmark_results.csv"
    df.to_csv(output_path, index=False)
    print(f"\nSaved results to {output_path}")
    logger.info(f"Results saved to {output_path}")

    # Now we compute the spearman correlation with the models
    logger.info("Computing Spearman correlation with human labels...")

    # Dictionary to hold correlation results
    spearman_dict = {}

    for col in colnames + ['vader', 'tr_xlm_roberta']:
        corr, pval = spearmanr(df[col], df['label'])
        spearman_dict[col] = {'Spearman': corr, 'p-value': pval}

    # Save to CSV
    df_spearman = pd.DataFrame.from_dict(spearman_dict, orient='index')
    df_spearman.to_csv(output_dir / "spearman_results.csv", index_label="Model")
    logger.info(f"Spearman correlation results saved to {output_dir / 'spearman_results.csv'}")

    # Save to TXT with timestamp in filename
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    txt_output_path = output_dir / f"{timestamp}_spearman_log.txt"

    with open(txt_output_path, "w") as f:
        f.write(f"Spearman correlation results - {timestamp}\n")
        f.write("=" * 40 + "\n")
        for model, values in spearman_dict.items():
            f.write(f"{model}:\n")
            f.write(f"  Spearman: {values['Spearman']:.4f}\n")
            f.write(f"  p-value : {values['p-value']:.4g}\n\n")

    logger.info(f"Spearman results also written to {txt_output_path}")

if __name__ == "__main__":
    app()