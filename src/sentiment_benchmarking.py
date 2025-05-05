# sentiment_benchmark.py

import pandas as pd
from datasets import load_dataset
from transformers import pipeline, AutoTokenizer
import typer
from loguru import logger
from pathlib import Path

from scipy.stats import spearmanr

from typing import List, Optional
from utils import get_sentiment
import re
from tqdm import tqdm

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
):
    # load dataset
    ds = load_dataset(dataset_name)
    df = pd.DataFrame(ds['train'], columns=['text', 'label', 'category', 'tr_xlm_roberta', 'vader', 'org_lang'])
    # clean text
    df['text'] = df['text'].apply(clean_whitespace)

    # TESTING PURPOSES (number of rows)
    if n_rows:
        df = df.head(n_rows)
        logger.info(f"Limiting to first {n_rows} rows for testing.")

    colnames = []

    for model_name in model_names:
        print(f"\nRunning model: {model_name}")
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
    output_path = output_dir / "sentiment_benchmark_results.csv"
    df.to_csv(output_path, index=False)
    print(f"\nSaved results to {output_path}")
    logger.info(f"Results saved to {output_path}")

    # Now we compute the spearman correlation with the models
    logger.info("Computing Spearman correlation with human labels...")
    spearmanddict = {}

    for col in colnames:
        # Compute the spearman correlation
        spearman_corr = spearmanr(df[col], df['label'])[0]
        pval = spearmanr(df[col], df['label'])[1]
        # write results to txt
        with open(output_dir / "correlation_results.txt", "a") as f:
            f.write(f"--- {col} ---\n")
            f.write(f"{spearman_corr}\n")
            f.write(f"p-value: {pval}\n")
            f.write("\n")
        # save to dict
        spearmanddict[col] = [spearman_corr, pval]
        # save to csv
        df_spearman = pd.DataFrame.from_dict(spearmanddict, orient='index', columns=['Spearman Correlation', 'p-value'])
        df_spearman.to_csv(output_dir / "spearman_results.csv", index=True)
        logger.info(f"Spearman correlation results saved to {output_dir / 'spearman_results.csv'}")

if __name__ == "__main__":
    app()