# sentiment_benchmark.py

import pandas as pd
import numpy as np
from datasets import load_dataset
from transformers import pipeline, AutoTokenizer
import typer
from loguru import logger
from datetime import datetime
import time
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


def safe_translate(text, retries=3, delay=5):
    """ Translate the text with retries in case of errors like timeouts. """
    translator = Translator()
    for _ in range(retries):
        try:
            return translator.translate(text, src='da', dest='en').text
        except Exception as e:
            tqdm.write(f"Translation error: {e}. Retrying...")
            time.sleep(delay)  # Wait for some time before retrying
    return text  # return original text after all retries


def translate_danish_to_english_in_batches(df: pd.DataFrame, text_col: str = "text", lang_col: str = "org_lang", batch_size: int = 100) -> pd.DataFrame:
    """
    Translates Danish text (where lang_col == 'dk') in the specified text column to English in batches.
    """
    if lang_col not in df.columns:
        tqdm.write(f"Warning: '{lang_col}' column not found. Skipping translation.")
        return df

    tqdm.write("Translating Danish ('dk') sentences to English in batches...")
    translator = Translator()
    tqdm.pandas(desc="Translating")
        
    # get rows where the language == Danish
    is_danish = df[lang_col] == 'dk'
    danish_texts = df.loc[is_danish, text_col]

    # Split texts into batches
    batches = np.array_split(danish_texts, len(danish_texts) // batch_size + 1)
    
    translated_texts = []
    for batch in tqdm(batches, desc="Processing batches"):
        # translate each batch
        translated_batch = batch.progress_apply(safe_translate)
        translated_texts.extend(translated_batch.tolist())

    # Reassign translated texts back to the DataFrame
    df.loc[is_danish, text_col] = translated_texts
    tqdm.write("Translation completed.")
    
    return df


@app.command()
def main(
    model_names: List[str] = typer.Option(..., help="List of HuggingFace model names"),
    dataset_name: str = typer.Option("chcaa/fiction4sentiment", help="HF Dataset name, must contain 'text' and 'label' columns"),
    n_rows: Optional[int] = typer.Option(None, help="Limit to first N rows"),
    output_dir: Path = typer.Option("results", help="Directory where the results CSV will be saved"),
    translate: bool = typer.Option(False, help="Translate Danish sentences to English using Google Translate"),
):
    # get model names
    print(f"Model names: {model_names}")
    logger.info(f"Model names: {model_names}")

    # load dataset
    ds = load_dataset(dataset_name)
    df = pd.DataFrame(ds['train'], columns=['text', 'label', 'category', 'tr_xlm_roberta', 'vader', 'org_lang'])

    # TESTING PURPOSES (number of rows)
    if n_rows:
        df = df.head(n_rows)
        logger.info(f"Limiting to first {n_rows} rows for testing.")

    # clean text
    df['text'] = df['text'].apply(clean_whitespace)

    # option to translate all sentences marked "dk" in org_lang to english w / google translate
    if translate:
        # check if the saved translations already exist
        if Path(f"results/translated_sents/{dataset_name.split('/')[-1]}_saved_translations.csv").exists():
            # load them
            df = pd.read_csv(f"results/translated_sents/{dataset_name.split('/')[-1]}_saved_translations.csv")
            logger.info("Loaded saved translations.")
        else:
            # translate the sentences
            df = translate_danish_to_english_in_batches(df, text_col='text', lang_col='org_lang')
            # save them
            df.to_csv(f"results/translated_sents/{dataset_name.split('/')[-1]}_saved_translations.csv", index=False)
            logger.info("Saved translations to CSV.")
        # change output dir to "translated" if using translation
        output_dir = Path("results/translated_sents")

    # save column names for later
    colnames = []

    for model_name in model_names:
        #print(f"\nRunning model: {model_name.upper()}")
        tqdm.write(f"\nRunning model: {model_name.upper()}")
        model = pipeline("text-classification", model=model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        col = model_name.split("/")[-1].replace("-", "_").lower()
        try: 
            # Apply sentiment analysis with tqdm for progress bar
            tqdm.pandas(desc=f"Processing {model_name}")  # Set the description for the progress bar
            df[col] = df['text'].progress_apply(lambda x: get_sentiment(x, model=model, tokenizer=tokenizer))
            logger.info(f"Model {model_name} completed.")
            colnames.append(col) # save for later
        except Exception as e:
            tqdm.write(f"Error processing model {model_name}: {e}")
            continue

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