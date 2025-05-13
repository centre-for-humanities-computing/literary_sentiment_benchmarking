# General
from loguru import logger
import numpy as np

# SA functions

# conversion of label+scores to continuous scale
# this will need to be updated sometimes (depending on the model used, it might have unreasonable names for the labels)
LABEL_NORMALIZATION = { 
    "positive": {"positive", "positiv", "pos"},
    "neutral": {"neutral", "neutr", "neut"},
    "negative": {"negative", "negativ", "neg"},}

def normalize_label(label):
    """
    Normalizes the model's sentiment label to a standard format.
    """
    label = label.lower().strip() # make sure we have a clean label
    for standard_label, variants in LABEL_NORMALIZATION.items():
        if label in variants:
            return standard_label
    raise ValueError(f"Unrecognized sentiment label: {label}")

def conv_scores(label, score):
    """
    Converts the sentiment score to a continuous scale based on (normalized) label.
    """
    sentiment = normalize_label(label)
    if sentiment == "positive":
        return score
    elif sentiment == "neutral":
        return 0
    elif sentiment == "negative":
        return -score


# Function to find the maximum allowed tokens for the model
def find_max_tokens(tokenizer):
    """
    Determines the maximum token length for the tokenizer, ensuring it doesn't exceed a reasonable limit.
    """
    max_length = tokenizer.model_max_length
    if max_length > 2000:  # sometimes, they default to ridiculously high values, so we set a max
        max_length = 512
    return max_length


# split long sentences into chunks
def split_long_sentence(text, tokenizer) -> list:
    """
    Splits long sentences into chunks if their token length exceeds the model's maximum length.
    """
    words = text.split()
    parts = []
    current_part = []
    current_length = 0

    max_length = find_max_tokens(tokenizer)

    for word in words:
        # Encode word and get the token length
        tokens = tokenizer.encode(word)
        seq_len = len(tokens)

        # Check if adding this word would exceed max length
        if current_length + seq_len > max_length:
            parts.append(" ".join(current_part))  # Append the current part as a chunk
            current_part = [word]  # Start a new part with the current word
            current_length = seq_len  # Reset the current length to the length of the current word
        else:
            current_part.append(word)  # Add the word to the current chunk
            current_length += seq_len  # Update the current length

    # Append any remaining part as a chunk
    if current_part:
        parts.append(" ".join(current_part))

    return parts


# get SA scores from xlm-roberta
def get_sentiment(text, pipe, tokenizer, model_name):
    """
    Gets the sentiment score for a given text, including splitting long sentences into chunks if needed.
    """
    
    #spec_labs = get_model_labels(pipe)  # labels for the model

    # Check that the text is a string
    if not isinstance(text, str):
        print(f"Warning: Text is not a string for text: '{text}'. Skipping.")
        return None

    # Split the sentence into chunks if it's too long
    chunks = split_long_sentence(text, tokenizer)

    if not chunks:
        print(f"Warning: No chunks created for text: '{text}'. Skipping.")
        return None

    # If there is only one chunk, we can directly use the original text
    if len(chunks) == 1:
        chunks = [text]  # Just use the original text

    # If the sentence is split into multiple chunks, print a warning
    elif len(chunks) > 1:
        print(f"Warning: Sentence split into {len(chunks)} chunks for text: '{text}'.")
        logger.info(f"Sentence split into {len(chunks)} chunks for text: '{text}'.")

    # Loop through the chunks and get sentiment scores for each
    sentiment_scores = []

    for chunk in chunks:
        # Get sentiment from the model
        sent = pipe(chunk)
        model_label = sent[0].get("label")
        model_score = sent[0].get("score")

        # Transform score to continuous scale
        converted_score = float(conv_scores(model_label, model_score))
        sentiment_scores.append(converted_score)

    # Calculate the mean sentiment score from the chunks
    mean_score = sum(sentiment_scores) / len(sentiment_scores)

    return mean_score