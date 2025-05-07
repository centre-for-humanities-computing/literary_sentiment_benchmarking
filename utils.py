# General
from loguru import logger

# SA functions

# to convert transformer scores to the same scale as the dictionary-based scores
def conv_scores(label, score, spec_lab):  # single label and score
    """
    Converts transformer-based sentiment scores to a uniform scale based on specified labels.
    We need to lowercase since sometimes, a model will have as label "Neutral" or "neutral" or "NEUTRAL"
    """
    if len(spec_lab) == 2:
        if label.lower() == spec_lab[0]:  # "positive"
            return score
        elif label.lower() == spec_lab[1]:  # "negative"
            return -score  # return negative score

    elif len(spec_lab) == 3:
        if label.lower() == spec_lab[0]:  # "positive"
            return score
        elif label.lower() == spec_lab[1]:  # "neutral"
            return 0  # return 0 for neutral
        elif label.lower() == spec_lab[2]:  # "negative"
            return -score  # return negative score

    else:
        raise ValueError("spec_lab must contain either 2 or 3 labels.")


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
def get_sentiment(text, model, tokenizer, model_name):
    """
    Gets the sentiment score for a given text, including splitting long sentences into chunks if needed.
    """
    if model_name == "vesteinn/danish_sentiment":
        spec_labs = ["positiv", "neutral", "negativ"]  # labels for the model
    else:
        spec_labs = ["positive", "neutral", "negative"]  # labels for the model

    # Check that the text is a string
    if not isinstance(text, str):
        print(f"Warning: Text is not a string for text: '{text}'. Skipping.")
        return None

    # Split the sentence into chunks if it's too long
    chunks = split_long_sentence(text, tokenizer)

    if len(chunks) == 0:
        print(f"Warning: No chunks created for text: '{text}'. Skipping.")
        return None

    elif len(chunks) == 1:
        # If the sentence is short enough, just use it as is
        chunks = [text]

    else:
        # If the sentence is split into chunks, print a warning
        print(f"Warning: Sentence split into {len(chunks)} chunks for text: '{text}'.")
        logger.info(f"Sentence split into {len(chunks)} chunks for text: '{text}'.")

    # Loop through the chunks and get sentiment scores for each
    sentiment_scores = []

    for chunk in chunks:
        # Get sentiment from the model
        sent = model(chunk)
        xlm_label = sent[0].get("label")
        xlm_score = sent[0].get("score")

        # Transform score to continuous scale
        xlm_converted_score = conv_scores(xlm_label, xlm_score, spec_lab=spec_labs)
        # make sure the score is a float
        xlm_converted_score = float(xlm_converted_score)
        sentiment_scores.append(xlm_converted_score)

    # Calculate the mean sentiment score from the chunks
    mean_score = sum(sentiment_scores) / len(sentiment_scores)

    return mean_score