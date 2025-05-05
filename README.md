### What to do:

- install requirements
- *run main command like so*:

    ```python -m src.sentiment_benchmarking \ --model-names cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual \```
    
    option to add, for testing: ``` --n-rows 10```. I.e.: script, modelnames (can be multiple) and testrows (optional)

- results (csv cols) in ```results``` folder

### What it does:
- takes (transformers-compatible) finetuned models to SA-score sentences of a HF dataset (default [Fiction4-dataset](https://huggingface.co/datasets/chcaa/fiction4sentiment)); format should be ["text", "label"], where label is human gold standard
- cleans up text (whitespace removal mainly)
- takes the categorical scoring (positive, [neutral,] negative) and turns it **continuous**, using the model's assigned confidence score (see ```utils.py```, conv_scores function) & saves results
- [forthcoming] computes the spearman correlation of chosen models (+precomputed vader and roberta_xlm_base) with human gold standard ("label") both on the raw and detrended gold standard (see ```utils.py``` detrend function)

