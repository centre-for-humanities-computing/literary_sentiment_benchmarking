### What to do:

- install requirements
- *run main command like so*:

    python -m src.sentiment_benchmarking \
    --dataset-name chcaa/fiction4sentiment \
    --model-names cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual \
    --model-names cardiffnlp/xlm-roberta-base-sentiment-multilingual \

    (english)


    (danish)
    --model-names MiMe-MeMo/MeMo-BERT-SA \
    --model-names alexandrainst/da-sentiment-base \
    --model-names vesteinn/danish_sentiment

    option to add, for testing a number of rows to run, e.g.: ```--n-rows 10``` 

    option to add, for google-translating, ```--translate``` (NB: will be slow going)
    
    > I.e.: script, model_names (can be multiple) and n-rows (optional) and translate (optional)

- results (csv cols data & spearman results) in ```results``` folder

### What it does:
- takes (transformers-compatible) finetuned models to SA-score sentences of a HF dataset (default [Fiction4-dataset](https://huggingface.co/datasets/chcaa/fiction4sentiment)); format should be ["text", "label"], where label is human gold standard
- cleans up text (whitespace removal mainly)
- takes the categorical scoring (positive, [neutral,] negative) and turns it **continuous**, using the model's assigned confidence score (see ```utils.py```, conv_scores function) & saves results
- computes the spearman correlation of chosen models (+precomputed* vader and roberta_xlm_base) with human gold standard ("label") both on the raw and -- [forthcoming] detrended gold standard (see ```utils.py```)


*Note that the precomputed vader & [roberta_xlm_base](https://huggingface.co/cardiffnlp/twitter-xlm-roberta-base-sentiment) were applied to Google-translated sentences. For details, see [our paper here](https://aclanthology.org/2024.wassa-1.15.pdf) -- these are the baselines to beat.


### What it could do [forthcoming]
- Compare performance on continuous-scale converted scores to binary classification performance (using, e.g., the memo dataset: [https://huggingface.co/datasets/MiMe-MeMo/MeMo-Dataset-SA](https://huggingface.co/datasets/MiMe-MeMo/MeMo-Dataset-SA))
