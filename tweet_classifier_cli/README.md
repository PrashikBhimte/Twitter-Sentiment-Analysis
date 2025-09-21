# Tweet Classifier CLI Tool

This is a command-line interface (CLI) tool to classify tweets using pre-trained sentiment analysis models.

## Installation

1.  Navigate to the `tweet_classifier_cli` directory:

    ```bash
    cd tweet_classifier_cli
    ```

2.  Install the package in editable mode (for development) or as a regular package:

    ```bash
    pip install .
    # or for development
    # pip install -e .
    ```

    This will install all necessary dependencies listed in `requirements.txt` and make the `classify-tweet` command available in your terminal.

## Usage

To classify a tweet, use the `classify-tweet` command followed by the tweet text. You can optionally specify which model to use.

### Basic Usage (using default model `bert_logistic_reg_model`)

```bash
classify-tweet "I love this new product!"
```

### Specifying a Model

You can choose between `bert_logistic_reg_model` and `logistic_reg_model` using the `--model` or `-m` option.

```bash
classify-tweet "This is a terrible movie." --model logistic_reg_model
# or
classify-tweet "What a fantastic day!" -m bert_logistic_reg_model
```

### Example Output

```
Attempting to classify tweet: "I love this new product!" using model: bert_logistic_reg_model
Looking for model at: /path/to/your/project/trained_models/bert_logistic_reg_model.joblib
Looking for vectorizer at: /path/to/your/project/trained_models/tfidf_vectorizer.joblib
Prediction for model 'bert_logistic_reg_model': POSITIVE
```
