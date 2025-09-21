import click
import os
from .predictor import TfidfTweetClassifier, BertTweetClassifier

@click.command()
@click.argument('tweet_text')
@click.option('--model', '-m', 
              type=click.Choice(['bert_logistic_reg_model', 'logistic_reg_model'], case_sensitive=False),
              default='bert_logistic_reg_model',
              help='Specify which model to use for classification.')
def cli(tweet_text, model):
    """Classify a tweet using pre-trained sentiment analysis models."""
    
    cli_base_dir = os.path.dirname(os.path.abspath(__file__))
    trained_models_dir = os.path.join(cli_base_dir, '..', '..', '..', 'trained_models')
    
    click.echo(f"Attempting to classify tweet: \"{tweet_text}\" using model: {model}")

    try:
        if model == 'logistic_reg_model':
            # Swapped filenames from the notebook
            model_path = os.path.join(trained_models_dir, 'tfidf_vectorizer.joblib')
            vectorizer_path = os.path.join(trained_models_dir, 'logistic_reg_model.joblib')
            click.echo(f"Looking for model at: {model_path}")
            click.echo(f"Looking for vectorizer at: {vectorizer_path}")
            classifier = TfidfTweetClassifier(model_path, vectorizer_path)
        elif model == 'bert_logistic_reg_model':
            model_path = os.path.join(trained_models_dir, 'bert_logistic_reg_model.joblib')
            click.echo(f"Looking for model at: {model_path}")
            classifier = BertTweetClassifier(model_path)
        else:
            raise click.UsageError("Invalid model specified.")

        prediction = classifier.predict(tweet_text)
        click.echo(f"Prediction for model '{model}': {prediction}")

    except FileNotFoundError as e:
        click.echo(f"Error: {e}", err=True)
        click.echo("Please ensure the trained models and vectorizer are in the 'trained_models/' directory.", err=True)
    except Exception as e:
        click.echo(f"An unexpected error occurred: {e}", err=True)

if __name__ == '__main__':
    cli()
