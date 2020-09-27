import os
import joblib
import warnings

from pyfiglet import Figlet

import click

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer

#import feature_extraction as fe
CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'],
                        version_option_names=['-V', '--version'])

click.echo(Figlet(font='larry3d').renderText('SysFake'))

#@click.option('--tfidf_kwargs',
#              default={},
#              #short_help='Arguments for `sklearn.feature_extraction.text.TfidfVectorizer`.',
#              help="Arguments in the form of a `dict`, \
#                    to be passed to the `TfidfVectorizer`. \
#                    Passing these arguments lets you configure \
#                    pre-processing and define whether you want \
#                    the transformer to make n-grams., \
#                    among other options.")
def tfidf_transform(single_text, **tfidf_kwargs):
    return TfidfVectorizer(**tfidf_kwargs).fit_transform(TRAIN_TEXTS + [single_text])

@click.command(no_args_is_help=True)
@click.option('--single-text', '-s', default=None,
              help="""Classify a single string enclosed in double-quotes, provided in the command line. The result will be returned in the command line.\n
                    Example:\n
                    python sfake.py --single-text \"Lorem ipsum dolor sit amet, consectetur adipiscing elit...\"""")
@click.option('--model', '-m', default='sgd-taxonomy',
              help="Filename of model to use for classification, in the `models` directory.")
@click.option('--rep', '-r', default='taxonomy', type=click.Choice(('bert', 'tfidf', 'taxonomy')), show_choices=True,
              help="Data represenation you wish to use.")
@click.version_option(prog_name='SysFake CLI', version='0.0.1')
def predict(model, rep, single_text):
    """
    Testing whether function docstring shows up in help.
    """
    try:
        TRAIN_TEXTS = pd.read_csv('data/texts_labelled.csv')
        click.echo("Training texts loaded.")
    except:
        # download from hosted source
        pass

    if not rep in model:
        raise click.BadParameter("Model filename does not contain the name of the data representation. \
                                  Verify that the model you are using was trained on the chosen representation \
                                  and add that representation to its filename to suppress this warning.",
                                 param_hint=["--rep", "--model"])

    try:
        with open(f'models/{model}.pickle', mode='rb') as filein:
            model_obj = joblib.load(filein)
        click.echo("Model loaded...")
    except:
        raise FileNotFoundError("Model file not found in the `models` directory")

    if single_text:
        pass

if __name__ == '__main__':
    predict()
