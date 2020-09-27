import click

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer

#import feature_extraction as fe

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])

#DATA_TAXONOMY = pd.read_csv('data/d_full.csv')

try:
    with open('data/texts_labelled.pickle', mode='rb') as filein:
        TRAIN_TEXTS = pickle.load(filein)
except:
    # download from hosted source
    pass

@click.option('--tfidf_kwargs',
              default={},
              #short_help='Arguments for `sklearn.feature_extraction.text.TfidfVectorizer`.',
              help="Arguments in the form of a `dict`, \
                    to be passed to the `TfidfVectorizer`. \
                    Passing these arguments lets you configure \
                    pre-processing and define whether you want \
                    the transformer to make n-grams., \
                    among other options.")
def tfidf_transform(single_text, **tfidf_kwargs):
    return TfidfVectorizer(**tfidf_kwargs).fit_transform(TRAIN_TEXTS + [single_text])

@click.command('predict',
               context_settings=CONTEXT_SETTINGS,
               short_help='Classify a new sample or set of samples with using a given model.',
               help="")
#@click.version_option(prog_name="SysFake", version='0.0.1')
@click.option('--model',
              default='sgd-taxonomy',
              #short_help='Choice of model to use in prediction from those in the `models` directory.',
              help="The filename of the model of choice \
                    to predict the new sample. Can be any \
                    `.pickle` file in the `models` directory. \
                    By default we use a log loss stochastic gradient descent\
                    model trained on the fake news taxonomy features from \
                    *Fake News Is Not Simply False Information, Explication & Taxonomy*")
@click.option('--text-set', default=None,
              #short_help='Path to a `.csv` file containing a series of texts to be classified.'
             )
@click.option('--single-text', default=None,
              #short_help='When flagged, predict a single text provided from \
              #            user input enclosed in double quotes (\"\").'
             )
#@click.argument('--rep',
                #short_help="One of ('bert', 'tfidf', 'taxonomy').",
                #help="Data representation you would like to use to predict the text. \
                #      `taxonomy` (n=14273) is the most complete and well-studied representation. \
                #      Its performance varies between news sources but its associated transformation is very fast. \
                #      `tfidf` (n=654) is robust in most cases but can fail in articles that use idioms or demonstrate uncharacteristic word choices. \
                #      `bert` (n=654, aka Bidirectional Encoder Representation from Transformers) is one of the best text representations, \
                #      with a focus on semantic meaning *in context* with other words within a fixed-length 'attention mask'. \
                #      It requires that text be segmented and run through a large neural network, which may be \
                #      memory-heavy. See the original paper at https://arxiv.org/abs/1810.04805 for more details."
#               )
def predict(text_set, single_text, model, rep):
    """Testing whether function docstring shows up in help"""
    if not rep in model:
        raise UserWarning("Model filename does not contain the name of the data representation. \
                           Verify that the model you are using was trained on the chosen representation \
                           and add that representation to its filename to suppress this warning.")

    try:
        with open(f'models/{model}.pickle', mode='rb') as filein:
            model_obj = pickle.load(filein)
    except:
        raise FileNotFoundError("Model file not found in the `models` directory")

    if not single_text and not text_set:
        raise ValueError("Flag either --single-text or --text-set with its respective text input")
    if single_text and text_set:
        raise ValueError("Only one of the options --single-text or --text-set may be flagged")
    
    if single_text:
        pass
