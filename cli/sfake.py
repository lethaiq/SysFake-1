import os
import glob
import joblib
import warnings

from functools import partial

import pandas as pd
import numpy as np
import nltk
import torch
import click

from sklearn.feature_extraction.text import TfidfVectorizer

from transformers import BertTokenizer, BertModel

from pyfiglet import Figlet

import feature_extraction as fe

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])

# bidirectional map between integer and string labels
# first direction: int -> str
CLASS_DICT = dict(zip((1, 2, 3, 5, 7, 9),
                      ('real', 'fake', 'opinion', 'polarized', 'satire', 'promotional')))
# next: update the dict with its own inverse for str -> int
CLASS_DICT.update({v:k for k, v in CLASS_DICT.items()})

REPRESENTATIONS = ('bert', 'tfidf', 'taxonomy')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

try:
    with open('models/tfidf/tfidf.pickle', mode='rb') as filein:
        TFIDF_TRANSFORMER = joblib.load(filein)
except:
    # download from hosted source
    pass

click.echo(Figlet(font='larry3d').renderText('sysfake'))
click.echo(f"Using device: {DEVICE}")

def bert_transform(single_text, pooling_strategy='cat'):
    """
    Create a BERT embedding representation from a single text by passing the text through the pre-trained network.
    """
    #TODO: pass the pooling strategy as a command line option
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased',
                                  output_hidden_states = True).to(DEVICE)
    sents = nltk.sent_tokenize(single_text)
    marked_sents = ["[CLS] " + sentence + " [SEP]" for sentence in sents]

    tokenized_text = [tokenizer.tokenize(sentence) for sentence in marked_sents]
    segments_ids = [[1] * len(sentence) for sentence in tokenized_text]
    indexed_tokens = [tokenizer.convert_tokens_to_ids(sentence) for sentence in tokenized_text]

    tokens_tensors = [torch.tensor([sentence], device=DEVICE) for sentence in indexed_tokens]
    segments_tensors = [torch.tensor([sentence], device=DEVICE) for sentence in segments_ids]
    sent_embeddings = []

    for sentence, segments in tuple(zip(tokens_tensors, segments_tensors)):
        # run a forward pass through BERT with no gradient evaluation
        with torch.no_grad():
            outputs = model(sentence, segments)

        # from the transformers documentation:
        # 0: last_hidden_state: only the hidden-states of the last layer
        # 1: pooler_output: hidden state associated with classification token (segment 0 or 1)
        # 2: hidden_states: optional, returned when output_hidden_states=True.
        #                   ALL hidden-states of each layer plus initial embedding outputs
        # 3: attentions: optional. returned when output_attentions=True.
        #                Attentions weights after the attention softmax
        # we are interested in the third element, the matrix of all hidden states.
        hidden_states = outputs[2]

        # the second to last hidden layer is widely regarded
        # as holding embeddings that are the least biased 
        # w.r.t. the target labels that still
        # contain information about the learned self-attention
        # if we choose that pooling strategy, then we should extract `hidden_states[-2][0]`
        # i personally opt to concatenate the last four hidden layers here.
        # see: https://mccormickml.com/2019/05/14/BERT-word-embeddings-tutorial/#35-pooling-strategy--layer-choice
        if pooling_strategy == 'cat':
            token_vecs = torch.cat(hidden_states[-5:-1], dim=2).squeeze()

        # produce a columnwise average of all of the token embeddings to form a sentence embedding
        sentence_embedding = torch.mean(token_vecs, dim=0)
        sent_embeddings.append(sentence_embedding)

    # pooling step
    # concatenate all of the sentence embeddings into one large document embedding
    document = torch.mean(torch.stack(sent_embeddings), dim=0)
    return document

def tfidf_transform(texts, **kwargs):
    return TFIDF_TRANSFORMER.transform(texts).todense()

def taxonomy_transform(texts=None, urls=None):
    if texts is None and urls is None:
        raise ValueError("Must provide either texts, urls or both.")
    return np.array([fe.ArticleVector(text=t, url=u).vector for t, u in zip(texts, urls)])

def bert_set(texts, pooling_strategy='cat', **kwargs):
    bert = partial(bert_transform, pooling_strategy=pooling_strategy)
    return torch.stack(list(map(bert, texts))).cpu().numpy()

TRANSFORMS = dict(zip(REPRESENTATIONS, (bert_set, tfidf_transform, taxonomy_transform)))

@click.command(no_args_is_help=True)
@click.option('--single-text', '-s', default=None, type=click.STRING,
              help="""Classify a single string enclosed in double-quotes, provided in the command line. The result will be returned in the command line.

                    Example:

                    python sfake.py --single-text \"Lorem ipsum dolor sit amet, consectetur adipiscing elit...\"""")
@click.option('--test-set', '-t',
              default=None,
              type=click.Path(dir_okay=False, resolve_path=True, allow_dash=False),
              help="""Path to a `.csv` file of labeled texts.
              Its columns must contain:
                    * str/int: labels 
              and one or both of
                    * str: raw text
                    * str: the text's associated URL.
              One column should contain raw text, the other (which should be named 'label') string labels.""")
@click.option('--model', '-m', default='sgd-taxonomy',
              type=click.Choice((os.path.basename(os.path.splitext(file)[0]) for file in glob.glob('models\\*[.pickle|.pkl]')),
                                case_sensitive=True),
              help="Filename of model to use for classification, in the `models` directory, sans file extension.",
              show_choices=True)
@click.option('--rep', '-r', default='taxonomy',
              type=click.Choice(REPRESENTATIONS),
              show_choices=True,
              help="Data representation you wish to use.")
@click.option('--report', '-R', default=False, is_flag=True)
@click.version_option('1.1.0', '--version', '-V', prog_name='SysFake CLI')
def predict(model, rep, single_text, test_set, report):
    """
    A diverse CLI for using SysFake models to classify your own texts.
    """
    #click.echo(f"[info] {model}, {rep}")
    if not single_text and not test_set:
        raise click.BadParameter("Must specify `single_text` or `test_set`.",
                                 param_hint=["--single-text", "--test-set"])
    
    if not rep in model.lower():
        raise click.BadParameter("Model filename does not contain the name of the data representation."
                                 " Verify that the model you are using was trained on the chosen"
                                 " representation and add that representation to its filename to suppress this warning.",
                                 param_hint=["--rep", "--model"])

    try:
        with open(f'models/{model}.pickle', mode='rb') as filein:
            model_obj = joblib.load(filein)
        click.echo(f"{model} loaded...")
    except:
        raise FileNotFoundError(f"{model} not found in the `models` directory")

    click.echo(f"Classifying {'single text' if single_text else 'new test set'} using {rep} data representation...")

    test_set_obj = pd.read_csv(test_set) if test_set else [single_text]
    transformed_text = TRANSFORMS[rep](texts=test_set_obj['text'] if test_set else test_set_obj, urls=test_set_obj['url'] if isinstance(test_set_obj, pd.DataFrame) and 'url' in test_set_obj.columns else None)
    predictions = model_obj.predict(transformed_text)[0]
    if single_text:
        click.echo(f"Integer label: {predictions!s}, Class: {CLASS_DICT[predictions]}")

    #if single_text:
    #    if rep=='taxonomy':
    #        vector = fe.ArticleVector(text=single_text).vector
    #        predicted_label = model_obj.predict([vector])[0]
    #        click.echo(f"Integer label: {predicted_label!s}, Class: {CLASS_DICT[predicted_label]}")
    #    if rep=='tfidf':
    #        vector = TFIDF_TRANSFORMER.transform([single_text])
    #        predicted_label = model_obj.predict(vector)[0]
    #        click.echo(f"Integer label: {predicted_label!s}, Class: {CLASS_DICT[predicted_label]}")
    #    if rep=='bert':
    #        vector = bert_transform(single_text).cpu().numpy()
    #        predicted_label = model_obj.predict(vector)[0]
    #        click.echo(f"Integer label: {predicted_label!s}, Class: {CLASS_DICT[predicted_label]}")

if __name__ == '__main__':
    predict()
