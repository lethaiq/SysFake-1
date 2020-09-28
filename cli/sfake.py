import os
import glob
import joblib
import warnings

import pandas as pd

import nltk
import torch
import click

from sklearn.feature_extraction.text import TfidfVectorizer

from transformers import BertTokenizer, BertModel

from pyfiglet import Figlet

import feature_extraction as fe

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])

CLASS_DICT = dict(zip((1, 2, 3, 5, 7, 9, 11),
                      ('real', 'fake', 'opinion', 'polarized', 'satire', 'promotional', 'correction')))

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

click.echo(Figlet(font='larry3d').renderText('sysfake'))
click.echo(f"Using device: {DEVICE}")

def tfidf_transform(single_text, **tfidf_kwargs):
    """
    Create a TF-IDF representation from a single text by transforming it along with all of the training texts.
    """
    return TfidfVectorizer(**tfidf_kwargs).fit_transform(TRAIN_TEXTS['text'].to_list() + [single_text])[-1]

def bert_transform(single_text):
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
        token_vecs = torch.cat(hidden_states[-5:-1], dim=2).squeeze()

        # produce a columnwise average of all of the token embeddings to form a sentence embedding
        sentence_embedding = torch.mean(token_vecs, dim=0)
        sent_embeddings.append(sentence_embedding)

    # pooling step
    # concatenate all of the sentence embeddings into one large document embedding
    document = torch.stack(sent_embeddings)
    return document.cpu().numpy()

@click.command(no_args_is_help=True)
@click.option('--single-text', '-s', default=None,
              help="""Classify a single string enclosed in double-quotes, provided in the command line. The result will be returned in the command line.

                    Example:

                    python sfake.py --single-text \"Lorem ipsum dolor sit amet, consectetur adipiscing elit...\"""")
@click.option('--model', '-m', default='sgd-taxonomy',
              type=click.Choice([os.path.split(os.path.splitext(file)[0])[-1] for file in glob.glob('models\\*[.pickle|.pkl]')]),
              help="Filename of model to use for classification, in the `models` directory.", show_choices=True)
@click.option('--rep', '-r', default='taxonomy', type=click.Choice(('bert', 'tfidf', 'taxonomy')), show_choices=True,
              help="Data representation you wish to use.")
@click.version_option('1.1.0', '--version', '-V', prog_name='SysFake CLI')
def predict(model, rep, single_text):
    """
    Testing whether function docstring shows up in help.
    """
    #click.echo(f"[info] {model}, {rep}")
    try:
        if rep=='tfidf':
            TRAIN_TEXTS = pd.read_csv('data/texts_labelled.csv')
            click.echo("Training texts loaded.")
    except:
        # download from hosted source
        pass

    if not rep in model.lower():
        raise click.BadParameter("Model filename does not contain the name of the data representation. Verify that the model you are using was trained on the chosen representation and add that representation to its filename to suppress this warning.",
                                 param_hint=["--rep", "--model"])

    try:
        with open(f'models/{model}.pickle', mode='rb') as filein:
            model_obj = joblib.load(filein)
        click.echo("Model loaded...")
    except:
        raise FileNotFoundError("Model file not found in the `models` directory")

    click.echo(f"Classifying {'single text' if single_text else 'new test set'} using {rep} data representation...")
    if single_text:
        if rep=='taxonomy':
            vector = fe.ArticleVector(text=single_text).vector
            predicted_label = model_obj.predict([vector])[0]
            click.echo(f"Integer label: {predicted_label!s}, Class: {CLASS_DICT[predicted_label]}")
        #if rep=='tfidf':
        #    vector = tfidf_transform(single_text)
        #    predicted_label = model_obj.predict([vector])[0]
        #    click.echo(f"Integer label: {predicted_label!s}, Class: {CLASS_DICT[predicted_label]}")
        if rep=='bert':
            vector = bert_transform(single_text)
            predicted_label = model_obj.predict(vector)[0]
            click.echo(f"Integer label: {predicted_label!s}, Class: {CLASS_DICT[predicted_label]}")

if __name__ == '__main__':
    predict()
