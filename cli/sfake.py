"""
TODO: pass the pooling strategy as a command line option
TODO: +classification report at end of prediction pipeline
TODO: write predictions to new `.csv` or back into the original csv with a `.filter` call
"""
import os
import pdb
import glob
import joblib
import pathlib
import warnings

from datetime import datetime as dt
from functools import partial

import pandas as pd
import numpy as np
import nltk
import torch

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report

from transformers import BertTokenizer, BertModel

from pyfiglet import Figlet

import feature_extraction as fe

HERE = pathlib.Path(__file__).parent

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])

# bidirectional map between integer and string labels
# first direction: int -> str
CLASS_DICT = dict(zip((1, 2, 3, 5, 7, 9),
                      ('real', 'fake', 'opinion', 'polarized', 'satire', 'promotional')))
INT_TARGETS = tuple(CLASS_DICT.keys())
STR_TARGETS = tuple(CLASS_DICT.values())
# next: update the dict with its own inverse for str -> int
CLASS_DICT.update({v:k for k, v in CLASS_DICT.items()})

REPRESENTATIONS = ('bert', 'tfidf', 'taxonomy')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

try:
    with open(os.path.join(HERE, 'models/tfidf/tfidf.pickle'), mode='rb') as filein:
        TFIDF_TRANSFORMER = joblib.load(filein)
except:
    pass

print(f"Using device: {DEVICE}")

def bert_transform(single_text, pooling_strategy='cat'):
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
        with torch.no_grad():
            outputs = model(sentence, segments)
        hidden_states = outputs[2]
        if pooling_strategy == 'cat':
            token_vecs = torch.cat(hidden_states[-5:-1], dim=2).squeeze()
        sentence_embedding = torch.mean(token_vecs, dim=0)
        sent_embeddings.append(sentence_embedding)
    document = torch.mean(torch.stack(sent_embeddings), dim=0)
    return document

def tfidf_transform(texts, **kwargs):
    return TFIDF_TRANSFORMER.transform(texts).todense()

def taxonomy_transform(texts=[], urls=[]):
    if texts is None and urls is None:
        raise ValueError("Must provide either texts, urls or both.")
    return np.array([fe.ArticleVector(text=t if t else '', url=u if u else '').vector for t, u in zip(texts, urls)])

def bert_set(texts, pooling_strategy='cat', **kwargs):
    bert = partial(bert_transform, pooling_strategy=pooling_strategy)
    return torch.stack(list(map(bert, texts))).cpu().numpy()

TRANSFORMS = dict(zip(REPRESENTATIONS, (bert_set, tfidf_transform, taxonomy_transform)))


def predict(single_text, model='svc-tfidf', rep="tfidf", test_set=None, report=False):
    with open(os.path.join(HERE, f'models/{model}.pickle'), mode='rb') as filein:
        model_obj = joblib.load(filein)
    print(f"{model} loaded...")

    print(f"Classifying {'single text' if single_text else 'new test set'} using {rep} data representation...")
    test_set_obj = pd.read_csv(test_set, index_col=0) if test_set else [single_text]

    if isinstance(test_set_obj, pd.DataFrame):
        test_set_obj.fillna('', inplace=True)
        transformed_text = TRANSFORMS[rep](texts=test_set_obj['text'], urls=test_set_obj['url'])
        labels = test_set_obj['label']
    else:
        print(TRANSFORMS[rep])
        transformed_text = TRANSFORMS[rep](texts=test_set_obj, urls=['']).reshape(1,-1)

    predictions = model_obj.predict(transformed_text)

    if single_text:
        print(f"Integer label: {predictions[0]!s}, Class: {CLASS_DICT[predictions[0]]}")

    else:
        if report:
            report_string = classification_report(y_true=labels,
                                                  y_pred=predictions,
                                                  zero_division=0,
                                                  labels=INT_TARGETS,
                                                  target_names=STR_TARGETS)
            print(report_string)

            pathlib.Path(os.path.join(HERE, 'reports/')).mkdir(parents=True, exist_ok=True)
            with open(os.path.join(HERE, "reports/"
                      f"{os.path.basename(os.path.splitext(test_set)[0])}_{model}_"
                      f"{dt.now().strftime('%m%d%H%M')}.txt", mode='w')) as fileout:
                fileout.write(report_string)

        pathlib.Path(os.path.join(HERE, '/predictions/')).mkdir(parents=True, exist_ok=True)
        test_set_obj['predicted'] = predictions
        test_set_obj.to_csv( os.path.join(HERE, "predictions/"
                             f"{os.path.basename(os.path.splitext(test_set)[0])}_{model}_"
                             f"{dt.now().strftime('%m%d%H%M')}.csv"), index=False)

if __name__ == '__main__':
    predict('dummy text')
