import wandb
import pandas as pd

from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

RANDOM_STATE = 4261998

CLASS_DICT = dict(zip(
    ('real', 'fake', 'opinion', 'polarized', 'satire', 'promotional', 'correction'),
    (1, 2, 3, 5, 7, 9, 11)))
LABELS = list(CLASS_DICT.values())

d_full = pd.read_csv('../data/d_full.csv')
x, y = *(d_full.drop('label', axis=1),
         d_full['label']),
x, y = map(lambda j: j.to_numpy(), (x,y))

FEATURES = d_full.drop('label', axis=1).columns

def train():
    config_defaults = {
        'C': 1.0,
        'gamma': 10.0
    }
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, shuffle=True, random_state=RANDOM_STATE)
    wandb.init(config=config_defaults, magic=True)
    model = SVC(C=wandb.config.C, gamma=wandb.config.gamma, probability=True)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    y_probas = model.predict_proba(x_test)
    wandb.sklearn.plot_classifier(model, x_train, x_test,
                                  y_train, y_test,
                                  y_pred, y_probas,
                                  labels=LABELS,
                                  is_binary=False,
                                  model_name='SVC',
                                  feature_names=FEATURES)
    wandb.log({'Weighted Recall': recall_score(y_true, y_pred, average='weighted'),
               'Micro-Averaged Recall': recall_score(y_true, y_pred, average='micro')})
    wandb.log({'roc': wandb.plots.ROC(y_test, y_probas, labels=LABELS)})
    wandb.log({'pr': wandb.plots.precision_recall(y_test, y_probas, plot_micro=True, labels=LABELS)})
