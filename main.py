import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, LabelBinarizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, make_scorer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.base import BaseEstimator, TransformerMixin

def load_data():
    # Load training data
    train_data = pd.read_csv('AnalyticsChallenge1-Train.csv')
    label_msk = train_data.keys() == 'Attrition'
    x_train = train_data[train_data.keys()[label_msk == False]]
    y_train = train_data[train_data.keys()[label_msk]] == 'Yes'
    y_train = (np.array(y_train).squeeze()) * 1

    # Load test data
    x_test = pd.read_csv('AnalyticsChallenge1-Testing.csv')

    return (np.array(x_train), y_train, np.array(x_test[x_train.keys()]), x_train.keys())

def gen_one_hot_feature_ndx(features):
    return np.where(np.logical_or.reduce(
        ('Department' == features,
        'EducationField' == features,
        'JobRole' == features,
        'MaritalStatus' == features)))

class ColEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, tx_col_names, all_col_names, encoder):
        self.tx_col_names = tx_col_names
        self.all_col_names = all_col_names
        self.encoder = encoder
        self.encoders = []

    def transform(self, X):
        x_out = X.copy()
        for ii, cn in enumerate(self.tx_col_names):
            msk = cn == self.all_col_names
            x_out[:,msk] = self.encoders[ii].transform(x_out[:,msk].squeeze()).reshape(-1,1)
        return x_out

    def fit(self, X, y=None):
        self.binarizers = []
        for cn in self.tx_col_names:
            encoder = eval(self.encoder + '()')
            encoder.fit(X[:,cn == self.all_col_names].squeeze())
            self.encoders.append(encoder)
        return self

class DirectColEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, tx_col_names, all_col_names, encoder_mappings):
        self.tx_col_names = tx_col_names
        self.all_col_names = all_col_names
        self.encoder_mappings = encoder_mappings

    def transform(self, X):
        x_out = X.copy()
        for ii, cn in enumerate(self.tx_col_names):
            col_msk = cn == self.all_col_names
            for jj in range(len(self.encoder_mappings[ii])):
                row_msk = (x_out[:,col_msk] == self.encoder_mappings[ii][jj][0]).squeeze()
                x_out[row_msk,col_msk] = self.encoder_mappings[ii][jj][1]
        return x_out

    def fit(self, X, y=None):
        return self

if __name__=="__main__":
    # Load data
    (x, y, x_test, features) = load_data()

    col_bin = ColEncoder(
        tx_col_names=['Gender','Over18','OverTime'],
        all_col_names=features,
        encoder='LabelBinarizer'
    )
    col_enc = ColEncoder(
        tx_col_names=['Department','EducationField','JobRole','MaritalStatus'],
        all_col_names=features,
        encoder='LabelEncoder'
    )
    col_direct = DirectColEncoder(
        tx_col_names=['BusinessTravel'],
        all_col_names=features,
        encoder_mappings=[
            [('Non-Travel',0), ('Travel_Rarely',1), ('Travel_Frequently',2)]
        ]
    )
    one_hot_enc = OneHotEncoder(n_values='auto', categorical_features=gen_one_hot_feature_ndx(features))

    # Build and execute pipeline
    pipe = Pipeline([
        ('binarize', col_bin), # Map select columns to binary 0 or 1
        ('str2int', col_enc), # Map select categorical columns to integers
        ('direct_encode', col_direct), # Map BusinessTravel categories to specific integers (enforce ranking)
        ('one_hot', one_hot_enc),
        ('clf1', RandomForestClassifier())
    ])

    # Setup & run CV model selection search
    default_max_feat = np.sqrt(len(features)) / len(features)
    model_params = dict(
        one_hot=[None, one_hot_enc],
        clf1__n_estimators=[1, 5, 10, 20, 40, 80, 100, 200, 400, 800],
        clf1__criterion=['gini','entropy'],
        clf1__max_features=[default_max_feat, 0.2, 0.5, 0.8, 1.0],
        clf1__max_depth=[1, 3, 5, 10, 20, 40, None]
    )
    random_search_cv = RandomizedSearchCV(
        pipe,
        param_distributions=model_params,
        n_iter=50, # Number of parameter combinations to try
        scoring=make_scorer(roc_auc_score),
        cv=5, # Stratified K-fold cross validation
        verbose=2,
        n_jobs=4 # Number of jobs to run in parallel
    )
    random_search_cv.fit(x,y)
    bp = 1