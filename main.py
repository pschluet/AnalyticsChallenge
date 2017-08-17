import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, LabelBinarizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

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

def gen_one_hot_feature_mask(features):
    return np.logical_or.reduce(
        ('Department' == features,
        'EducationField' == features,
        'JobRole' == features,
        'MaritalStatus' == features))

class ColEncoder(object):

    def __init__(self, tx_col_names, all_col_names, encoder_type):
        self.tx_col_names = tx_col_names
        self.all_col_names = all_col_names
        self.encoder_type = encoder_type
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
            encoder = self.encoder_type()
            encoder.fit(X[:,cn == self.all_col_names].squeeze())
            self.encoders.append(encoder)
        return self

class DirectColEncoder(object):

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
        encoder_type=LabelBinarizer
    )
    col_enc = ColEncoder(
        tx_col_names=['Department','EducationField','JobRole','MaritalStatus'],
        all_col_names=features,
        encoder_type=LabelEncoder
    )
    col_direct = DirectColEncoder(
        tx_col_names=['BusinessTravel'],
        all_col_names=features,
        encoder_mappings=[
            [('Non-Travel',0), ('Travel_Rarely',1), ('Travel_Frequently',2)]
        ]
    )
    one_hot_enc = OneHotEncoder(n_values='auto', categorical_features=gen_one_hot_feature_mask(features))

    # Build and execute pipeline
    pipe = Pipeline([
        ('binarize', col_bin), # Map select columns to binary 0 or 1
        ('str2int', col_enc), # Map select categorical columns to integers
        ('direct_encode', col_direct), # Map BusinessTravel categories to specific integers (enforce ranking)
        ('onehot', one_hot_enc),
        ('clf', RandomForestClassifier(n_estimators=100))
    ])
    pipe.fit(x,y)
    y_pred = pipe.predict(x)

    # Compute area under the ROC curve
    auc = roc_auc_score(y_true=y, y_score=y_pred)
    print('Area Under ROC: {}'.format(auc))

    # Setup CV model selection search
    # model_params = dict(
    #     one_hot=[None, OneHotEncoder(n_values='auto', categorical_features=)]
    # )