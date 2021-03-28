#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file includes utilities used in data analysis.

@author: Wei Zhao @ Metis, 01/27/2021
"""
#%%
import pickle
from collections import defaultdict
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import RandomizedSearchCV
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import precision_score, recall_score, precision_recall_curve,f1_score
from sklearn.metrics import roc_auc_score, roc_curve
from yellowbrick.model_selection import FeatureImportances
#%%
#--------------------------------------------------------
def save_as_pickle(fn, data):
    """
    Function to save data as a pickled file

    Parameters
    ----------
    fn : str
        File directory.
    data : any type, but recommend dictionary
        data to save.

    Returns
    -------
    None.

    """
    with open(fn, 'wb') as to_write:
        pickle.dump(data, to_write)
    print('Saved data to "' + fn + '"')

#--------------------------------------------------------
def read_from_pickle(fn):
    """
    Function to read data from a pickled file

    Parameters
    ----------
    fn : str
        File directory.
    data : any type, but recommend dictionary
        data to read.

    Returns
    -------
    data : same as data
        Read in this variable.

    """
    with open(fn,'rb') as read_file:
        data = pickle.load(read_file)
    print('Read data from "' + fn + '"')
    return data

#--------------------------------------------------------
def gen_engine(name_of_db):
    """
    Function to generate an engine of sql.

    Parameters
    ----------
    name_of_db : str
        Name of database.

    Returns
    -------
    engine : sqlalchemy.engine.base.Engine
        Engine used with pd.read_sql.

    """
    url = ['postgresql://'
               + 'weizhao:'
               + 'localhost@localhost:'
               +'5432/' + name_of_db]
    engine = create_engine(url[0])
    return engine

#--------------------------------------------------------

def do(query, name_of_db='refinance'):
    """
    Functio to query sql database

    Parameters
    ----------
    query : str
        sql command lines.
    name_of_db : str, optional
        Name of database. The default is 'refinance'.

    Returns
    -------
    Query results from sql database

    """
    engine = gen_engine(name_of_db)

    return pd.read_sql(query, engine)
#--------------------------------------------------------
def race_approval_rate(df):
    """
    Calculate race-wise approval rate

    Parameters
    ----------
    df : pandas data frame
        input data frame.

    Returns
    -------
    d : dictionary
        Keys are race; values are approval rates.

    """
    idx = df['applicant_race_1'].value_counts().index
    d = defaultdict(list)
    for i in idx:
        mask = df['applicant_race_1'] == i
        app_stat= df['action_taken'][mask]
        d[i].append(100*sum(app_stat[app_stat==1])/len(app_stat))
    return d

#--------------------------------------------------------
def ohe_data(x_cat):
    """
    This function converts categorical variables
    to numerical variables

    Parameters
    ----------
    x_cat : pandas data frame
            Input data frame.

    Returns
    -------
    x_cat_tform : pandas data frame
                  data frame including dummy variables.
    """
    ohe = OneHotEncoder(sparse=False, drop='first')
    ohe.fit(x_cat)
    columns = ohe.get_feature_names(x_cat.columns)
    t_x_cat = ohe.transform(x_cat)
    x_cat_tform = pd.DataFrame(t_x_cat,
                               columns=columns,
                               index=x_cat.index)

    return x_cat_tform
#--------------------------------------------------------
def std_data(x_cont):
    """
    This function standardize a dataset
    only including continuous variables.

    Parameters
    ----------
    x_cont : pandas data frame
            Input data frame.

    Returns
    -------
    x_cont_tform : pandas data frame
                  data frame including standardized variables.
    """
    std = StandardScaler()
    std.fit(x_cont)
    t_x_cont = std.transform(x_cont)
    columns = x_cont.columns
    x_cont_tform = pd.DataFrame(t_x_cont,
                               columns=columns,
                               index=x_cont.index)

    return x_cont_tform
#--------------------------------------------------------
def gen_cat(df):
    """
    Generate a data frame only including catgorical variables.

    Parameters
    ----------
    df : pandas data frame
        whole data frame.

    Returns
    -------
    df_new: pandas data frame
        new data frame only including categorical variables.

    """
    feat_cat = ['derived_msa_md', 'county_code',
                'conforming_loan_limit',
                'derived_race', 'derived_sex',
                'hoepa_status',
                'interest_only_payment',
                'balloon_payment', 'occupancy_type',
                'total_units', 'applicant_race_1', 'applicant_sex',
                'applicant_age_above_62', 'co_applicant_age_above_62',
                'derived_loan_product_type',
                'lien_status', 'open_end_line_of_credit',
                'business_or_commercial_purpose'
                ]
    df_new = df[feat_cat]
    return df_new
#--------------------------------------------------------
def gen_cont(df):
    """
    Generate a data frame only including continuous variables.

    Parameters
    ----------
    df : pandas data frame
        whole data frame.

    Returns
    -------
    df_new: pandas data frame
        new data frame only including continuous variables.

    """
    feat_cont = ['loan_term', 'loan_amount',
                 'property_value','loan_to_value_ratio',
                 'income', 'debt_to_income_ratio',
                 'total_age', 'applicant_age',
                 'co_applicant_age'
                ]
    df_new = df[feat_cont]
    return df_new
#--------------------------------------------------------
def gen_hypergrid_for_rf_cv():
    """
    Function to generate a hypergrid as the input of
    random forest for cross-validation classifier.

    Returns
    -------
    random_grid : dictionary
        A dictionary including hyperparameters
        need to be tuned .

    """

    n_estimators = [int(x)
                    for x in np.linspace(start = 100,
                                         stop = 500,
                                         num = 5)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt', 'log2']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 50, num = 5)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10, 15, 20]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]# Create the random grid
    random_grid = {'classifier__n_estimators': n_estimators,
                   'classifier__max_features': max_features,
                   'classifier__max_depth': max_depth,
                   'classifier__min_samples_split': min_samples_split,
                   'classifier__min_samples_leaf': min_samples_leaf,
                   'classifier__bootstrap': bootstrap}
    print(random_grid)
    return random_grid
#--------------------------------------------------------
def get_logreg_models():
    """
    Function to generate a series of logistic regression models
    with varying penalty strength, p
    smaller p --> more penalty
    larger p --> less penalty


    Returns
    -------
    models : dictionary
        logistic regression models with varying penalty values.

    """
    models = dict()
    for p in [0.0, 0.0001, 0.001, 0.01, 0.1, 1.0]:
		# create name for model
        key = '%.4f' % p
        # turn off penalty in some cases
        if p == 0.0:
			# no penalty in this case
            models[key] = LogisticRegression(
                                 solver='newton-cg',
                                 penalty='none', n_jobs=-1)
        else:
            models[key] = LogisticRegression(
                                 solver='newton-cg',
                                 penalty='l2', C=p,
                                 n_jobs=-1)
    return models
#--------------------------------------------------------
def evaluate_model(model, X, y, scoring):
    """
    Function to evaluate model performance

    Parameters
    ----------
    model : dictionary
        A dictionary of models generated with sklearn.
    X : pandas data frame
        training features.
    y : pandas data frame
        training labels.
    scoring : TYPE, optional
        DESCRIPTION. The default is scoring.

    Returns
    -------
    scores : numpy array
        An array of scores.

    """
	# define the evaluation procedure
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
	# evaluate the model
    scores = cross_val_score(model, X, y, scoring=scoring, cv=cv, n_jobs=-1)
    return scores
#--------------------------------------------------------
def make_cv_pipelinie(classifier,
                      categorical_columns,
                      numerical_columns,
                      random_grid,
                      scoring='roc_auc'
                      ):
    """
    Function to make a pipeline for 5-fold cross-validation

    Parameters
    ----------
    categorical_columns : list
        A list of column names from pandas series
        for categorical variables.
    numerical_columns : list
        A list of column names from pandas series
        for numerical variables..
    random_grid : dictionary
        A grid of hyperparameters,
        e.g. the output of  gen_hypergrid_for_rf_cv.
    scoring : dtr, optional
        Available scores are listed here.
        https://scikit-learn.org/stable/modules/model_evaluation.html
        The default is 'roc_auc'.

    Returns
    -------
    clf_random : pipeline
        A pipeline variable. To get the best estimator,
        use rf_random.best_estimator_.named_steps['classifier'],
        which is the tuned model from cross-validation.

    """
    categorical_encoder = OneHotEncoder(handle_unknown='ignore')
    numerical_scalar = StandardScaler()

    preprocessing = ColumnTransformer(
        [('cat', categorical_encoder, categorical_columns),
         ('num', numerical_scalar, numerical_columns)])

    clf_pipe = Pipeline([
        ('preprocess', preprocessing),
        ('classifier', classifier),
    ])

    # Random search of parameters, using stratified 5 fold cross validation,
    # search across all different combinations, and use all available cores

    clf_random = RandomizedSearchCV(estimator = clf_pipe,
                                    param_distributions = random_grid,
                                    n_iter = 100,
                                    cv = StratifiedKFold(5,
                                                         random_state=15),
                                    verbose=2, random_state=15,
                                    scoring=scoring, 
                                    n_jobs = -1) # Fit the random search model

    return clf_random
#--------------------------------------------------------
def make_data_dict(x_train, y_train,
                   x_test, y_test,
                   x_val, y_val,
                   categorical_columns=None,
                   numerical_columns=None
                   ):
    """
    Function to compile the split dataset into a dictionary to save

    Parameters
    ----------
    x_train : pandas data frame
        Training features.
    y_train : pandas data frame
        Training labels.
    x_test : pandas data frame
        Testing features.
    y_test : pandas data frame
        Testing labels.
    x_val : pandas data frame
        Validation features.
    y_val : pandas data frame
        Validation labels.
    categorical_columns : list
        A list of column names from pandas series
        for categorical variables.
    numerical_columns : list
        A list of column names from pandas series
        for numerical variables..

    Returns
    -------
    data_dict : TYPE
        DESCRIPTION.

    """
    data_dict = {'x_train': x_train, 'y_train': y_train,
             'x_test': x_test, 'y_test': y_test,
             'x_val': x_val, 'y_val': y_val,
             'cat_col': categorical_columns,
             'num_col': numerical_columns}

    return data_dict
#--------------------------------------------------------
def make_mdl_eval_pipeline(classifier,
                           categorical_columns,
                           numerical_columns
                           ):
    """
    Function to make a pipeline for model performance evaluaton

    Parameters
    ----------
    classifier : sklearn classifier
        base classifier from sklearn.
    categorical_columns : list
        A list of column names from pandas series
        for categorical variables.
    numerical_columns : list
        A list of column names from pandas series
        for numerical variables.

    Returns
    -------
    clf_pipe : pipeline
        pipeline created.

    """
    categorical_encoder = OneHotEncoder(handle_unknown='ignore')
    numerical_scalar = StandardScaler()

    preprocessing = ColumnTransformer(
        [('cat', categorical_encoder, categorical_columns),
         ('num', numerical_scalar, numerical_columns)])


    clf_pipe = Pipeline([
        ('preprocess', preprocessing),
        ('classifier', classifier),
    ])

    return clf_pipe
#--------------------------------------------------------
def disp_confusion_matrix(cf_matrix,
                          vmin,vmax,
                          cmap='Blues',
                          annot_kws={"size": 15}
                          ):
    """
    Function to display confusion matrix with details reported

    Parameters
    ----------
    cf_matrix : numpy array
        confusion matrix from sklearn.

    Returns
    -------
    ax : handle
        handle from seaborns.heatmap.

    """
    group_names = ['True Neg','False Pos', 'False Neg', 'True Pos']
    
    group_counts = ['{0:0.0f}'.format(value) for value in
     cf_matrix.flatten()]

    group_percentages = ['{0:.2%}'.format(value) for value in
                         (cf_matrix /
                          (np.sum(cf_matrix, axis=1)[:, None])).flatten()]

    labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in
              zip(group_names, group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(2,2)

    ax = sns.heatmap(cf_matrix, annot=labels, annot_kws=annot_kws,
                     fmt='', cmap=cmap, vmin=vmin, vmax=vmax,
                     xticklabels=True, yticklabels=True)

    return ax
 #--------------------------------------------------------
class multi_metrics:
    """
        Function to calculate performance metrics

        Parameters
        ----------
        y_true : pandas data frame
            Actual labels.
        y_pred : pandas data frame
            Predicted labels.

        Returns
        -------
        None.

        """
    def __init__(self, y_true, y_pred, y_score=None):

        self.confusion_matrix = confusion_matrix(y_true, y_pred)
        self.confusion_matrix_norm_by_true = \
            confusion_matrix(y_true, y_pred, normalize='true')
        self.accuracy = accuracy_score(y_true, y_pred)
        self.precision = precision_score(y_true, y_pred)
        self.recall = recall_score(y_true, y_pred)
        self.f1 = f1_score(y_true, y_pred)

        if y_score is not None:
            self.roc_auc = roc_auc_score(y_true, y_score[:,1])
            self.roc_curve =roc_curve(y_true, y_score[:,1])
            self.precision_recall_curve = precision_recall_curve(y_true, y_score[:,1])
#--------------------------------------------------------
def print_metrics(metrics):
    """
    Function to print metrics, including roc_auc,
    accuracy, precision, recall, f1

    Parameters
    ----------
    metrics : Class objects
        The output of multi_metrics.

    Returns
    -------
    None.

    """
    print('roc_auc = {:.2f}'.format(metrics.roc_auc),
          '\naccuracy = {:.2f}'.format(metrics.accuracy),
          '\nprecision = {:.2f}'.format(metrics.precision),
          '\nrecall = {:.2f}'.format(metrics.recall),
          '\nf1 = {:.2f}'.format(metrics.f1),
          )
#--------------------------------------------------------
def get_feat_importance_logreg(mdl_logreg, x, y):
    """
    Calculate feature importance for logistic regression.
    This is similar to random forest.

    Parameters
    ----------
    mdl_logreg : sklearn classifier
        classifier.
    x : pandas data frame
        features.
    y : pandas data frame
        Label.

    Returns
    -------
    feat_importance : TYPE
        DESCRIPTION.

    """
    visualizer = FeatureImportances(mdl_logreg,
                                    title='Logistic regression')
    visualizer.fit(x, y)
    visualizer.ax.remove()
    feat_importance = visualizer.feature_importances_[::-1]
    return feat_importance, visualizer
