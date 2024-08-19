#!/usr/bin/env python

import argparse
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import seaborn as sns
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description="Process a PepSeq Z score TSV file and perform Random Forest classification.")
    parser.add_argument("-i", "--input_file", type=str, required=True, help="Path to the input TSV file")
    parser.add_argument("-c", "--control_category", type=str, required=True, help="String in sample name indicating control category")
    parser.add_argument("-d", "--disease_category", type=str, required=True, help="String in sample name indicating disease category")
    parser.add_argument("-s", "--sblk", type=str, default='Sblk', help="Starting string ID for superblocks")
    parser.add_argument("--index_name", type=str, default='Sequence name', help='Name of column to use as index.')
    parser.add_argument("--n_estimators", type=int, default=100, help="The number of features to consider when looking for the best split.")
    parser.add_argument("--max_features",type=str, default='sqrt', help="The number of features to consider when looking for the best split. [“sqrt”, “log2”, None, int or float]")
    parser.add_argument("--max_depth",type=int, default=None, help="The maximum depth of the tree [int]. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.")
    parser.add_argument("--min_samples_split", type=int, default=2, help="The minimum number of samples required to split an internal node.")
    parser.add_argument("--min_samples_leaf", type=int, default=1, help="The minimum number of samples required to be at a leaf node.")
    parser.add_argument("--bootstrap", type=bool, default=True, help="Whether bootstrap samples are used when building trees.")
    parser.add_argument("--criterion", type=str, default='gini', help="The function to measure the quality of a split. Supported criteria are “gini” for the Gini impurity and “log_loss” and “entropy” both for the Shannon information gain.")
    parser.add_argument("--random_hyperparameter_tuning", type=bool, default=False, help="Whether to conduct random hyperparameter tuning and rerun with tuned random forest")
    parser.add_argument("--grid_hyperparameter_tuning", type=bool, default=False, help="Whether to conduct grid hyperparameter tuning and rerun with tuned random forest")
    parser.add_argument("--manual_params", type=bool, default=False, help="Whether to fit random forest using user specified parameters")
    
    args = parser.parse_args()
    
    # Load and preprocess data
    zdf, feature_list, train_features, test_features, train_labels, test_labels = load_and_preprocess_data(args)
    
    # Train and evaluate base model
    rf_base = RandomForestClassifier(random_state=1, n_jobs=-1)
    train_and_evaluate_model(rf_base, train_features, train_labels, test_features, test_labels, feature_list, 'base')

    if args.random_hyperparameter_tuning:
        random_grid = create_random_grid()
        rf_random = perform_hyperparameter_tuning(
            RandomizedSearchCV, random_grid, train_features, train_labels, 'randomizedSearch_bestParams.tsv'
        )
        train_and_evaluate_model(rf_random, train_features, train_labels, test_features, test_labels, feature_list, 'random')

    if args.grid_hyperparameter_tuning:
        grid = create_grid()
        rf_grid = perform_hyperparameter_tuning(
            GridSearchCV, grid, train_features, train_labels, 'gridSearch_bestParams.tsv'
        )
        train_and_evaluate_model(rf_grid, train_features, train_labels, test_features, test_labels, feature_list, 'grid')

    if args.manual_params:
        if args.max_features.replace('.','').isnumeric():
            args.max_features = float(args.max_features)
        rf_manual = RandomForestClassifier(
            random_state=1, n_jobs=-1, n_estimators=args.n_estimators, min_samples_split=args.min_samples_split,
            min_samples_leaf=args.min_samples_leaf, max_features=args.max_features, max_depth=args.max_depth,
            criterion=args.criterion, bootstrap=args.bootstrap
        )
        train_and_evaluate_model(rf_manual, train_features, train_labels, test_features, test_labels, feature_list, 'manual')

###------------------->>> End of main()

def load_and_preprocess_data(args):
    """ Load and preprocess the data. """
    try:
        zdf = pd.read_csv(args.input_file, sep='\t', index_col=args.index_name)
        zdf = zdf.loc[:, ~zdf.columns.str.startswith(args.sblk)].T
        zdf['Category'] = zdf.index.to_series().str.split('_').str[0].replace(
            {args.control_category: 'Control', args.disease_category: 'Disease'}
        )
        labels = zdf.pop('Category').values
        features = zdf.values
        feature_list = zdf.columns.tolist()

        train_features, test_features, train_labels, test_labels = train_test_split(
            features, labels, test_size=0.25,random_state=1, stratify=labels
        )
        return zdf, feature_list, train_features, test_features, train_labels, test_labels
    except Exception as e:
        print(f"Error during data preprocessing: {e}")
        exit()

def create_random_grid():
    """ Create a random grid for RandomizedSearchCV. """
    return {
        'n_estimators': [int(x) for x in np.linspace(start=500, stop=2000, num=10)],
        'max_features': ['sqrt', 'log2', None, 0.01, 0.1, 0.2],
        'max_depth': [int(x) for x in np.linspace(10, 110, num=11)] + [None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4, 10, 20, 50],
        'bootstrap': [True, False],
        'criterion': ['gini', 'entropy']
    }

def create_grid():
    """ Create a parameter grid for GridSearchCV. """
    return {
        'n_estimators': [int(x) for x in np.linspace(start=600, stop=1400, num=5)],
        'max_features': [0.01, 0.1],
        'max_depth': [80,85,90],
        'min_samples_split': [10],
        'min_samples_leaf': [4],
        'bootstrap': [True,False],
        'criterion': ['gini','entropy']
    }

def perform_hyperparameter_tuning(search_method, param_grid, train_features, train_labels, param_filename):
    """ Perform hyperparameter tuning using the specified search method. """
    search = search_method(
        estimator=RandomForestClassifier(), param_distributions=param_grid if isinstance(search_method, RandomizedSearchCV) else param_grid,
        n_iter=200 if isinstance(search_method, RandomizedSearchCV) else None, cv=3, verbose=3, random_state=1, n_jobs=-1
    )
    search.fit(train_features, train_labels)
    best_params = search.best_params_
    print(f"Best parameters from {search_method.__name__}: {best_params}")
    
    best_params_df = pd.DataFrame.from_dict(best_params, orient='index')
    best_params_df.to_csv(param_filename, sep='\t')
    
    return RandomForestClassifier(
        random_state=1, n_estimators=best_params['n_estimators'], min_samples_split=best_params['min_samples_split'],
        min_samples_leaf=best_params['min_samples_leaf'], max_features=best_params['max_features'],
        max_depth=best_params['max_depth'], criterion=best_params['criterion'], bootstrap=best_params['bootstrap'],
        n_jobs=-1
    )

def train_and_evaluate_model(model, train_features, train_labels, test_features, test_labels, feature_list, prefix):
    """ Train and evaluate the model. Save results with the specified prefix. """
    model.fit(train_features, train_labels)
    predictions = model.predict(test_features)
    
    print(f"Classification Report for {prefix} model:")
    print(classification_report(predictions, test_labels))
    
    create_confusion_matrix(test_labels, predictions, f'Confusion_matrix_{prefix}.pdf')
    
    predictions_df = pd.DataFrame({'Predictions': predictions, 'Actual': test_labels})
    predictions_df.to_csv(f'predictions_{prefix}.tsv', sep='\t', index=False)
    
    feature_importances = sorted(
        [(feature, round(importance, 5)) for feature, importance in zip(feature_list, model.feature_importances_)],
        key=lambda x: x[1], reverse=True
    )
    print(f'Top 20 Feature Importances for {prefix}:', feature_importances[:20])
    
    feature_importances_df = pd.DataFrame(feature_importances, columns=['Feature', 'Importance'])
    feature_importances_df.to_csv(f'feature_importances_{prefix}.tsv', sep='\t', index=False)

def create_confusion_matrix(y_test, y_pred_test, outName):
    """ Create and save a confusion matrix. """
    matrix = confusion_matrix(y_test, y_pred_test)
    matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(16, 7))
    sns.set(font_scale=1.4)
    sns.heatmap(matrix, annot=True, annot_kws={'size': 10}, cmap=plt.cm.Greens, linewidths=0.2)
    
    plt.xticks(np.arange(len(['Control', 'Disease'])) + 0.5, ['Control', 'Disease'], rotation=0)
    plt.yticks(np.arange(len(['Control', 'Disease'])) + 0.5, ['Control', 'Disease'], rotation=0)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title('Confusion Matrix for Random Forest Model')
    plt.savefig(outName)

###------------->>>

if __name__ == "__main__":
    main()
