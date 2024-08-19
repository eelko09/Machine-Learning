#!/usr/bin/env python

import argparse
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn.inspection import permutation_importance
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
    parser.add_argument("--max_features",type=str, default='sqrt', help="The number of features to consider when looking for the best split. [“sqrt”, “log2”, None, int or float] ")
    parser.add_argument("--max_depth",type=int, default=None, help="The maximum depth of the tree [int]. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.")
    parser.add_argument("--min_samples_split", type=int, default=2, help="The minimum number of samples required to split an internal node.")
    parser.add_argument("--min_samples_leaf", type=int, default=1, help="The minimum number of samples required to be at a leaf node.")
    parser.add_argument("--bootstrap", type=bool, default=True, help="Whether bootstrap samples are used when building trees. If False, the whole dataset is used to build each tree.")
    parser.add_argument("--criterion", type=str, default='gini', help="The function to measure the quality of a split. Supported criteria are “gini” for the Gini impurity and “log_loss” and “entropy” both for the Shannon information gain.")

    parser.add_argument("--random_hyperparameter_tuning", type=bool, default=False, help="Whether to conduct random hyperparameter tuning an rerun with tuned random forest")
    parser.add_argument("--grid_hyperparameter_tuning", type=bool, default=False, help="Whether to conduct grid hyperparameter tuning an rerun with tuned random forest")
    parser.add_argument("--manual_params", type=bool, default=False, help="Whether to fit random forest using user specified parameters")

    
    args = parser.parse_args()
    
    # Load data
    try:
        zdf = pd.read_csv(args.input_file, sep='\t', index_col=args.index_name)
    except Exception as e:
        print(f"Error loading input file: {e}")
        return

    # Preprocess data
    try:
        zdf = zdf.loc[:, ~zdf.columns.str.startswith(args.sblk)]
        zdf = zdf.T
        
        # Create 'Category' column
        zdf['Category'] = zdf.index.to_series().str.split('_').str[0]
        zdf['Category'] = zdf['Category'].replace({args.control_category: 'Control', args.disease_category: 'Disease'})

        # Prepare features and labels
        labels = zdf.pop('Category').values
        features = zdf.values
        feature_list = zdf.columns.tolist()
    except Exception as e:
        print(f"Error during data preprocessing: {e}")
        return

    # Split the data and stratify to make sure test set has the same proportion of groups as full dataset
    train_features, test_features, train_labels, test_labels = train_test_split(
        features, labels, test_size=0.25,stratify=labels)
        
    # Train the RandomForest model with base parameters
    rf_base = RandomForestClassifier(random_state=1,n_jobs = -1)
    rf_base.fit(train_features, train_labels)
    
    # Make predictions and calculate accuracy
    predictions_base = rf_base.predict(test_features)
    print(classification_report(predictions_base, test_labels))
    #accuracy = np.mean(predictions == test_labels)
    #print(f'Accuracy: {accuracy:.2f}')
    
    # Create and output confusion matrix
    create_confusion_matrix(test_labels,predictions_base,'Confusion_matrix_base.pdf')
    
    # Save predictions to TSV file
    predictions_base_df = pd.DataFrame({'Predictions': predictions_base, 'Actual': test_labels})
    predictions_base_df.to_csv('predictions_base.tsv', sep='\t', index=False)
    
    # Get feature importances
    feature_importances_base = sorted(
        [(feature, round(importance, 5)) for feature, importance in zip(feature_list, rf_base.feature_importances_)],
        key=lambda x: x[1], 
        reverse=True
    )
    print('Top 20 Feature Importances Impunity Base:', feature_importances_base[:20])
    
    #result = permutation_importance(rf_base, test_features, test_labels, n_repeats=10, random_state=42, n_jobs=-1)
    #forest_importances = pd.Series(result.importances_mean, index=feature_list)
    #forest_importances.to_csv('feature_permutation_importances_base.tsv', sep='\t')
    
    #print(forest_importances)

    # Save feature importances to TSV file
    feature_importances_base_df = pd.DataFrame(feature_importances_base, columns=['Feature', 'Importance'])
    feature_importances_base_df.to_csv('feature_importances_base.tsv', sep='\t', index=False)

    if args.random_hyperparameter_tuning:
        # Calculate optimal number of trees for dataset
        #optimal_n = pick_optimal_tree_num(train_features,train_labels,args.min_estimator,args.max_estimator,args.estimator_step)
        # Number of trees in random forest
        n_estimators = [int(x) for x in np.linspace(start = 500, stop = 2000, num = 10)]
        # Number of features to consider at every split
        max_features = ['sqrt', 'log2', None, 0.01, 0.1, 0.2]
        # Maximum number of levels in tree
        max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
        max_depth.append(None)
        # Minimum number of samples required to split a node
        min_samples_split = [2, 5, 10]
        # Minimum number of samples required at each leaf node
        min_samples_leaf = [1, 2, 4, 10, 20, 50]
        # Method of selecting samples for training each tree
        bootstrap = [True, False]
        # Function to measure the quality of a split
        criterion = ['gini', 'entropy']
        # Create the random grid
        random_grid = {'n_estimators': n_estimators,
                'max_features': max_features,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf,
                'bootstrap': bootstrap,
                'criterion' : criterion} 
        random_search = RandomizedSearchCV(estimator = RandomForestClassifier(), param_distributions = random_grid, n_iter = 200, cv = 3, verbose=3, random_state=1, n_jobs = -1) 
        random_search.fit(train_features, train_labels) 
        best_paramsD = random_search.best_params_
        print(random_search.best_params_)
        bestParamsDf = pd.DataFrame.from_dict(random_search.best_params_, orient='index')
        bestParamsDf.to_csv('randomizedSearch_bestParams.tsv',sep='\t')

        rf_random = RandomForestClassifier(random_state=1, n_estimators=best_paramsD['n_estimators'],min_samples_split=best_paramsD['min_samples_split'],min_samples_leaf=best_paramsD['min_samples_leaf'],max_features=best_paramsD['max_features'],max_depth=best_paramsD['max_depth'],criterion=best_paramsD['criterion'],bootstrap=best_paramsD['bootstrap'],n_jobs = -1)
        rf_random.fit(train_features, train_labels) 
        
        #rf_random = RandomForestClassifier(n_estimators=889,min_samples_split=5,min_samples_leaf=4,max_features=None, criterion='entropy',bootstrap=True, max_depth=60, random_state=1, oob_score=True, n_jobs = -1)
        #rf_random.fit(train_features, train_labels)
        
        # Make predictions and calculate accuracy
        predictions_random = rf_random.predict(test_features)
        print(classification_report(predictions_random, test_labels))
        
        # Create and output confusion matrix
        create_confusion_matrix(test_labels,predictions_random,'Confusion_matrix_random.pdf')
            
        # Save predictions to TSV file
        predictions_random_df = pd.DataFrame({'Predictions': predictions_random, 'Actual': test_labels})
        predictions_random_df.to_csv('predictions_random.tsv', sep='\t', index=False)
        
        # Get feature importances
        feature_importances_random = sorted(
            [(feature, round(importance, 5)) for feature, importance in zip(feature_list, rf_random.feature_importances_)],
            key=lambda x: x[1], 
            reverse=True
        )
        print('Top 20 Feature Importances Random:', feature_importances_random[:20])
        
        # Save feature importances to TSV file
        feature_importances_random_df = pd.DataFrame(feature_importances_random, columns=['Feature', 'Importance'])
        feature_importances_random_df.to_csv('feature_importances_random.tsv', sep='\t', index=False)
        
    if args.grid_hyperparameter_tuning:
            # Calculate optimal number of trees for dataset
        #optimal_n = pick_optimal_tree_num(train_features,train_labels,args.min_estimator,args.max_estimator,args.estimator_step)
        # Number of trees in random forest
        #n_estimators = [int(x) for x in np.linspace(start = 600, stop = 1400, num = 5)]
        n_estimators = [int(x) for x in np.linspace(start = 600, stop = 1400, num = 2)]
        # Number of features to consider at every split
        #max_features = [0.01, 0.1]
        max_features = [0.01]
        # Maximum number of levels in tree
        #max_depth = [80,85,90]
        max_depth = [80]
        # Minimum number of samples required to split a node
        min_samples_split = [10]
        # Minimum number of samples required at each leaf node
        min_samples_leaf = [4]
        # Method of selecting samples for training each tree
        #bootstrap = [True, False]
        bootstrap = [True]
        # Function to measure the quality of a split
        #criterion = ['gini', 'entropy']
        criterion = ['gini']
        # Create the parameter grid
        grid = {'n_estimators': n_estimators,
                'max_features': max_features,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf,
                'bootstrap': bootstrap,
                'criterion' : criterion} 
        grid_search = GridSearchCV(estimator = RandomForestClassifier(), param_grid = grid, cv = 2, verbose=3, n_jobs = -1) 
        grid_search.fit(train_features, train_labels) 
        grid_best_paramsD = grid_search.best_params_
        print(grid_search.best_params_)
        grid_bestParamsDf = pd.DataFrame.from_dict(grid_search.best_params_, orient='index')
        grid_bestParamsDf.to_csv('gridSearch_bestParams.tsv',sep='\t')

        rf_grid = RandomForestClassifier(random_state=1, n_estimators=grid_best_paramsD['n_estimators'],min_samples_split=grid_best_paramsD['min_samples_split'],min_samples_leaf=grid_best_paramsD['min_samples_leaf'],max_features=grid_best_paramsD['max_features'],max_depth=grid_best_paramsD['max_depth'],criterion=grid_best_paramsD['criterion'],bootstrap=grid_best_paramsD['bootstrap'],n_jobs = -1)
        rf_grid.fit(train_features, train_labels) 
        
        # Make predictions and calculate accuracy
        predictions_grid = rf_grid.predict(test_features)
        print(classification_report(predictions_grid, test_labels))
        
        # Create and output confusion matrix
        create_confusion_matrix(test_labels,predictions_grid,'Confusion_matrix_grid.pdf')
            
        # Save predictions to TSV file
        predictions_grid_df = pd.DataFrame({'Predictions': predictions_grid, 'Actual': test_labels})
        predictions_grid_df.to_csv('predictions_grid.tsv', sep='\t', index=False)
        
        # Get feature importances
        feature_importances_grid = sorted(
            [(feature, round(importance, 5)) for feature, importance in zip(feature_list, rf_grid.feature_importances_)],
            key=lambda x: x[1], 
            reverse=True
        )
        print('Top 20 Feature Importances Grid:', feature_importances_grid[:20])
        
        # Save feature importances to TSV file
        feature_importances_grid_df = pd.DataFrame(feature_importances_grid, columns=['Feature', 'Importance'])
        feature_importances_grid_df.to_csv('feature_importances_grid.tsv', sep='\t', index=False)

        
    if args.manual_params:
        # Check if max_features input is a string or int/float
        if args.max_features.replace('.','').isnumeric():
            args.max_features = float(args.max_features)
        rf_manual = RandomForestClassifier(random_state=1, n_jobs = -1, n_estimators=args.n_estimators,min_samples_split=args.min_samples_split,min_samples_leaf=args.min_samples_leaf,max_features=args.max_features,max_depth=args.max_depth,criterion=args.criterion,bootstrap=args.bootstrap)

        rf_manual = RandomForestClassifier(random_state=1, n_jobs = -1, n_estimators=args.n_estimators,min_samples_split=args.min_samples_split,min_samples_leaf=args.min_samples_leaf,max_features=args.max_features,max_depth=args.max_depth,criterion=args.criterion,bootstrap=args.bootstrap)
        rf_manual.fit(train_features, train_labels) 
        
        # Make predictions and calculate accuracy
        predictions_manual = rf_manual.predict(test_features)
        print(classification_report(predictions_manual, test_labels))
        
        # Create and output confusion matrix
        create_confusion_matrix(test_labels,predictions_manual,'Confusion_matrix_manual.pdf')
            
        # Save predictions to TSV file
        predictions_manual_df = pd.DataFrame({'Predictions': predictions_manual, 'Actual': test_labels})
        predictions_manual_df.to_csv('predictions_manual.tsv', sep='\t', index=False)
        
        # Get feature importances
        feature_importances_manual = sorted(
            [(feature, round(importance, 5)) for feature, importance in zip(feature_list, rf_manual.feature_importances_)],
            key=lambda x: x[1], 
            reverse=True
        )
        print('Top 20 Feature Importances Manual Params:', feature_importances_manual[:20])
        
        # Save feature importances to TSV file
        feature_importances_manual_df = pd.DataFrame(feature_importances_manual, columns=['Feature', 'Importance'])
        feature_importances_manual_df.to_csv('feature_importances_manual.tsv', sep='\t', index=False)


    
    ###-----------------End of main()--------------------------->>>

def pick_optimal_tree_num(train_features,train_labels,start,stop,step):
    n_estimator = list(range(start, stop, step))
    oobScoresD = {}
    oobScores = []
    for n in n_estimator:
        rf = RandomForestClassifier(n_estimators=n, criterion='entropy', max_depth=10, random_state=1, oob_score=True)
        rf.fit(train_features, train_labels)
        oobScoresD[n] = rf.oob_score_
        oobScores.append(rf.oob_score_)
    optimal_tree = min(oobScoresD, key = oobScoresD.get)
    print(f"Optimal Number of Trees: {optimal_tree}")
    df = pd.DataFrame({ 'n': n_estimator, 'oobScore': oobScores })
    df.plot(x='n', y='oobScore')
    plt.savefig('OutOfBagScores.pdf')
    
    return optimal_tree
    
def create_confusion_matrix(y_test,y_pred_test,outName):
    # Get and reshape confusion matrix data
    matrix = confusion_matrix(y_test, y_pred_test)
    matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]

    # Build the plot
    plt.figure(figsize=(16,7))
    sns.set(font_scale=1.4)
    sns.heatmap(matrix, annot=True, annot_kws={'size':10},
            cmap=plt.cm.Greens, linewidths=0.2)

    # Add labels to the plot
    class_names = ['Control', 'Disease']
    tick_marks = np.arange(len(class_names)) + 0.5
    tick_marks2 = tick_marks
    plt.xticks(tick_marks, class_names,rotation=0)
    plt.yticks(tick_marks2, class_names, rotation=0)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title('Confusion Matrix for Random Forest Model')
    plt.savefig(outName)

###------------->>>

if __name__ == "__main__":
    main()
