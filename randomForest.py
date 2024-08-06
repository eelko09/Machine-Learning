#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import argparse

def main():

    parser = argparse.ArgumentParser(description="Process a PepSeq Z score TSV file and perform Random Forest classification.")
    parser.add_argument("-i","--input_file", type=str, help="Path to the input TSV file")
    parser.add_argument("-c","--control_category", type=str, help="String in sample name indicating control category")
    parser.add_argument("-d","--disease_category", type=str, help="String in sample name indicating disease category")
    parser.add_argument("-s","--sblk", type=str, default='Sblk', help="Starting string ID for superblocks")
    parser.add_argument("-n","--n_estimators", type=int, default=1000, help="Number of estimators to use in random forest classifier")
    

    
    args = parser.parse_args()    
    
    # Read in Z-score df
    zdf = pd.read_csv(args.input_file, sep='\t', index_col='Sequence name')
    print(zdf.shape)

    # Remove sblk samples
    zdf = zdf.loc[:, ~zdf.columns.str.startswith(args.sblk)]

    # Get average of replicates and transpose df
    #zdf.columns = zdf.columns.str.rsplit(pat='_', n=3).str[0]
    zdf = zdf.T
    print(zdf.shape)

    # Create column for designation of Control or Disease
    zdf['Category'] = zdf.index.str.split('_').str[0]
    zdf['Category'] = zdf['Category'].str.replace(args.control_category, 'Control')
    zdf['Category'] = zdf['Category'].str.replace(args.disease_category, 'Disease')
    print(zdf.head(5))

    # Labels are the values we want to predict
    labels = np.array(zdf['Category'])
    # Remove the labels from the features
    features = zdf.drop('Category', axis=1)
    # Saving feature names for later use
    feature_list = list(features.columns)
    # Convert to numpy array
    features = np.array(features)

    # Split the data into training and testing sets
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.25, random_state=42)
    print('Training Features Shape:', train_features.shape)
    print('Training Labels Shape:', train_labels.shape)
    print('Testing Features Shape:', test_features.shape)
    print('Testing Labels Shape:', test_labels.shape)

    # Train model
    # Instantiate model with 1000 decision trees
    rf = RandomForestClassifier(n_estimators=args.n_estimators, random_state=42)
    # Train the model on training data
    rf.fit(train_features, train_labels)
    predictions = rf.predict(test_features)
    print(predictions)
    print(test_labels)

    # Save predictions to TSV file
    predictions_df = pd.DataFrame({
        'Predictions': predictions,
        'Actual': test_labels
    })
    predictions_df.to_csv('predictions.tsv', sep='\t', index=False)

    # Calculate accuracy
    count = 0
    for i, ele in enumerate(predictions):
        if ele == test_labels[i]:
            count = count + 1
    print(count / len(predictions))

    # Get numerical feature importances
    importances = list(rf.feature_importances_)
    # List of tuples with variable and importance
    feature_importances = [(feature, round(importance, 5)) for feature, importance in zip(feature_list, importances)]
    # Sort the feature importances by most important first
    feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)
    print(feature_importances[0:20])

    # Save feature importances to TSV file
    feature_importances_df = pd.DataFrame(feature_importances, columns=['Feature', 'Importance'])
    feature_importances_df.to_csv('feature_importances.tsv', sep='\t', index=False)

if __name__ == "__main__":
    main()