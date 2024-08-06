#!/usr/bin/env python

import argparse
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def main():
    parser = argparse.ArgumentParser(description="Process a PepSeq Z score TSV file and perform Random Forest classification.")
    parser.add_argument("-i", "--input_file", type=str, required=True, help="Path to the input TSV file")
    parser.add_argument("-c", "--control_category", type=str, required=True, help="String in sample name indicating control category")
    parser.add_argument("-d", "--disease_category", type=str, required=True, help="String in sample name indicating disease category")
    parser.add_argument("-s", "--sblk", type=str, default='Sblk', help="Starting string ID for superblocks")
    parser.add_argument("-n", "--n_estimators", type=int, default=1000, help="Number of estimators to use in random forest classifier")
    
    args = parser.parse_args()
    
    # Load data
    try:
        zdf = pd.read_csv(args.input_file, sep='\t', index_col='Sequence name')
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

    # Split the data
    train_features, test_features, train_labels, test_labels = train_test_split(
        features, labels, test_size=0.25, random_state=42)
    
    # Train the RandomForest model
    rf = RandomForestClassifier(n_estimators=args.n_estimators, n_jobs=-1, random_state=42)
    rf.fit(train_features, train_labels)
    
    # Make predictions and calculate accuracy
    predictions = rf.predict(test_features)
    accuracy = np.mean(predictions == test_labels)
    print(f'Accuracy: {accuracy:.2f}')
    
    # Save predictions to TSV file
    predictions_df = pd.DataFrame({'Predictions': predictions, 'Actual': test_labels})
    predictions_df.to_csv('predictions.tsv', sep='\t', index=False)
    
	# Get feature importances
    feature_importances = sorted(
        [(feature, round(importance, 5)) for feature, importance in zip(feature_list, rf.feature_importances_)],
        key=lambda x: x[1], 
        reverse=True
    )
    print('Top 20 Feature Importances:', feature_importances[:20])
      
    # Save feature importances to TSV file
    feature_importances_df = pd.DataFrame(feature_importances, columns=['Feature', 'Importance'])
    feature_importances_df.to_csv('feature_importances.tsv', sep='\t', index=False)

if __name__ == "__main__":
    main()