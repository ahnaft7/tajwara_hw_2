"""
Ahnaf Tajwar
Class: CS 677
Date: 3/23/23
Homework Problem # 3
Description of Problem (just a 1-2 line summary!): This problem is to compute the ensemble label by averaging the predicted label.
    The percentage and accuracy of correct labels predicted using ensemble is also computed.
"""

import pandas as pd

tickers = ['SPY.csv', 'TSLA.csv']

def check_sequence(train_data, w, seq):
        neg_string_count = 0
        pos_string_count = 0
        # Iterating over each row in the filtered DataFrame with a sliding window of length k+1
        for i in range(len(train_data) - 1):
            # Get the slice of the dataframe corresponding to the sliding window of length k+1
            window = train_data.iloc[i:i+w+1]
            
            # Check if the first k elements in the window are consecutive 'down days'
            if window.iloc[:w+1]['True Label'].tolist() == seq + ['-']: # check for expected string (pass in as input?)
                neg_string_count += 1
            elif window.iloc[:w+1]['True Label'].tolist() == seq + ['+']:
                pos_string_count += 1
        
        return neg_string_count, pos_string_count

for ticker in tickers:

    print(f"\n*****Start ticker: {ticker}*****\n")

    # Read csv as pandas dataframe
    df = pd.read_csv(ticker)

    # Add True Label to each return
    df['True Label'] = df['Return'].apply(lambda x: '+' if x >= 0 else '-')
    print(df)

    # Get the year of the first row
    start_year = df['Year'].min()

    # Filter dataframe for the first three years
    df_train_years = df[df['Year'].between(start_year, start_year + 2)]

    # Count the occurrences of "+" and "-" days in the filtered dataframe
    train_up_days = (df_train_years['True Label'] == '+').sum()
    print(f"Total up days for first three years for {ticker}:", train_up_days)
    train_down_days = (df_train_years['True Label'] == '-').sum()
    print(f"Total up down for first three years for {ticker}:", train_down_days)
    train_total_days = len(df_train_years)
    print(f"Total days for first three years for {ticker}:", train_total_days)

    # Calculate the probability of getting a "+" day for the first three years
    train_up_prob = train_up_days / train_total_days

    print(f"Probability of getting a '+' day for the first three years for {ticker}:", train_up_prob)
    
    # Get the last two years
    last_two_years = df['Year'].max() - 1  # Assuming years are consecutive

    # Filter DataFrame for the last two years
    df_test_years = df[df['Year'].isin([last_two_years, last_two_years + 1])]
    print(df_test_years)

    print("\n--------------------------------------------------Predict Label-----------------------------------------------------------------\n")

    values_of_w = [2,3,4]

    prediction_accuracy = {}
    pos_prediction_accuracy = {}
    neg_prediction_accuracy = {}
    pos_labels = (df_test_years['True Label'] == '+').sum()
    neg_labels = (df_test_years['True Label'] == '-').sum()  
    print("Number of positive True Labels: ", pos_labels)
    print("Number of negative True Labels: ", neg_labels)

    for w in values_of_w:
        # Initializing variables to count sequences
        neg_string_count = 0
        pos_string_count = 0
        next_result = ''
        new_column = f'W{w}'
        correct_label = 0
        incorrect_label = 0
        correct_pos_label = 0
        incorrect_pos_label = 0
        correct_neg_label = 0
        incorrect_neg_label = 0

        print(f"Creating {new_column} column...Please wait a few seconds\n")

        if new_column not in df_test_years:
            df_test_years[new_column] = ''

        for i in range(w-1,len(df_test_years) - 1):
            # Get the slice of the dataframe corresponding to the sliding window of length k+1
            window = df_test_years.iloc[i-w+1:i+1]
            seq = window['True Label'].tolist()
            neg_string_count, pos_string_count = check_sequence(df_train_years, w, seq)

            if neg_string_count > pos_string_count:
                next_result = '-'
            elif neg_string_count < pos_string_count:
                next_result = '+'
            else:
                if train_up_prob > 0.5:
                    next_result = '+'
                else:
                    next_result = '-'
            df_test_years.iloc[i+1, df_test_years.columns.get_loc(new_column)] = next_result

            if df_test_years.iloc[i+1, df_test_years.columns.get_loc(new_column)] == df_test_years.iloc[i+1, df_test_years.columns.get_loc("True Label")]:
                correct_label += 1
            else:
                incorrect_label += 1

            if df_test_years.iloc[i+1, df_test_years.columns.get_loc(new_column)] == df_test_years.iloc[i+1, df_test_years.columns.get_loc("True Label")] and (df_test_years.iloc[i+1, df_test_years.columns.get_loc("True Label")] == '+'):
                correct_pos_label += 1
            # else:
            #     incorrect_pos_label += 1

            if df_test_years.iloc[i+1, df_test_years.columns.get_loc(new_column)] == df_test_years.iloc[i+1, df_test_years.columns.get_loc("True Label")] and (df_test_years.iloc[i+1, df_test_years.columns.get_loc("True Label")] == '-'):
                correct_neg_label += 1
            # else:
            #     incorrect_neg_label += 1
        
        print(df_test_years)

        prediction_accuracy[new_column] = correct_label / (correct_label + incorrect_label)
        # pos_prediction_accuracy[new_column] = correct_pos_label / (correct_pos_label + incorrect_pos_label)
        # neg_prediction_accuracy[new_column] = correct_neg_label / (correct_neg_label + incorrect_neg_label)

        pos_prediction_accuracy[new_column] = correct_pos_label / (pos_labels)
        neg_prediction_accuracy[new_column] = correct_neg_label / (neg_labels)

    
    print("...Label Prediction Complete")
    
    print("Overall Prediction Accuracy: ", prediction_accuracy)
    print("Positive Prediction Accuracy: ", pos_prediction_accuracy)
    print("Negative Prediction Accuracy: ", neg_prediction_accuracy)

    print("\n--------------------------------------------------Ensemble-------------------------------------------------------------------\n")

    ensemble_column = 'Ensemble'

    print(f"Creating {ensemble_column} column...Please wait a few seconds\n")

    if ensemble_column not in df_test_years:
        df_test_years[ensemble_column] = ''
    
    # Calculate majority value for each row
    majority_value = df_test_years[['W2', 'W3', 'W4']].mode(axis=1)[0]

    # Add 'Ensemble' column to DataFrame with the majority value for each row
    df_test_years['Ensemble'] = majority_value

    print(df_test_years)

    correct_ensemble = 0
    incorrect_ensemble = 0
    correct_pos_ensemble = 0
    incorrect_pos_ensemble = 0
    correct_neg_ensemble = 0
    incorrect_neg_ensemble = 0

    for index, row in df_test_years.iterrows():
        if row['Ensemble'] == row['True Label']:
            correct_ensemble += 1
        else:
            incorrect_ensemble += 1
        
        if row['Ensemble'] == row['True Label'] and row['Ensemble'] == '+':
            correct_pos_ensemble += 1
        # else:
        #     incorrect_pos_ensemble += 1
        
        if row['Ensemble'] == row['True Label'] and row['Ensemble'] == '-':
            correct_neg_ensemble += 1
        # else:
        #     incorrect_neg_ensemble += 1

    print("Number of correctly predicted ensembles: ", correct_ensemble)
    print("Number of incorrectly predicted ensembles: ", incorrect_ensemble)

    correct_ensemble_percent = correct_ensemble / (correct_ensemble + incorrect_ensemble)
    print("Percentage of correctly predicted ensembles: ", correct_ensemble_percent)

    print("Number of correctly predicted positive ensembles: ", correct_pos_ensemble) 
    # print("Number of incorrectly predicted positive ensembles: ", incorrect_pos_ensemble)

    # correct_pos_ensemble_percent = correct_pos_ensemble / (correct_pos_ensemble + incorrect_pos_ensemble)
    correct_pos_ensemble_percent = correct_pos_ensemble / (pos_labels)
    print("Percentage of correctly predicted positive ensembles: ", correct_pos_ensemble_percent)

    print("Number of correctly predicted negative ensembles: ", correct_neg_ensemble) 
    # print("Number of incorrectly predicted negative ensembles: ", incorrect_neg_ensemble) 

    # correct_neg_ensemble_percent = correct_neg_ensemble / (correct_neg_ensemble + incorrect_neg_ensemble)
    correct_neg_ensemble_percent = correct_neg_ensemble / (neg_labels)
    print("Percentage of correctly predicted negative ensembles: ", correct_neg_ensemble_percent) 

    print(f"\n*****End ticker: {ticker}*****\n")