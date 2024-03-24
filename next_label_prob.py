"""
Ahnaf Tajwar
Class: CS 677
Date: 3/16/23
Homework Problem # 1
Description of Problem (just a 1-2 line summary!): This problem is to create pandas dataframe from the stock return csv and add the "true label" to each day.
    It also asks to compute the probability of seeing the next true label after a certain history of labels.
"""

import pandas as pd

tickers = ['SPY.csv']

for ticker in tickers:

    # Read csv as pandas dataframe
    df = pd.read_csv(ticker)

    # Add True Label to each return
    df['True Label'] = df['Return'].apply(lambda x: '+' if x >= 0 else '-')
    print(df)

    # Get the year of the first row
    start_year = df['Year'].min()

    # Filter dataframe for the first three years
    df_train_years = df[df['Year'].between(start_year, start_year + 2)]
    print(df_train_years)

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

    # 

    up_probabilities = {}
    down_probabilities = {}

    values_of_k = [1, 2, 3]

    for k in values_of_k:
        # Initializing variables to count sequences
        down_followed_by_up = 0
        down_followed_by_down = 0
        up_followed_by_down = 0
        up_followed_by_up = 0

        # Iterating over each row in the filtered DataFrame with a sliding window of length k+1
        for i in range(len(df_train_years) - k):
            # Get the slice of the dataframe corresponding to the sliding window of length k+1
            window = df_train_years.iloc[i:i+k+1]
            
            # Check if the first k elements in the window are consecutive 'down days'
            if window.iloc[:k]['True Label'].tolist() == ['-'] * k:
                # Check if the last element in the window is an 'up day'
                if window.iloc[-1]['True Label'] == '+':
                    down_followed_by_up += 1
                else:
                    down_followed_by_down += 1
            
            # Check if the first k elements in the window are consecutive 'up days'
            if window.iloc[:k]['True Label'].tolist() == ['+'] * k:
                # Check if the last element in the window is a 'down day'
                if window.iloc[-1]['True Label'] == '-':
                    up_followed_by_down += 1
                else:
                    up_followed_by_up += 1
        

        total_down_seq = down_followed_by_up + down_followed_by_down
        total_up_seq = up_followed_by_down + up_followed_by_up

        # Calculate probability for k down days down followed by up day
        up_probability = down_followed_by_up / total_down_seq if total_down_seq != 0 else 0
        up_probabilities[k] = up_probability

        # Calculate probability for k up days followed by down day
        down_probability = up_followed_by_down / total_up_seq if total_down_seq != 0 else 0
        down_probabilities[k] = down_probability

    # Print probabilities for different values of k
    for k, up_probability in up_probabilities.items():
        print(f"Probability of observing an 'up day' after seeing {k} consecutive 'down days' for {ticker}: {up_probability:.4f}")
    
    for k, down_probability in down_probabilities.items():
        print(f"Probability of observing an 'down day' after seeing {k} consecutive 'up days' for {ticker}: {down_probability:.4f}")