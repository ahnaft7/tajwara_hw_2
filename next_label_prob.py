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

    df['True Label'] = df['Return'].apply(lambda x: '+' if x >= 0 else '-')
    print(df)

    # Get the year of the first row
    start_year = df['Year'].min()

    # Filter aataframe for the first three years
    df_train_years = df[df['Year'].between(start_year, start_year + 2)]
    print(df_train_years)

    # Count the occurrences of "+" and "-" days in the filtered dataframe
    train_up_days = (df_train_years['True Label'] == '+').sum()
    print("Total up days for first three years:", train_up_days)
    train_down_days = (df_train_years['True Label'] == '-').sum()
    print("Total up down for first three years:", train_down_days)
    train_total_days = len(df_train_years)
    print("Total days for first three years:", train_total_days)

    # Calculate the probability of getting a "+" day for the first three years
    train_up_prob = train_up_days / train_total_days

    print("Probability of getting a '+' day for the first three years:", train_up_prob)
