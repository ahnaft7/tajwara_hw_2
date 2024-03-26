"""
Ahnaf Tajwar
Class: CS 677
Date: 3/23/23
Homework Problem # 5
Description of Problem (just a 1-2 line summary!): This problem is to plot the growth of a portfolio starting with $100 using the best W*, Ensemble, and buy and hold methods.
"""

import matplotlib.pyplot as plt
import pickle

# Getting access to dataframe
with open('TSLA.csv_df.pkl', 'rb') as f:
    df_test_years = pickle.load(f)

print(df_test_years)

# Function to calculate the growth of investment for each method
def calculate_growth(df):
    initial_investment = 100
    ensemble_investment = initial_investment
    W3_investment = initial_investment
    buy_hold_investment = initial_investment
    ensemble_portfolio_value = []
    W3_portfolio_value = []
    buy_hold_portfolio_value = []

    # Iterates through each row and invests if condition is met
    for index, row in df.iterrows():
        if row['Ensemble'] == '+':
            ensemble_investment *= (1 + row['Return'])
        ensemble_portfolio_value.append(ensemble_investment)

        if row['W3'] == '+':
            W3_investment *= (1 + row['Return'])
        W3_portfolio_value.append(W3_investment)

        buy_hold_investment *= (1 + row['Return'])
        buy_hold_portfolio_value.append(buy_hold_investment)

    return ensemble_portfolio_value, W3_portfolio_value, buy_hold_portfolio_value

# Call calculate_growth function
ensemble_portfolio_value, W3_portfolio_value, buy_hold_portfolio_value = calculate_growth(df_test_years)

# Plot portfolio value over time
plt.plot(ensemble_portfolio_value, label="Ensemble")
plt.plot(W3_portfolio_value, label="W3")
plt.plot(buy_hold_portfolio_value, label="Buy and Hold")
plt.xlabel('Day')
plt.ylabel('Portfolio Value ($)')
plt.title('Portfolio Value Over Time for TSLA')
plt.grid(True)
plt.legend()
plt.show()