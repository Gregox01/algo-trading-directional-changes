# Trend Reversal Analysis

This code performs trend reversal analysis on financial data using various machine learning techniques.

## Dependencies

The following dependencies are required to run the code:

- pandas
- requests
- numpy
- matplotlib
- scikit-learn (sklearn)
- autosklearn
- gplearn
- deap

## Data Loading

The code assumes that the financial data is stored in a CSV file. You need to replace the `csv_file_path` variable with the actual path to the CSV file.

The code loads the data from the CSV file into a pandas DataFrame.

## Data Preprocessing

- The 'time' column in the data is converted to datetime if it's not already in that format.
- The datetime column is set as the index of the DataFrame.
- Rows with missing values in the 'close' column are dropped.
- Returns over a specified window length are calculated and added as a new column in the DataFrame.
- DC (Directional Change) and OS (Overbought/Oversold) events are identified based on predefined thresholds, and corresponding event columns are added to the DataFrame.

## Data Visualization

The code creates a plot to visualize the financial data and the identified DC and OS events.

## AutoSklearnClassifier

- The code prepares the input features (`X`) and the target variable (`y`) for training the AutoSklearnClassifier.
- The data is split into training and testing sets using the `train_test_split` function from scikit-learn.
- An instance of `AutoSklearnClassifier` is created with specified time and resource limits.
- The classifier is fitted to the training data.
- Predictions are made on the test data using the trained classifier.

## SymbolicRegressor

- The code prepares the input features (`X`) and the target variable (`y`) for training the SymbolicRegressor.
- The data is split into training and testing sets using the `train_test_split` function from scikit-learn.
- An instance of `SymbolicRegressor` is created with specified parameters.
- NaN values in the training and test data are filled with 0.
- The regressor is fitted to the training data.
- Predictions are made on the test data using the trained regressor.

## Genetic Algorithm

The code uses a genetic algorithm to optimize a trading strategy.

- A fitness function is defined to evaluate the profitability of a trading strategy.
- The `deap` library is used to define the genetic operators and create the toolbox.
- The initial population is created.
- The genetic algorithm (`eaSimple`) is executed to evolve the population over a specified number of generations.
- The fitness values of the best individuals in each generation are extracted and stored.

## Additional Analysis

- The code calculates the price change and adds a new column to the DataFrame indicating the direction of the next price change.
- NaN values in the DataFrame are removed.
- The lengths of DC and OS events are calculated based on the start and end points.
- Additional analysis is performed using the lengths as features and the target as the trend.

## Visualization of Trading Strategy

The code visualizes the trading strategy by plotting the closing price and marking the buy and sell points based on the predictions.

---

Please note that this documentation assumes basic familiarity with the libraries used in the code.
