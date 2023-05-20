# Algorithmic Trading with Directional Changes

This repository contains a research paper titled "Algorithmic trading with directional changes" that discusses a machine learning-based approach to predicting trends in Forex trading. The paper presents a method called the Multi-Threshold Directional Change (MTDC) strategy, which involves using multiple directional change (DC) thresholds to make trading decisions.

## Abstract

The abstract provides a brief overview of the research paper and highlights the key findings. It mentions the MTDC strategy and its outperformance compared to single-threshold strategies and traditional financial benchmarks. However, it also emphasizes that the strategy's performance in future market conditions is not guaranteed, and further research is recommended.

## Introduction

The introduction section provides background information about algorithmic trading and the significance of predicting trends in Forex trading. It introduces the problem statement and explains the motivation behind the research.

## Methodology

This section explains the Multi-Threshold Directional Change (MTDC) strategy in detail. It discusses the classification of trends into DC and overshoot (OS) events or only DC events. It also describes how the strategy assigns weights to each DC threshold and evolves them using a genetic algorithm (GA). The section covers the regression model used to estimate the length of an OS event based on the length of a DC event.

## Experimental Results

The experimental results section presents the findings of applying the MTDC strategy to historical data. It compares the strategy's performance in terms of return and risk-adjusted return (Sharpe ratio) against single-threshold strategies and traditional financial benchmarks such as RSI, EMA, MACD, and BandH strategy.

## Limitations and Future Research

This section discusses the limitations of the research, particularly the reliance on historical data and the need for further testing in different market conditions. It suggests potential areas of future research, including exploring different numbers of thresholds, higher frequency data, and other markets.

## Applying the MTDC Strategy

This section provides a high-level overview of how to apply the MTDC strategy. It outlines the necessary steps and highlights the importance of understanding machine learning algorithms and financial markets.

### Data Collection

This subsection explains the need to collect high-frequency Forex data for the USD/JPY pair. It mentions the paper's use of 10-minute interval data and suggests the possibility of experimenting with different intervals.

### Preprocessing

This subsection describes the preprocessing step, which involves identifying DC and OS events by determining when the price changes by a certain threshold. It emphasizes the significance of this step in preparing the data for classification.

### Classification

In the classification step, a machine learning algorithm is utilized to classify trends into two categories: trends composed of directional change (DC) and overshoot (OS) events, or trends consisting of only DC events. This classification helps determine the appropriate trading actions to take.

To perform this classification, the research paper employs Auto-WEKA, an automated machine learning framework. Auto-WEKA explores a range of machine learning algorithms and their configurations to identify the most effective model for trend classification.

The classification process involves training the machine learning model on a labeled dataset that includes features extracted from the Forex data. These features may include price changes, technical indicators, or other relevant variables. Once trained, the model can classify future trends as either including OS events or not, providing valuable information for subsequent trading decisions.

### Regression

The regression step in the MTDC strategy aims to learn the relationship between the lengths of directional change (DC) events and overshoot (OS) events. The goal is to predict the length of an OS event based on the length of a DC event.

To perform this regression, the research paper employs a symbolic regression algorithm. Symbolic regression is a machine learning technique that seeks to discover mathematical expressions or models that best fit the relationship between input variables (DC event length) and output variables (OS event length).

During the preprocessing stage, the data is prepared with labeled DC and OS event lengths. The symbolic regression algorithm then analyzes the dataset to find an equation or model that can accurately estimate the length of an OS event based on the observed length of a DC event.

The resulting regression model can be utilized in the trading phase to predict the length of an OS event when a trend is classified as consisting only of DC events. This prediction helps determine the expected reversal point for such trends.

### Optimization

Optimization plays a crucial role in determining the weights assigned to each DC threshold in the MTDC strategy. These weights dictate the actions to be taken (buy, sell, or hold) and the timing of those actions.

To optimize the weights, the research paper utilizes a genetic algorithm (GA). A genetic algorithm is an optimization technique inspired by natural selection and genetics. It starts with an initial population of weight configurations and iteratively evolves them to find the most suitable set of weights that maximizes the desired objective, such as return or risk-adjusted return.

The genetic algorithm applies selection, crossover, and mutation operations to create new generations of weight configurations. Each generation undergoes evaluation based on a fitness function that quantifies the performance of the MTDC strategy using historical data. The fittest individuals, which correspond to weight configurations with higher performance, are more likely to be selected for the next generation.

The optimization process continues until a termination criterion is met, such as reaching a maximum number of generations or achieving satisfactory performance. The final set of weights obtained from the genetic algorithm represents an optimized configuration for the MTDC strategy.

### Trading

Once the weights are determined through optimization, the MTDC strategy utilizes these weights in combination with the classification and regression results to make trading decisions. The strategy aims to open and close positions based on predicted trend reversal points.

When a trend is classified as composed of DC and OS events, a trading action is taken at the DC completion (DCC) point. The specific action (buy, sell, or hold) is determined by the weights assigned to the DC thresholds. These weights reflect the confidence in the trend reversal and guide the decision-making process.

For trends classified as consisting only of DC events, the strategy expects a reversal at the end of the sum of the DC event length and the estimated OS event length. The length of the OS event is predicted using the regression model trained during the preprocessing stage.

Based on these predictions and the assigned weights, the MTDC strategy opens positions by buying or selling and closes positions when the predicted reversal points are reached.

It's important to note that the trading decisions should be regularly evaluated and adjusted based on the strategy's performance using metrics like return, risk-adjusted return (Sharpe ratio), maximum drawdown, and standard deviation of returns.

Remember, the MTDC strategy is complex and requires a solid understanding of both machine learning algorithms and Forex trading. Additionally, past performance is not indicative of future results, and trading Forex involves inherent risks.

### Evaluation:

Regular evaluation is essential to assess the performance of the MTDC strategy and make informed decisions about its effectiveness. The evaluation process involves measuring various metrics that reflect the strategy's performance.

Common evaluation metrics used in the research paper and algorithmic trading include:

- **Return**: Measures the overall profitability of the strategy by calculating the percentage increase or decrease in the trading account balance over a specific period.
- **Risk-adjusted return (Sharpe ratio)**: Evaluates the strategy's return in relation to the risk taken. The Sharpe ratio considers both the strategy's return and the volatility or risk of the returns, providing a measure of risk-adjusted performance.
- **Maximum drawdown**: Represents the largest percentage decline in the trading account balance from a peak to a subsequent trough. It provides insight into the strategy's worst-case scenario in terms of loss magnitude.
- **Standard deviation of returns**: Measures the volatility or dispersion of the strategy's returns. A higher standard deviation indicates higher variability in returns, implying greater risk.

By regularly evaluating these metrics, traders and researchers can gain insights into the performance, risk profile, and stability of the MTDC strategy. This evaluation allows for adjustments, refinements, and comparisons with other trading strategies or benchmarks to make informed decisions regarding its continued use or potential improvements.

It's important to note that evaluation should be conducted using appropriate historical data and realistic assumptions. However, past performance is not a guarantee of future results, and trading in financial markets always involves inherent risks.

## Conclusion

The conclusion summarizes the key findings and

 reiterates the outperformance of the MTDC strategy compared to other approaches. It emphasizes the complexity of the strategy and the need for a solid understanding of machine learning and Forex trading.

## Risk Disclosure

This section highlights the risks associated with algorithmic trading and Forex trading in general. It advises caution and recommends only trading with funds that one can afford to lose. It also suggests considering professional assistance if implementing the strategy independently.

## References


### Definitions

- **Directional Change (DC) Events**: In the context of the MTDC strategy, directional change events refer to significant price movements or shifts in the market direction. These events are identified based on predefined thresholds that determine when a price change is considered significant enough to be classified as a directional change. DC events play a crucial role in trend identification and trading decisions.

- **Overshoot (OS) Events**: Overshoot events, in the context of the MTDC strategy, represent price movements that extend beyond the expected reversal point of a trend. These events occur after a directional change event, and their lengths are estimated using regression models. OS events help determine the expected duration of a trend before a reversal occurs.

- **MTDC Strategy**: The Multi-Threshold Directional Change (MTDC) strategy is a machine learning-based approach to predicting trends in Forex trading. It involves classifying trends into two categories: those composed of directional change and overshoot events, and those consisting only of directional change events. The strategy utilizes multiple DC thresholds, assigns weights to each threshold, and employs a genetic algorithm to optimize these weights. The MTDC strategy aims to make trading decisions based on the predicted trend reversal points.

- **Auto-WEKA**: Auto-WEKA is an automated machine learning framework used in the research paper. It explores a range of machine learning algorithms and their configurations to automatically select the most suitable model for trend classification. Auto-WEKA helps streamline the process of model selection and parameter tuning, saving time and effort in the implementation of the MTDC strategy.

- **Weights**: In the context of the MTDC strategy, weights refer to the assigned values associated with each directional change threshold. These weights are determined through optimization using a genetic algorithm. The weights indicate the confidence or importance given to each threshold in making trading decisions. They influence the choice of trading actions (buy, sell, or hold) and the timing of those actions.

- **Sharpe Ratio**: The Sharpe ratio is a widely used metric for measuring the risk-adjusted return of an investment strategy. It quantifies the excess return earned per unit of risk taken. A higher Sharpe ratio indicates better risk-adjusted performance. In the context of evaluating the MTDC strategy, the Sharpe ratio can be calculated to assess how effectively the strategy generates returns relative to the volatility or risk of those returns.

---
