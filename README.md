# 2024 March Madness Challenge

# Introduction

March Madness is single elimination college baskeball tournament featuring 64 teams. Typically this tournament is full of upsets creating general excitment. However, this makes predicting a prefect bracket nearly impossible. 

# Methods

Following the march madness 2024 challenge, I generated 100,000 brackets.

I chose to use data from 2000s on to build a model on historical data to predict tournament game performences. For each tournament game, season summary data was used as input features and a probability of winning the game was outputted. Of particular interest as predictors were the following statistics:  'Score', 'Location', 'Field Goals Made', 'Field Goals Attempted', '3 Pt Field Goals Made', '3 Pt Field Goals Attempted', 'Free Throws Made', 'Free Throws Attempted', 'Offensive Rebounds', 'Defensive Rebounds', 'Assists', 'Turnovers', 'Steals', 'Blocks', 'Personal Fouls', 'Field Goal Precentage', '3 poitn Field Goal Percentage', 'Pace', 'Opponent Field Goal Precentage', '3 point Opponent Field Goal Precentage', 'RPI' (An index), 'Pace of play'. 

When training the model, I created a training set where each row contains these columns for both the winning and losing team. Then, I ctreate a class-balanced dataset by randomizing which team was assigned labels (a, b), then the dependent variable, 'winner', stores the a value corresponding to the label of the winning team (a = 1, b = 0). Therefore, models were trained so that the prediction is the probabiltiy that team A wins. I considered two models that can can fit non-linear boundaries : RandomForests, XGbooost, and SVMs with a Gaussian Kernel. I used RandomizedSearchCV to find optimal hyperparameters while keeping computational cost relatively low.

Now I are ready to generate the brackets. First, I pre-calculated these probabilities for all possible matchups to reduce the computation cost of our simulation. Then I procedurely generated results for each game such that $W ~ bernoulli(\hat{p}_{A vs. B})$ where A and B is the specific game I are predicting a winner for. By randomly generating $W$, I determine the result of the game at hand. If W = 1, I choose A to advance; otherwise, I choose B to advance. Doing this game-by-game and round-by-round to generate an entire tournament. I do this 100,000 to generate the portfolio.

# Results

UCONN was the most common winner for the Men's tournament and South Carolina was the most common winner for the Women's tournament. This matches the winners in the real tournaments. Overall, the portfolio finished at the 70% percentile for the Kaggle Competion and I'm really excited to implement more changes for next years tournament.

# Most Common simulated winners 

Mens:

![alt text](https://github.com/stevturn3/2024_march_madness/blob/main/mens.png)


Womens:

![alt text](https://github.com/stevturn3/2024_march_madness/blob/main/womans.png)





March Madness Predictions for 2024 using Machine Learning in Python




Structure of Repository:

-EDA.ipynb: Explaratory Data Analysis

-Kaggle NB.ipynb: Notebook making predictions and used for submission

-Kaggle NB.pdf: Results from my run across 100000 brackets

-functions.py: Functions used throughout my analysis

-march-machine-learning-mania-2024: Data for the competition. 
