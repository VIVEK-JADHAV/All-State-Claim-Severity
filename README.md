# Claim Prediction Challenge

### Problem Definition and Dataset
AllState, an Insurance Company in US, wanted to automate their claims prediction process so that the insured has hassle-free transaction.
The data set provided for this competition had two files: train.csv and test.csv. The features were completely masked with 116 categorical features(categories were represented as alphabets) and 14 numerical features. The task was to predict the loss value(claim) for the test data set, making this a Regression problem.

### Performance Metric
The performance metric used for this challenge was Mean Absolute Error (MAE).It is computed as the mean of absolute difference between the predicted loss and the actual loss. The aim was to get MAE value as low as possible on the test data set.

### Exploratory Data Analysis
1. Loss Feature: The loss value was highly right skewed. Training machine learning models on such skewed data would not generalize and performs poorly on test data set. One solution was to apply log-transform. This resulted in normal like distribution.
2. Categorical Features: There were mainly two types of categorical features: binary and multilevel.
   - Most of the binary categorical features had one category more in number than other. In fact, there were features that had around 99.9% of the data belonging to one category. Such features add less value to the model and hence, were dropped.
   - Multilevel categorical features had a minimum of three categories to as large as 326 categories. Even here, they followed a similar pattern of one category clearly dominating the other categories.
3. Numerical Features: All the numerical variables were in the range of 0–1. Univariate plots showed intermediate peaks with tailed distribution. Scatter plot between loss and each numerical feature was plotted with most of the features having log of loss between 4 and 10.
Pearson's co-relation co-efficient between every pair of numerical features were calculated. There were two pairs: (cont1 and cont9) and (cont11 and cont12) that had high co-relation value of greater than 0.9.
Medium article: https://medium.com/@vivekjadhavr/all-state-claim-severity-kaggle-challenge-solution-overview-74586bb31ee6
