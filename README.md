# Claim Prediction Challenge

### Problem Definition and Dataset
AllState, an Insurance Company in US, wanted to automate their claims prediction process so that the insured has hassle-free transaction.
The data set provided for this competition had two files: train.csv and test.csv. The features were completely masked with 116 categorical features(categories were represented as alphabets) and 14 numerical features. The task was to predict the loss value(claim) for the test data set, making this a Regression problem.

### Performance Metric
The performance metric used for this challenge was Mean Absolute Error (MAE).It is computed as the mean of absolute difference between the predicted loss and the actual loss. The aim was to get MAE value as low as possible on the test data set.
![PerformanceMetric](https://github.com/VIVEK-JADHAV/ClaimPrediction/blob/master/Images/PerformanceMetric.png)
### Exploratory Data Analysis
1. Loss Feature: The loss value was highly right skewed. Training machine learning models on such skewed data would not generalize and performs poorly on test data set. One solution was to apply log-transform. This resulted in normal like distribution.

![LossFeatureDistribution](https://github.com/VIVEK-JADHAV/ClaimPrediction/blob/master/Images/LossFeatureDistribution.png)

2. Categorical Features: There were mainly two types of categorical features: binary and multilevel.
   - Most of the binary categorical features had one category more in number than other. In fact, there were features that had around 99.9% of the data belonging to one category. Such features add less value to the model and hence, were dropped.
   - Multilevel categorical features had a minimum of three categories to as large as 326 categories. Even here, they followed a similar pattern of one category clearly dominating the other categories.
3. Numerical Features: All the numerical variables were in the range of 0–1. 
   - Univariate plots showed intermediate peaks with tailed distribution. Scatter plot between loss and each numerical feature was plotted with most of the features having log of loss between 4 and 10.
   - Pearson's co-relation co-efficient between every pair of numerical features were calculated. There were two pairs: (cont1 and cont9) and (cont11 and cont12) that had high co-relation value of greater than 0.9.
   - To determine multi co-linearity among numerical features, variance inflation factor(VIF) was used. In this method, each numerical feature is tried to express in terms of other numerical features through Regression and R² value is computed.Higher value of VIF indicates that the feature can be expressed more precisely by other features. This is an iterative method i.e the feature with highest VIF is dropped and the Regression is repeated. Generally, features with VIF value greater than 5 are dropped. The results of VIF method were similar to that of Pearson's co-relation value.
   
### Feature Transform
The categorical features have to be converted to numerical values. There were many categorical features(Eg: cat90) which had few categories only in test data and not in train data. If techniques like one-hot encoding, label encoding are to be used, then both train and test data set have to be merged, resulting in data leakage. To overcome this issue, lexical encoding was performed. In this technique, alphabets were converted to their corresponding numerical values similar to conversion from numerical to binary.

Eg: AC=1x26¹+3x26⁰=29

### Feature Engineering
SVD Features: Single Value Decomposition is a method that decomposes a matrix into three components: U,Σ,V where U(left singular matrix) and V(right singular matrix) represents the given data in a different perspective and Σ(singular) represents strength of each perspective.
For example: Consider a matrix with rows representing different users,columns with different movies and the values in the matrix are the ratings given by users to different movies. Upon applying SVD on this matrix, U matrix would represent the liking of each user to different genres (thriller,crime,romance), V matrix would represent the genre to which each movie is likely to belong to and Σ represents the strength of each genre. Thus, SVD transforms the given data into different concepts(features) which machine learning models find it hard to find out.

### Machine Learning Models
1. Linear SVR: The data was normalized using Sklearn's StandardScalar. The two hyper parameters, C and epsilon were determined using RandomSearchCV. The best values were found to be C=0.01 and epsilon=0.01. The model returned a train MAE value of 1272. 

   A feature selection technique called Recursive Feature Elimination was applied. In this method, features are recursively dropped and best features are retained.This method   marginally improved the MAE value. However, the MAE value for Kaggle test data set was a high score of 1417, indicating that the linear models may not perform satisfactorily.

2. KNN Regressor: The only hyper parameter to be tuned is the number of nearest neighbors (k value). The data was split into two parts: train and cross validation (cv) and were normalized. For different values of k, knn model was fit on train data and evaluated on cv data. The train and cv loss curves were obtained as shown below:

![Knn_Plot](https://github.com/VIVEK-JADHAV/ClaimPrediction/blob/master/Images/KnnPlot.png)

The best value of k was found to 23. Though this model took long time to compute, the MAE on Kaggle test data was improved to 1330.

3. Random Forest: A tree based bagging model, was used to compute the loss. Though Random Forest has very low evaluation time(because of bunch of if-else conditions)and highly parallelizable(a tree can be built independent of other trees), it has very high training time(to determine the best split, it looks at every feature and every value in a feature).The main hyper parameters to be tuned are the number of estimators and tree depth. To train Random forest with 100 trees, each tree having a depth of 50 and considering only 50,000 samples at a time, it took around three hours. The wait did not go in vain as it produced a much improved Kaggle test MAE of 1221.

4. XGBoost Single Model: XGBoost is a tree based boosting method. It is a form of Gradient Boosted decision trees with both row sampling and column sampling. It has a bunch of hyper parameters, which when tuned accurately, gives great results. Some of the important hyper parameters are:
   - Max-depth: This is one of the most important hyper parameter for XGBoost model, determining the depth of each tree. I found a depth of 4 to 7 would work well as the train and cv curves would start to diverge at higher number of rounds.
   - subsample and colsample_bytree: Subsample specifies the percent of rows and colsample_bytree specifies the percent of features to be considered to build decision trees. Subsample of 1 and colsample_bytree of 0.3 gave lower train and cv error and hence, were considered.
   - eta: eta(also called shrinkage parameter)refers to the amount of weight-age to be given to each tree. eta=0.1 was found to be optimal value.
   
With these hyper parameters, the XGBoost model was trained for 1500 rounds with early stopping of 25 rounds(training would stop if there is no improvement in score for 25 rounds). The Kaggle test MAE for this model was an impressive 1130. More importantly, most of the important features were svd features and distance features.

![Feature-Importance](https://github.com/VIVEK-JADHAV/ClaimPrediction/blob/master/Images/XGBoostFeatureImportance.png)

5. XGBoost 5-Fold Model: Since there is lot of randomization in XGBoost model, building 5 models and taking average of the prediction of each model would give better results. In this method, the train data was split into 5 folds using Sklearn's Kfold method. 4 folds were used to train and one fold for cross validation, thus building 5 models. More importantly, each data point is part of train and cv data set at some point in time, allowing to train on entire data set. The Kaggle test MAE obtained was best of all the models with score of 1124.

![Kaggle-Score](https://github.com/VIVEK-JADHAV/ClaimPrediction/blob/master/Images/KaggleScore.png)

   
Medium article: https://medium.com/@vivekjadhavr/all-state-claim-severity-kaggle-challenge-solution-overview-74586bb31ee6
