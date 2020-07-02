# IEEE-CIS-Fraud-Detection
## Business problem : 
When the transaction going on, then detect
whether this transaction is being fraudulent or legit. If some kind of
fraud happens with the transaction the company should immediately
block the card. The model should avoid not to make a transaction
fraud which is in reality is a legit transaction.
## Source of dataset :
https://www.kaggle.com/c/ieee-fraud-detection/data for training data
there are two files(train_transaction.csv, train_identity.csv) and for test
data (test_transaction.csv, test_identity.csv)
## Objective : 
given a transaction predict the probability score of its
being fraudulent transaction. The Model should provide the
probability estimate of a transaction being fraud.
## Constraints : 
It is low latency problem, because we have to predict the
probability of transaction being fraudulent during the transaction is
happening. The model should give probabilistic output.(The model
should give what is the probability that the transaction belongs to
fraud class). Minimize both false positive and false negative(Means
minimizing both falsely predicted legit as fraud and fraud as legit
transaction).
## Performance metrics : 
Area under ROC curve. Because the given
dataset is highly imbalanced, so if we choose accuracy as our metrics
then this can be bias towards majority class, for this problem of
imbalanced dataset we should have to use a metric which can give
score if accuracy of both the class is high and we want accuracy for
both class are high, Area under ROC curve represent accuracy for
both the class by a single value. The AUC score represent the ability
to classify +ve point as +ve and -ve point as -ve.
AUC_ROC is the degree or measure of separability. For example, if my
problem is binary classification and model gave AUC=0.75 then its
mean if we fed two query points to the model first is +ve point and
second is -ve point to the trained model then with 0.75 probability that
model will classify first point as +ve point and with 0.75 probability
model will predict second point as -ve point.
## Steps for solving this problem :
1. First step is EDA on almost all the features of train and test data to
understand the distribution of data , whether this feature will be
helpful in the classification or not by plotting scatter plot and pdf for
numerical features and histogram for categorical features, plotting
violinplot, boxplot and percentile values to check the presence of
outliers.
2. Create some feature from the exiting feature like day, hour from
TransactionDT feature, distribution of the TransactionAmt is looks like
skewed distribution thats why took log of TransactionAmt feature that
looks like a gaussian distribution, create some other features that are
based on the frequency of the some categorical features.
3. Create a feature Email_bin that represent the company name of that
email domain of purchaser email id and receiver email id, and from
DeviceInfo feature create two new feature one is OS type and other is
its version.
4. Printing the %age of NULL for all the id_x features it observes that
many features contain more than 80% of NULL values, so removed
those id_x features which contain >=80% of NULL values.
5. And did Aggregation feature engineering on some features in order
to get good result , i borrowed this feature engineering concept from
kaggle kernels of the other solution.
6. Trained the LGBMClassifier on only Vx features to check the important
Vx features and removed all the Vx features which has importance
score less than 3(threshold).
7. And removed some features which might not help in the separation of
fraud detection from legit detection like TransactionDT,
P_emaildomain, R_emaildomain, TransactionAmt and DeviceInfo.
8. Then fill nan values with median if numerical feature and with mode if
categorical feature.
9.Split the train data in 7:3 ratio of train and CV data using time based
splitting.
10. First train the logistic regression model and did hyperparameter tuning
by using RandomizedSearchCV then got train_AUC=0.8698 and
CV_AUC=0.8327 , 
11. Then train LGBMClassifier and got train_AUC=0.9267 and
CV_AUC=0.8943
##### And got 0.91 score on the kaggle.
12. Plotting the heatmap of correlation matrix of C features and
correlation matrix of D features removed some features which
correlated in very high extent(>0.96), because correlated feature does
not contribute much for classification problem and it can increase the
training time.
13. After removing all the correlated features of C and D and then train
models for Logistic Regression it gave train_AUC=0.8668 and
CV_AUC=0.8362 , and for LGBMClassifier it gave train_AUC=0.9032
and cv_AUC=0.8819.
14. So it shows that removing the correlated features did not change the
performance of model very much but reduces the training time much.

