# Indentifying Fraud from Enron Email
 Udacity's Intro to Machine Learning, Final Project.

## Project Overview

In this project, I will play detective, and put my machine learning skills to use by building an algorithm to identify Enron Employees who may have committed fraud based on the public Enron financial and email dataset. The goal of this project is to leverage machine learning methods along with financial and email data from Enron to construct a predictive model for identifying potential parties of financial fraud. These parties are termed “persons of interest”.

## Project Introduction

In 2000, Enron was one of the largest companies in the United States. By 2002, it had collapsed into bankruptcy due to widespread corporate fraud. In the resulting Federal investigation, there was a significant amount of typically confidential information entered into public record, including tens of thousands of emails and detailed financial data for top executives. In this project, I will build a person of interest identifier based on financial and email data made public as a result of the Enron scandal.

In addition to being the largest bankruptcy reorganization in American history at that time, Enron was cited as the biggest audit failure From a $90 price per share, to a $1 value represents the huge value loss and scam that happened in Enron. This case has been a point of interest for machine learning analysis because of the huge real-world impact that ML could help out and try to figure out what went wrong and how to avoid it in the future. It would be of great value to find a model that could potentially predict these types of events before much damage is done, so as to permit preventive action. Corporate governance, the stock market, and even the Government would be quite interested in a machine learning model that could signal potential fraud detections before hand.

## Enron Dataset

The [dataset]() is comprised of:

* 146 points, each theoretically representing a person
* 18 of these points is labeled as a POI and 128 as non-POI
* Each point/person is associated with 21 features (14 financial, 6 email, 1 labeled)

There are 3 clear outliers in the data, TOTAL, THE TRAVEL AGENCY IN THE PARK and LOCKHART EUGENE E . The first one seems to be the sum total of all the other data points, while the second outlier is quite bizarre, the last one is a person in the comapny but almost all values of its features are NaN values. These outliers are removed from the dataset for all the analysis. 

## Feature Selection

The features in the data fall into three major types, namely financial features, email features and POI labels.

* financial features: `['salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees'] (all units are in US dollars)`
* email features: `['to_messages', 'email_address', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi'] (units are generally number of emails messages; notable exception is ‘email_address’, which is a text string)`
* POI label: `[‘poi’] (boolean, represented as integer)`

At the begining, I started of by using only the features that has NaN values in less than 50% of the whole dataset, finally, I did feature selection by selecting only the top 5 features which had the highest impact on my model.
I did not create or add new features from the ones originally in the dataset.

## Algorithm Selection and Tuning

I tried out different algorithms: NaiveBayes, DecisionTrees, svc. I used the algorithm defaults for NaiveBayes and applied Kfold crossvalidation on the svc and DescisionTreeClassifier. The perfromance pparameters suggest that the NaiveBayes Algorithm is the best choice.

## Validation and Performance

By running the `poi_id.py` script, my perfromance metrics were 90% accuracy and 60% for both the precision and recall.

Using the 3-fold StratifiedKFold cross-validation as provided in the `tester.py` script, my final model had the following average performance metrics:

- Accuracy:  0.83960

This means my model was 83.96% accurate in predicting whether a person was a POI or not. That is, my model averaged 83.96% accuracy in predicting a 0 or 1 correctly. It is important to get the accuracy as high as possible; however, it is interesting to note that since only 12.5% of the people in the final data set were POIs (18/144), if my model had predicted all 0s, that is, all non-POIs, the accuracy would have been 87.5%. So, accuracy is important, but not the most important metric in quantifying how good the model is.

- Precision: 0.37712

This means that 37.7% of the people my model classified as POIs were actually POIs. So, the ratio of true_POIs/(false_POIs + true_POIs) was 0.37712. This is important because we don't want to "falsely accuse" too many people of being POIs -- so the higher our precision, the lower our percentage of false accusations compared to accurate accusations.

- Recall: 0.31150

This means that 31.15% of the POIs in the data were correctly identified by the model. In mathematical terms, this is the ratio of true_POIs/(false_non_POIs + true_POIs). This is important because we want to catch as many of the POIs as possible, to make sure they face justice.

The Requirements for the final projet is to have at least 0.3 for the precision and recall when the `tester.py` script is run.

## Files in this Repo:
- final_project_dataset.pkl -- Udacity-provided dataset
- Machine Learning file: poi_id.py (run this file if needed)
- Pickle files: my_dataset.pkl, my_classifier.pkl, my_feature_list.pkl
- Tester file: tester.py (unmodified from Udacity-distributed code)
- ./tools -- Udacity-provided tools directory incase you want to clone this repo and run on your system.
