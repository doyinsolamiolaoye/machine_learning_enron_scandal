# Investigating Enron's scandal using Machine Learning
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

The features in the data fall into three major types, namely financial features, email features and POI labels.

* financial features: `['salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees'] (all units are in US dollars)`
* email features: `['to_messages', 'email_address', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi'] (units are generally number of emails messages; notable exception is ‘email_address’, which is a text string)`
* POI label: `[‘poi’] (boolean, represented as integer)`

## Feature Selection

## Algorithm Selection and Tuning

## Validation and Performance

## Discussions and Conclusions


## How to Run the model

