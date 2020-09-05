#!/usr/bin/python

import sys
import pickle
import pandas as pd
import numpy as np
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn import preprocessing

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

# I would be selecting the features containing less than 50 NaN values
features_list = data_dict['METTS MARK'].keys() # You will need to use more features
features_list.remove('poi')
features_list.remove('other')
features_list.remove('email_address')

df = pd.DataFrame(data_dict).T
df.replace('NaN', np.nan, inplace = True)

# Remove columns with > 50% NaN's which implies that we remove features with values containing more than 50 NaN's
for key in features_list:
    if df[key].isnull().sum() > df.shape[0] * 0.5:
        features_list.remove(key)

features_list = ['poi'] + features_list
print len(features_list)
### Task 2: Remove outliers

data_dict.pop('TOTAL')
data_dict.pop('THE TRAVEL AGENCY IN THE PARK')
data_dict.pop("LOCKHART EUGENE E", 0)

### Store to my_dataset for easy export below.
my_dataset = data_dict

### Task 3: Create new feature(s)


### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)

labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
# from sklearn.naive_bayes import GaussianNB
# clf = GaussianNB()

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, f1_score
# import warnings
# import sklearn.exceptions
# warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)

minmaxscaler = preprocessing.MinMaxScaler()
features_train = minmaxscaler.fit_transform(features_train)
features_test = minmaxscaler.transform(features_test)

#from sklearn.decomposition import PCA

from sklearn.feature_selection import SelectKBest
selector = SelectKBest( k = 5)
selector.fit(features_train, labels_train)
features_train = selector.transform(features_train)
features_test = selector.transform(features_test)

# 1st classifier
# param_grid = {'kernel': ('linear', 'rbf') ,'C':[1e3, 1e4, 5e4, 1e5], 'gamma':[0.0001,0.00005,0.001,0.01]}
# svr = SVC()
# clf = GridSearchCV(svr , param_grid)
def model_score(clf,features_test,labels_test,features_train, labels_train):
    clf = clf.fit(features_train, labels_train)
    pred = clf.predict(features_test)
    print 'accuracy:', accuracy_score(labels_test, pred)
    print 'precison:', precision_score(labels_test, pred)
    print 'f1_score:', f1_score(labels_test, clf.predict(features_test))
    
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
model_score(clf, features_test, labels_test,features_train, labels_train)

# from sklearn.tree import DecisionTreeClassifier
# param_grid = {'criterion': ('gini', 'entropy')}
# clf = GridSearchCV(DecisionTreeClassifier(random_state = 10) , param_grid)
# model_score(clf, features_test, labels_test,features_train, labels_train)


### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)