import pickle
import pandas as pd
import matplotlib.patches as mpatches

#importing pickled dataframe
with open('masterdf.pkl', 'rb') as masterdfpkl:
    masterdf= pickle.load(masterdfpkl)

#cleaning data    
def s_party(y):
    if y['party_x'] == "Republican":
        return 'Republican'
    if y['party_x'] == "Democrat":
        return 'Democrat'
    if y['party_x'] == "Independent":
        return 'Independent'
    else:
        return y['group']

masterdf['party'] = masterdf.apply(s_party, axis = 1)
masterdf['group'] = masterdf.apply(s_party, axis = 1)

masterdf.replace('male', 0, inplace = True)
masterdf.replace('female', 1, inplace = True)
masterdf.replace('Republican', 0, inplace = True)
masterdf.replace('Democrat', 1, inplace = True)
masterdf.replace('Senate', 0, inplace = True)
masterdf.replace('governor', 1, inplace = True)

masterdf = masterdf[masterdf.ruling != 'Full Flop']
masterdf = masterdf[masterdf.ruling != 'Half Flip']
masterdf = masterdf[masterdf.ruling != 'No Flip']

#binning the outcome 
def ruling_bin(y):
    if y['ruling'] == 'True':
        return 1
    if y['ruling'] == 'Mostly True':
        return 1
    if y['ruling'] == 'Half-True':
        return 0
    if y['ruling'] == 'Mostly False':
        return 0
    if y['ruling'] == 'False':
        return 0
    if y['ruling'] == 'Pants on Fire!':
        return 0
        
masterdf['ruling_bin_2_4'] = masterdf.apply(ruling_bin, axis = 1)

#making dummies for state and name
dummies = pd.get_dummies(masterdf['state'])
dummies2 = pd.get_dummies(masterdf['name'])
masterdf = pd.concat([masterdf, dummies, dummies2], axis=1)   


#analysis
from sklearn.cross_validation import train_test_split
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.learning_curve import learning_curve
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import auc
from sklearn.metrics import classification_report

from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import NearMiss
from imblearn.under_sampling import CondensedNearestNeighbour
from imblearn.under_sampling import OneSidedSelection
from imblearn.under_sampling import NeighbourhoodCleaningRule
from imblearn.under_sampling import TomekLinks
from imblearn.under_sampling import ClusterCentroids
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.under_sampling import InstanceHardnessThreshold
from imblearn.under_sampling import RepeatedEditedNearestNeighbours
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from imblearn.combine import SMOTEENN
from imblearn.ensemble import EasyEnsemble
from imblearn.ensemble import BalanceCascade
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp

import pandas as pd
%matplotlib inline
from datetime import date
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import statsmodels.api as sm

y, X = np.ravel(masterdf[['ruling_bin_2_4']]), masterdf[['chamber','gender','group','median_age',
 'median_hh_income','pct_aa','pct_asian','pct_college','pct_divorced','pct_english','pct_graduate',
 'pct_hs','pct_islander','pct_latino','pct_less_hs','pct_male','pct_native','pct_never_married',
 'pct_other', 'pct_separated','pct_some_college','pct_white','pct_widowed','total_pop','year',
 'Alabama','Alaska','Arizona','Arkansas','California','Colorado','Connecticut','Delaware',
 'Florida','Georgia','Illinois','Indiana','Iowa','Kansas','Kentucky','Louisiana','Massachusetts',
 'Michigan','Minnesota','Mississippi','Missouri','Montana','Nevada','New Hampshire','New Jersey',
 'New Mexico','North Carolina','Ohio','Oklahoma','Oregon','Pennsylvania','South Carolina',
 'Tennessee','Texas','Utah','Virginia','Washington','Wisconsin','Wyoming',]]
 
#function to run model through each imbalance function and then each model
imbalances = [RandomUnderSampler(), TomekLinks(), ClusterCentroids(), NearMiss(version=1),
              NearMiss(version=2), NearMiss(version=3), CondensedNearestNeighbour(size_ngh=51, n_seeds_S=51),
              OneSidedSelection(size_ngh=51, n_seeds_S=51), 
              InstanceHardnessThreshold(),
              RandomOverSampler(ratio='auto'), SMOTE(ratio='auto', kind='regular'), SMOTE(ratio='auto', kind='borderline1'),
              SMOTE(ratio='auto', kind='borderline2'), 
              SMOTETomek(ratio='auto'), SMOTEENN(ratio='auto')]

classifiers = [LogisticRegression(), SVC(),
                      GaussianNB(), DecisionTreeClassifier(), RandomForestClassifier(),
                      KNeighborsClassifier(n_neighbors=6)]
best_dict = {}
from sklearn import preprocessing


def func():
    for imbalance in imbalances:
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, 
                                                         random_state=4444)
        
        X_train, y_train = imbalance.fit_sample(X_train, y_train)

        

        for clf in classifiers:
            print('-----------------')
            print("%s  "  %imbalance)
            print('-----------------')

            clf.fit(X_train, y_train)
            print('-----------------')
            print("%s   " %clf)
            print('-----------------')
            print("")

            print("Accuracy score", accuracy_score(y_test, clf.predict(X_test)))
            print('auc', roc_auc_score(y_test, clf.predict_proba(X_test)[:,1]))
            print("")

            print(classification_report(y_test, clf.predict(X_test)))
            print("")

            
            print('-----------------')
            best_dict[imbalance] = [clf, roc_auc_score(y_test, clf.predict(X_test))]

#analysis with just cluster centroids(best imbalancer)
classifiers = [LogisticRegression(), SVC(probability=True),
                      GaussianNB(), DecisionTreeClassifier(), RandomForestClassifier(),
                      KNeighborsClassifier(n_neighbors=6)]

cc = ClusterCentroids()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, 
                                                         random_state=4444)
        
X_train, y_train = cc.fit_sample(X_train, y_train)


fprs,tprs,roc_aucs = [],[],[]
for clf in classifiers:
    clf.fit(X_train,y_train)
    y_pred = clf.predict_proba(X_test)[:,1]

    y_true = y_test
    
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    
    fprs.append(fpr)
    tprs.append(tpr)
    roc_aucs.append(roc_auc)
    
#plot ROC of all models
for u,clf in enumerate(classifiers):
    s = str(clf).split('(')[0]
    plt.plot(fprs[u], tprs[u], label='%s (area = %0.2f)' % (s,roc_aucs[u]))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC')
plt.legend(loc="upper left")


#plotting confusion matrix for svm model
from sklearn import svm, datasets
from sklearn.metrics import confusion_matrix

clf = SVC(C = 1, probability = True)
y_pred = clf.fit(X_train, y_train).predict(X_test)


def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)
print('Confusion matrix, without normalization')
print(cm)
plt.figure()
plot_confusion_matrix(cm)

# Normalize the confusion matrix by row (i.e by the number of samples
# in each class)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print('Normalized confusion matrix')
print(cm_normalized)
plt.figure()
plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')

plt.show()

#tuning hyperparameters of svm
# Set the parameters by cross-validation
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1,1e-1,1e-2,1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},{'kernel': ['linear'], 'C': [1, 10, 100, 1000]},{'kernel':['poly'],'degree':[1,2,3]}]

scores = ['precision', 'recall']

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(SVC(C=1), tuned_parameters, cv=5, scoring=score)
    clf.fit(X2_train, y2_train)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_estimator_)
    print()
    print("Grid scores on development set:")
    print()
    for params, mean_score, scores in clf.grid_scores_:
        print("%0.3f (+/-%0.03f) for %r"
              % (mean_score, scores.std() / 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y2_test, clf.predict(X2_test)
    print(classification_report(y_true, y_pred))
    print()
 

 