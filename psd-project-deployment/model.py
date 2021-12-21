import pandas as pd
import pickle

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv("Website Phishing.csv")

target = df['Result']
df = df.drop('Result', axis=1)

enc = OneHotEncoder()
enc.fit(df)

x = enc.transform(df)
y = target.values


sm = SMOTE(random_state=2)
x_res, y_res = sm.fit_resample(x, y)

x_train, x_test, y_train, y_test = train_test_split(x_res, y_res , test_size=0.2, random_state=0)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25, random_state=0)


# Random Forest Classifier
model = RandomForestClassifier()
grid = {'n_estimators':[50,100], 
        'criterion': ["gini", "entropy"],
        'min_samples_split': [2, 3, 4],
        'random_state':[1,51,101]}
clf = GridSearchCV(model, grid, scoring='accuracy')
clf.fit(x_val, y_val)
model = RandomForestClassifier(criterion = clf.best_params_['criterion'], 
                               min_samples_split = clf.best_params_['min_samples_split'],
                               n_estimators = clf.best_params_['n_estimators'],
                               random_state = clf.best_params_['random_state'])
model.fit(x_train, y_train)

# Decision Tree Classifier
model_DT = DecisionTreeClassifier()
grid_DT = {'criterion': ["gini", "entropy"],
        'splitter' : ["best", "random"],
        'min_samples_split': [2, 3, 4, 5, 6],
        'random_state':[1,51,101]}
clf_DT = GridSearchCV(model_DT, grid_DT, scoring='accuracy')
clf_DT.fit(x_val, y_val)
model_DT = DecisionTreeClassifier(criterion = clf_DT.best_params_['criterion'],
                                  splitter = clf_DT.best_params_['splitter'],  
                                  min_samples_split = clf_DT.best_params_['min_samples_split'],
                                  random_state = clf_DT.best_params_['random_state'])
model_DT.fit(x_train, y_train)

# Support Vector Machine
model_SVC = SVC()
grid_SVC = {'C': [0.1, 1, 10, 100],
        'kernel' : ["rbf", "poly", "sigmoid", "linear"],
        'gamma' : [1, 0.1, 'scale', 'auto'],
        'random_state':[1,51,101]}
clf_SVC = GridSearchCV(model_SVC, grid_SVC, scoring='accuracy')
clf_SVC.fit(x_val, y_val)
model_SVC = SVC(gamma=clf_SVC.best_params_['gamma'], 
                C=clf_SVC.best_params_['C'], 
                kernel=clf_SVC.best_params_['kernel'],
                random_state = clf_SVC.best_params_['random_state'])
model_SVC.fit(x_train, y_train)

pickle.dump(model, open('model_random_forest.pkl', 'wb'))
pickle.dump(model_DT, open('model_decision_tree.pkl', 'wb'))
pickle.dump(model_SVC, open('model_svm.pkl', 'wb'))
pickle.dump(enc, open('enc.pkl', 'wb'))
