import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score, cohen_kappa_score, roc_curve, classification_report
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier as rf
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import RandomOverSampler
from sklearn.linear_model import LogisticRegression

le = LabelEncoder()

def data_load_and_process():
    # Load data from file
    data = pd.read_csv('clearAUSdata.csv')  #Loaded files extracted based on data preprocessing code
    
    # Encoding 'rainTomorrow' and 'rainToday'
    data['RainTomorrow'] = data['RainTomorrow'].map({'No': 0, 'Yes': 1})
    data['RainToday'] = data['RainToday'].map({'No': 0, 'Yes': 1})
    
    # Data processing
    x = data.drop(['Date', 'RainTomorrow'], axis=1) # Drop columns 'data' and 'Rain Tomorrow'
    x['Location'] = le.fit_transform(x['Location']) # Convert categorical variables to numeric ones
    y = data['RainTomorrow']
    
   
    # Apply random oversampling for resampling
    oversampler = RandomOverSampler(random_state=12345)
    x_resampled, y_resampled = oversampler.fit_resample(x, y)        
   
    return x_resampled, y_resampled
    


# Using Random Forest to Select Attribute Importance
def feature_importance_select_from_model(X, y):
    # Create a random forest model of 200 decision trees
    selector = SelectFromModel(rf(n_estimators=200, random_state=0))
    selector.fit(X, y)
    support = selector.get_support()
    features = X.loc[:, support].columns.tolist()

    return features

# draw receiver operating characteristic curve graph (ROC)
# use to assess the performance of classification models
def plot_roc_curve(fper, tper):  
    plt.plot(fper, tper, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')  # reference line
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()

def run_model(model, X_train, y_train, X_test, y_test, verbose=True) :
    t_now = time.time() # current time
    if not verbose: # verbose == False, Hide output messages during model training
        model.fit(X_train, y_train, verbose=0)
    else:   # verbose == True
        model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred) 
    time_taken = time.time() - t_now
    print("Accuracy = {}".format(accuracy)) # calculate accuracy
    print("ROC Area under Curve = {}".format(roc_auc))  # under the ROC curve
    print("Time taken = {}".format(time_taken)) # Time taken learning model
    print(classification_report(y_test, y_pred, digits=5))

    # draw roc curve
    probs = model.predict_proba(X_test)  
    probs = probs[:, 1]  
    fper, tper, thresholds = roc_curve(y_test, probs) 
    plot_roc_curve(fper, tper)
    
    confusion_matrix(y_test, y_pred, labels=['0', '1'], normalize='true')

    return model, accuracy, roc_auc, time_taken

# Create a Decision Tree Classification Model
def decision_tree(X_train, y_train, X_test, y_test):
    params_dt = {'max_depth': 18, 'max_features': "sqrt"}
    model_dt = DecisionTreeClassifier(**params_dt)

    return run_model(model_dt, X_train, y_train, X_test, y_test)

# Create an XGBoost classification model
def xgboost_classifier(X_train, y_train, X_test, y_test):
    
    param_grid = {'n_estimators': [50, 100, 150, 200], 'max_depth': [5, 10, 15, 20]}
   
    grid_search = GridSearchCV(xgb.XGBClassifier(), param_grid)
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_

    print("Best parameters : ", best_params)  # Print the best parameters

    model_xgb = xgb.XGBClassifier(**best_params)
    
    return run_model(model_xgb, X_train, y_train, X_test, y_test)


def logistic_reg(X_train, y_train, X_test, y_test):
    model_lr = LogisticRegression(penalty='l1', solver='liblinear', random_state=42)

    return run_model(model_lr, X_train, y_train, X_test, y_test)
    