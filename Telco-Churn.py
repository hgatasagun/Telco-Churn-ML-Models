#############################################################
# Telco Customer Churn Prediction: Machine Learning Application
#############################################################

# The task at hand involves developing a machine learning model capable of predicting customers who are likely to churn from the company.

# Dataset Story
###############
# The dataset contains information about a fictional telecom company operating in California. The company provides home phone and internet services 
# to 7043 customers during the third quarter. The dataset reveals which customers have decided to discontinue the services, which have remained loyal, 
# and which have recently subscribed.

# Variables
############
# CustomerId: Customer ID
# SeniorCitizen: Whether the customer is a senior citizen (1, 0)
# Partner: Whether the customer has a partner (Yes, No)
# Dependents: Whether the customer has dependents (Yes, No)
# Tenure: Number of months the customer has stayed with the company
# PhoneService: Whether the customer has a phone service (Yes, No)
# MultipleLines: Whether the customer has multiple lines (Yes, No, No Phone Service)
# InternetService: Internet service provider for the customer (DSL, Fiber Optic, No)
# OnlineSecurity: Whether the customer has online security (Yes, No, No Internet Service)
# OnlineBackup: Whether the customer has online backup (Yes, No, No Internet Service)
# DeviceProtection: Whether the customer has device protection (Yes, No, No Internet Service)
# TechSupport: Whether the customer receives tech support (Yes, No, No Internet Service)
# StreamingTV: Whether the customer has streaming TV (Yes, No, No Internet Service)
# StreamingMovies: Whether the customer has streaming movies (Yes, No, No Internet Service)
# Contract: Contract term of the customer (Month-to-Month, One Year, Two Years)
# PaperlessBilling: Whether the customer has paperless billing (Yes, No)
# PaymentMethod: Payment method of the customer (Electronic Check, Mailed Check, Bank Transfer (Automatic), Credit Card (Automatic))
# MonthlyCharges: Monthly amount charged to the customer
# TotalCharges: Total amount charged to the customer
# Churn: Whether the customer has churned (Yes or No)


################################################
# TASK 1: EXPLORATORY DATA ANALYSIS (EDA)
################################################

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from tabulate import tabulate

from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.model_selection import cross_val_predict

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from sklearn.model_selection import GridSearchCV, cross_validate, RandomizedSearchCV, validation_curve

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

import warnings
warnings.simplefilter("ignore")


df = pd.read_csv('Case studies-Miuul/datasets/Telco-Customer-Churn.csv')
df.columns = [col.lower() for col in df.columns]


def check_df(dataframe, head=5):
    print('################# Columns ################# ')
    print(dataframe.columns)
    print('################# Types  ################# ')
    print(dataframe.dtypes)
    print('##################  Head ################# ')
    print(dataframe.head(head))
    print('#################  Shape ################# ')
    print(dataframe.shape)
    print('#################  NA ################# ')
    print(dataframe.isnull().sum())
    print('#################  Quantiles ################# ')
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99]).T)


check_df(df)


# Capturing numerical and Ccategorical variables
#############################################################
def grab_col_names(dataframe, cat_th=10, car_th=20):

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)



# Correcting data with incorrect types
######################################

empty_rows = df[df['totalcharges'] == ' '] # bos degerler var

df['totalcharges'] = df['totalcharges'].replace(' ', float('nan')).astype(float)

df['totalcharges'].isnull().sum()



# Numerical and categorical variable distribution within the data
##################################################################
cat_cols, num_cols, cat_but_car = grab_col_names(df)

# Categorical:
##############
def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)

for col in cat_cols:
    if df[col].dtypes == 'bool':
        df[col] = df[col].astype(int)
        cat_summary(df, col, plot=True)
    else:
        cat_summary(df, col, plot=True)

# Numerical:
############
def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)

for col in num_cols:
    num_summary(df, col, plot = True)



# Examining categorical variables in relation to the target variable
####################################################################

# 'target_summary_with_cat' fonksiyonu hata verdigi icin bu islemi simdilik sonuc gormek icin yaptim.
def label_encoder(dataframe, col):
    le = LabelEncoder()
    original_values = {}  # Orijinal değerleri saklamak için
    encoded_df = dataframe.copy()
    for c in col:
        le.fit(dataframe[c])
        encoded_df[c] = le.transform(dataframe[c])
        original_values[c] = dict(zip(le.transform(le.classes_), le.classes_))  # Orijinal değerleri sakla
    return encoded_df, original_values

df_encoded, original_values = label_encoder(df, cat_cols)

cat_cols, num_cols, cat_but_car = grab_col_names(df_encoded)


# Identification of Important Variables
#######################################

significant_interactions = []

for col in cat_cols:
    target_summary = df_encoded.groupby(col)['churn'].mean()  # target_summary_with_cat yerine doğrudan gruplama yapılıyor
    if len(target_summary) > 1 and col != 'churn':
        max_diff = target_summary.max() - target_summary.min()
        if max_diff > 0.34:
            significant_interactions.append(col)



# Examination of outliers
###########################################

def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

for col in num_cols:
    print(col, check_outlier(df, col))



# Examination of missing values
##########################################
def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns

missing_values_table(df)


################################################
# TASK 2: FEATURE ENGINEERING
################################################

# Performing necessary procedures for missing and outlier observations
######################################################################

# No outlier observations were found, so no action has been taken.

# 11 missing values in the 'totalcharges' variable have been imputed with the mean.

df['totalcharges'] = df['totalcharges'].fillna(df['totalcharges'].mean())



# Creating new variables
#######################################

# Monthly charges ratio
df['monthlychargesratio'] = df['monthlycharges'] / df['totalcharges']

# Monthly expenditure groups
bins = [0, 40.000, 80.000, 120000]
labels = ['low', 'medium', 'high']
df['monthlychargesgroups'] = pd.cut(df['monthlycharges'], bins=bins, labels=labels)

# Contract duration and number of dependents
df['contractanddependents'] = df['contract'] + '_' + df['dependents']

df.shape


# Encoding
###########

# Binary cols - Label encoder
#############################
def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

binary_cols = [col for col in df.columns if df[col].dtype not in [int, float]
               and df[col].nunique() == 2]

for col in binary_cols:
    label_encoder(df, col)

df.head()
cat_cols, num_cols, cat_but_car = grab_col_names(df)

# Rare encoding
################

def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")

rare_analyser(df, "churn", cat_cols)
# No need for rare encoding!


# One-hot encoder
#################################
def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

cat_cols, num_cols, cat_but_car = grab_col_names(df)

ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]

df = one_hot_encoder(df, ohe_cols)

df.head()
df.shape


# Adding more new variables using 'Significant Interactions'
############################################################
for i, col1 in enumerate(significant_interactions):
    for col2 in significant_interactions[i+1:]:  # Yalnızca sonraki sütunlarla çarp
        new_col_name = f'{col1}_x_{col2}'
        df[new_col_name] = df_encoded[col1.split('_')[0]] * df_encoded[col2.split('_')[0]]

df.shape
df.head()

cat_cols, num_cols, cat_but_car = grab_col_names(df)


# Standardization for Logistic regression and KNN
############################################################
df_ = df.copy()

scaler = StandardScaler()
df_[num_cols] = scaler.fit_transform(df_[num_cols])


#######################################
# TASK 3: MODELLING
#######################################

# 3.1. Logistic Regression
################################################################################################
y = df_["churn"]
X = df_.drop(["churn", 'customerid'], axis=1)

log_model = LogisticRegression(max_iter=1000, random_state=17).fit(X, y)

cv_results_log = cross_validate(log_model,
                            X, y,
                            cv=5,
                            scoring=["accuracy", "f1", "roc_auc"])

y_preds = cross_val_predict(log_model, X, y, cv=5)

report_log = classification_report(y, y_preds, target_names=['Class 0', 'Class 1'])
print(report_log)
# Accuracy - 0.81
# F1 - 0.60

print("Mean: %0.3f and Standard deviation: %0.3f"
      % (cv_results_log['test_roc_auc'].mean(), cv_results_log['test_roc_auc'].std()))
# Mean: 0.850 and Standard deviation: 0.011


# 3.2. KNN
################################################################################################
knn_model = KNeighborsClassifier(n_neighbors=5)

X_c_contiguous = np.ascontiguousarray(X)  # bellek duzeni hatasi

cv_results_knn = cross_validate(knn_model,
                                X_c_contiguous, y,
                                cv=5,
                                scoring=["accuracy", "f1", "roc_auc"])

y_preds_knn = cross_val_predict(knn_model, X_c_contiguous, y, cv=5)

report_knn = classification_report(y, y_preds_knn, target_names=['Class 0', 'Class 1'])
print(report_knn)
# Accuracy 0.77
# F1 score 0.55

print("Mean: %0.3f and Standard deviation: %0.3f"
      % (cv_results_knn['test_roc_auc'].mean(), cv_results_knn['test_roc_auc'].std()))
# Mean: 0.782 and Standard deviation: 0.015


# 3.3. Ensemble Models
############################################
y = df["churn"]
X = df.drop(["churn", "customerid"], axis=1)

def evaluate_model(model, X, y):
    cv_results = cross_validate(model, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])
    report = classification_report(y, cross_val_predict(model, X, y, cv=5))
    mean_accuracy = cv_results['test_accuracy'].mean()
    mean_f1 = cv_results['test_f1'].mean()
    mean_roc_auc = cv_results['test_roc_auc'].mean()
    return {
        "report": report,
        "accuracy_mean": mean_accuracy,
        "f1_mean": mean_f1,
        "roc_auc_mean": mean_roc_auc
    }

# Define the models
models = [
    ("CART", DecisionTreeClassifier(random_state=17)),
    ("Random Forest", RandomForestClassifier(random_state=17)),
    ("Gradient Boosting", GradientBoostingClassifier(random_state=17)),
    ('XGBoost', XGBClassifier(random_state=17)),
    ('LGBM', LGBMClassifier(random_state=17)),
    ('CatBoost', CatBoostClassifier(random_state=17, verbose=False))
]

# Create an empty DataFrame
results_df = pd.DataFrame(columns=["Model", "Accuracy", "F1 Score", "ROC AUC"])

# Evaluate each model and add results to DataFrame
results = []

# Evaluate each model and add results to list
for model_name, model in models:
    model_results = evaluate_model(model, X, y)
    results.append({
        "Model": model_name,
        "Accuracy": model_results["accuracy_mean"],
        "F1 Score": model_results["f1_mean"],
        "ROC AUC": model_results["roc_auc_mean"]
    })

results_df = pd.DataFrame(results)

print(results_df)

#################################
# 4. HYPERPARAMETER OPTIMIZATION
#################################

# 4.1. Logistic regression
###########################
log_param_grid = {
    'C': [0.01, 0.1, 1, 2], # karmasik model
    'penalty': ['l1', 'l2'], # l1-Lasso, l2-Ridge
    'solver': ['liblinear', 'saga'], # genelde liblinear kucuk veri seti icin uygun
    'l1_ratio': [0.2, 0.5, 0.8] # elasticnet
}

y = df_["churn"]
X = df_.drop(["churn", 'customerid'], axis=1)

log_model = LogisticRegression(max_iter=1000, random_state=17).fit(X, y)

log_gs_best = GridSearchCV(log_model,
                           log_param_grid,
                           cv=5,
                           n_jobs=-1,
                           scoring='accuracy',
                           verbose=1).fit(X, y)

log_final = log_model.set_params(**log_gs_best.best_params_).fit(X, y)

cv_results_log = cross_validate(log_final,
                            X,
                            y,
                            cv=5,
                            scoring=["accuracy", "f1", "roc_auc"])

logistic_results = {
    'Accuracy': cv_results_log['test_accuracy'].mean(),
    'F1 Score': cv_results_log['test_f1'].mean(),
    'ROC AUC': cv_results_log['test_roc_auc'].mean()
}

# Feature importance for Logistic Regression
#############################################
def plot_coefficients(model, features):
    coef = model.coef_[0]
    coef_df = pd.DataFrame({'Feature': features.columns, 'Coefficient': coef})
    coef_df = coef_df.sort_values(by='Coefficient', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Coefficient', y='Feature', data=coef_df)
    plt.title('Feature Coefficients')
    plt.xlabel('Coefficient')
    plt.ylabel('Feature')
    plt.show()

plot_coefficients(log_final, X)


# 4.2. Gradient Boosting
####################
gbm_params = {
    "learning_rate": [0.01, 0.1], # ogrenme hizi
    "max_depth": [3, 8, 10],  # deger arttikca asiri ogrenme artar.
    "n_estimators": [100, 200], # toplam agac sayisi
    "subsample": [1, 0.5, 0.7]  # kucuk deger, modelin hizli egitilmesi
}

y = df["churn"]
X = df.drop(["churn", "customerid"], axis=1)

gbm_model = GradientBoostingClassifier(random_state=17)

gbm_best_grid = GridSearchCV(gbm_model,
                             gbm_params,
                             cv=5,
                             n_jobs=-1,
                             verbose=True).fit(X, y)

gbm_final = gbm_model.set_params(**gbm_best_grid.best_params_, random_state=17).fit(X, y)

cv_results_gbm = cross_validate(gbm_final,
                                X, y,
                                cv=5,
                                scoring=["accuracy", "f1", "roc_auc"])

gbm_results = {
    'Accuracy': cv_results_gbm['test_accuracy'].mean(),
    'F1 Score': cv_results_gbm['test_f1'].mean(),
    'ROC AUC': cv_results_gbm['test_roc_auc'].mean()
}

# Top 5 important features for GBM
##################################
def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_,
                                'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature",
                data=feature_imp.sort_values(by="Value",
                                             ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')

plot_importance(gbm_final, X, num=5)

# Top 5 least important features for GBM
########################################
def plot_least_importance(model, features, num=5, save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_,
                                'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature",
                data=feature_imp.sort_values(by="Value",
                                             ascending=True)[0:num])
    plt.title('Least Important Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('least_importances.png')

plot_least_importance(gbm_final, X, num=5)


# 4.3. LightGBM
################
lgbm_params = {
    "learning_rate": [0.01, 0.02],
    "n_estimators": [450, 480, 500],  # 10000 e kadar denendi.
    "colsample_bytree":[0.7, 0,72, 0,75] # her agacin farkli ozellikleri gozlemesi icin
}

y = df["churn"]
X = df.drop(["churn", "customerid"], axis=1)

lgbm_model = LGBMClassifier(random_state=17)

lgbm_best_grid = GridSearchCV(lgbm_model,
                             lgbm_params,
                             cv=5,
                             n_jobs=-1,
                             verbose=True).fit(X, y)

lgbm_final = lgbm_model.set_params(**lgbm_best_grid.best_params_, random_state=17).fit(X, y)

cv_results_lgbm = cross_validate(lgbm_final,
                                X, y,
                                cv=5,
                                scoring=["accuracy", "f1", "roc_auc"])

lgbm_results = {
    'Accuracy': cv_results_lgbm['test_accuracy'].mean(),
    'F1 Score': cv_results_lgbm['test_f1'].mean(),
    'ROC AUC': cv_results_lgbm['test_roc_auc'].mean()
}

# Feature importance for LGBM
##############################
plot_importance(lgbm_final, X, num=5)

plot_least_importance(gbm_final, X, num=5)


# 4.4. CatBoost
#################
y = df["churn"]
X = df.drop(["churn", "customerid"], axis=1)

catboost_params = {
    "iterations": [200, 500],  # n-estimators
    "learning_rate": [0.01, 0.1],
    "depth": [3, 6]
}

catboost_model = CatBoostClassifier(random_state=17)

catboost_best_grid = GridSearchCV(catboost_model,
                                   catboost_params,
                                   cv=5,
                                   n_jobs=-1,
                                   verbose=False).fit(X, y)

catboost_final = catboost_model.set_params(**catboost_best_grid.best_params_, random_state=17).fit(X, y)

cv_results_catboost = cross_validate(catboost_final,
                                     X, y,
                                     cv=5,
                                     scoring=["accuracy", "f1", "roc_auc"])

catboost_results = {
    'Accuracy': cv_results_catboost['test_accuracy'].mean(),
    'F1 Score': cv_results_catboost['test_f1'].mean(),
    'ROC AUC': cv_results_catboost['test_roc_auc'].mean()
}

# Feature importance for CatBoost
#################################
plot_importance(catboost_final, X)
plot_least_importance(gbm_final, X, num=5)

#########################
# 5. Results
#########################
results_df2 = pd.DataFrame([logistic_results, gbm_results, lgbm_results, catboost_results],
                          index=['Logistic Regression', 'Gradient Boosting', 'LightGBM', 'CatBoost'])

print(tabulate(results_df2, headers='keys', tablefmt='pretty'))
