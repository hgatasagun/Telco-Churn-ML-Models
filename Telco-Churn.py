##################################################################
# Telco Müşteri Kaybı Tahminlemesi: Makine Öğrenmesi Uygulaması
#################################################################


# Is Problemi
######################
# Şirketi terk edecek müşterileri tahmin edebilecek bir makine öğrenmesi modeli geliştirilmesi beklenmektedir.


# Veri Seti Hikayesi
#######################
# Telco müşteri kaybı verileri, üçüncü çeyrekte Kaliforniya'daki 7043 müşteriye ev telefonu ve İnternet
# hizmetleri sağlayan hayali bir telekom şirketi hakkında bilgi içerir. Hangi müşterilerin hizmetlerinden
# ayrıldığını, kaldığını veya hizmete kaydolduğunu gösterir.


# Degiskenler
# CustomerId: Müşteri İd’si
# Gender: Cinsiyet
# SeniorCitizen: Müşterinin yaşlı olup olmadığı (1, 0)
# Partner: Müşterinin bir ortağı olup olmadığı (Evet, Hayır)
# Dependents: Müşterinin bakmakla yükümlü olduğu kişiler olup olmadığı (Evet, Hayır)
# tenure: Müşterinin şirkette kaldığı ay sayısı
# PhoneService: Müşterinin telefon hizmeti olup olmadığı (Evet, Hayır)
# MultipleLines: Müşterinin birden fazla hattı olup olmadığı (Evet, Hayır, Telefon hizmeti yok)
# InternetService: Müşterinin internet servis sağlayıcısı (DSL, Fiber optik, Hayır)
# OnlineSecurity: Müşterinin çevrimiçi güvenliğinin olup olmadığı (Evet, Hayır, İnternet hizmeti yok)
# OnlineBackup: Müşterinin online yedeğinin olup olmadığı (Evet, Hayır, İnternet hizmeti yok)
# DeviceProtection: Müşterinin cihaz korumasına sahip olup olmadığı (Evet, Hayır, İnternet hizmeti yok)
# TechSupport: Müşterinin teknik destek alıp almadığı (Evet, Hayır, İnternet hizmeti yok)
# StreamingTV: Müşterinin TV yayını olup olmadığı (Evet, Hayır, İnternet hizmeti yok)
# StreamingMovies: Müşterinin film akışı olup olmadığı (Evet, Hayır, İnternet hizmeti yok)
# Contract: Müşterinin sözleşme süresi (Aydan aya, Bir yıl, İki yıl)
# PaperlessBilling: Müşterinin kağıtsız faturası olup olmadığı (Evet, Hayır)
# PaymentMethod: Müşterinin ödeme yöntemi (Elektronik çek, Posta çeki, Banka havalesi (otomatik), Kredi kartı (otomatik))
# MonthlyCharges: Müşteriden aylık olarak tahsil edilen tutar
# TotalCharges: Müşteriden tahsil edilen toplam tutar
# Churn: Müşterinin kullanıp kullanmadığı (Evet veya Hayır)



################################################
# GOREV 1: KESIFCI VERI ANALIZI
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

################################################
# GOREV 1: KESIFCI VERI ANALIZI
################################################
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


# Adım 1: Numerik ve kategorik değişkenleri yakalayınız.
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



# Adım 2: Gerekli düzenlemeleri yapınız. (Tip hatası olan değişkenler gibi)
###########################################################################

empty_rows = df[df['totalcharges'] == ' '] # bos degerler var

df['totalcharges'] = df['totalcharges'].replace(' ', float('nan')).astype(float)

df['totalcharges'].isnull().sum()



# Adım 3: Numerik ve kategorik değişkenlerin veri içindeki dağılımını gözlemleyiniz.
###################################################################################
cat_cols, num_cols, cat_but_car = grab_col_names(df)

# Kategorik:
############
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

# Numerik:
##########
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



# Adim 4: Kategorik değişkenler ile hedef değişken incelemesini yapınız.
########################################################################

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


# Onemli degisken tespiti !!!!
################################
# def target_summary_with_cat(dataframe, target, categorical_col):
#   print(pd.DataFrame({'TARGET_MEAN': dataframe.groupby(categorical_col)[target].mean()}))

significant_interactions = []

for col in cat_cols:
    target_summary = df_encoded.groupby(col)['churn'].mean()  # target_summary_with_cat yerine doğrudan gruplama yapılıyor
    if len(target_summary) > 1 and col != 'churn':
        max_diff = target_summary.max() - target_summary.min()
        if max_diff > 0.34:
            significant_interactions.append(col)

# for col in cat_cols:
#    target_summary_with_cat(df, 'churn', col)



# Adım 5: Aykırı gözlem var mı inceleyiniz.
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



# Adım 6: Eksik gözlem var mı inceleyiniz.
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
# GOREV 2: FEATURE ENGINEERING
################################################

# Adım 1: Eksik ve aykırı gözlemler için gerekli işlemleri yapınız.
###################################################################

# Aykiri gozlem bulunmadigi icin herhangi bir islem yapilmamistir.

# 'totalcharges' degiskeninde yer alan 11 adet eksik degere ortalama atanmistir.
df['totalcharges'] = df['totalcharges'].fillna(df['totalcharges'].mean())



# Adım 2: Yeni değişkenler oluşturunuz.
#######################################

# Aylık harcama oranı
df['monthlychargesratio'] = df['monthlycharges'] / df['totalcharges']

# Aylık harcama grupları
bins = [0, 40.000, 80.000, 120000]
labels = ['düşük', 'orta', 'yüksek']
df['monthlychargesgroups'] = pd.cut(df['monthlycharges'], bins=bins, labels=labels)

# Sözleşme süresi ve bağımlı sayısı
df['contractanddependents'] = df['contract'] + '_' + df['dependents']

df.shape


# Adım 3: Encoding işlemlerini gerçekleştiriniz.
################################################

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


# Kalanlar icin - One-hot encoder
#################################
def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

cat_cols, num_cols, cat_but_car = grab_col_names(df)

ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]

df = one_hot_encoder(df, ohe_cols)

df.head()
df.shape


# 'target_summary_with_cat' yaparken tespit ettigim onemli degiskenlerin carpimini ekledim.

for i, col1 in enumerate(significant_interactions):
    for col2 in significant_interactions[i+1:]:  # Yalnızca sonraki sütunlarla çarp
        new_col_name = f'{col1}_x_{col2}'
        df[new_col_name] = df_encoded[col1.split('_')[0]] * df_encoded[col2.split('_')[0]]

df.shape
df.head()

cat_cols, num_cols, cat_but_car = grab_col_names(df)



# Adım 4: Numerik değişkenler için standartlaştırma yapınız.
############################################################
df_ = df.copy()

scaler = StandardScaler()
df_[num_cols] = scaler.fit_transform(df_[num_cols])


#######################################
# GOREV 3: MODELLEME
#######################################

# Adım 1: Sınıflandırma algoritmaları ile modeller kurup, accuracy skorlarını inceleyip.
# En iyi 4 modeli seçiniz.
########################################################################################

# 1.1. Logistic Regression
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


# 1.2. KNN
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


# 1.3. Agac modelleri
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


# Adım 2: Seçtiğiniz modeller ile hiperparametre optimizasyonu gerçekleştirin ve bulduğunuz
# hiparparametreler ile modeli tekrar kurunuz.
###########################################################################################

# 'Logistic regression', 'Gradient Boosting', 'LGBM' ve 'CatBoost' icin hiperparametre optimizasyonu

# Logistic regression
#####################
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


# Gradient Boosting
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


# LightGBM
###########
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

plot_importance(lgbm_final, X, num=5)
plot_least_importance(gbm_final, X, num=5)


# CatBoost
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

plot_importance(catboost_final, X)
plot_least_importance(gbm_final, X, num=5)

# Sonuçları birarada göster
results_df2 = pd.DataFrame([logistic_results, gbm_results, lgbm_results, catboost_results],
                          index=['Logistic Regression', 'Gradient Boosting', 'LightGBM', 'CatBoost'])

print(tabulate(results_df2, headers='keys', tablefmt='pretty'))
