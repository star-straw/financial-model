# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
# 本地化代码省去了数据统计及其可视化 仅在评估时会有分析可视化

import numpy as np  # 数值计算
import pandas as pd  # 数据处理与分析
import os  # 文件处理
import joblib  # 模型管理
import warnings  # 警告信息处理

# 以下均为可视化库
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import ticker
import seaborn as sns

# setting up options
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
warnings.filterwarnings('ignore')  # 忽略警告


### 数据导入
df_data = pd.read_csv('data.csv')
df_model = df_data.copy()


### 数据处理
# 对X进行处理
# Rename target column
df_model = df_model.rename(columns={'conversion': 'target'})
# Rename & Label encode treatment column
df_model = df_model.rename(columns={'offer': 'treatment'})
df_model.treatment = df_model.treatment.map({'No Offer': 0, 'Buy One Get One': -1, 'Discount': 1})
# 将定类变量进行独热编码
df_model = pd.get_dummies(df_model)

#Y值分类
#后续将对两种策略分别统计
df_model_bogo = df_model[df_model.treatment <= 0] # include no offer, bogo
df_model_discount = df_model[df_model.treatment >= 0] # include no offer, discount

# 将客户分为良好目标客户和不良目标客户
def declare_tc(df:pd.DataFrame): #二分类
    df['target_class'] = 0 # CN and TR
    df.loc[(df.treatment == 0) & (df.target != 0) | ((df.treatment != 0) & (df.target == 0)),'target_class' ] = 1 # CR and TN
    return df
# def declare_tc(df:pd.DataFrame):#四分类
#     df['target_class'] = 0 # CN
#     df.loc[(df.treatment == 0) & (df.target != 0),'target_class'] = 1 # CR
#     df.loc[(df.treatment != 0) & (df.target == 0),'target_class'] = 2 # TN
#     df.loc[(df.treatment != 0) & (df.target != 0),'target_class'] = 3 # TR
#     return df
df_model_bogo = declare_tc(df_model_bogo)
df_model_discount = declare_tc(df_model_discount)

### 模型构建
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
#划分数据集
def uplift_split(df_model:pd.DataFrame):
    ## 1 - Train-Test Split
    X = df_model.drop(['target','target_class'],axis=1) #自变量丢弃原始类别和分组类别
    y = df_model.target_class #因变量取分类变量
    X_train, X_test, y_train, y_test  = train_test_split(X, y,
                                       test_size=0.3,
                                       random_state=42,
                                       stratify=df_model['treatment'])#划分数据集后还需要将自变量的TC变量删去
    return X_train,X_test, y_train, y_test

#模型定义与result返回
def uplift_model(X_train: pd.DataFrame,
                 X_test: pd.DataFrame,
                 y_train: pd.DataFrame,
                 y_test: pd.DataFrame):
    ## 2 - Using XGB to get the uplift score
    # Create new dataframe
    result = pd.DataFrame(X_test).copy()
    # Fit the model 模型训练
    #提供了三种的方法XGB、Logistic、随机森林
    # uplift_model = xgb.XGBClassifier().fit(X_train.drop('treatment', axis=1), y_train)
    # uplift_model = LogisticRegression().fit(X_train.drop('treatment', axis=1), y_train)
    uplift_model = RandomForestClassifier().fit(X_train.drop('treatment', axis=1), y_train)

    #验证集拟合
    # Predict using test-data
    uplift_proba = uplift_model.predict_proba(X_test.drop('treatment', axis=1))
    # 取出两个定类变量的概率 目标客户概率大预测分类设置为1 否则为0
    result['proba_CN_TR'] = uplift_proba[:, 0]
    result['proba_CR_TN'] = uplift_proba[:, 1]
    # 将 'proba_CN_TR' 和 'proba_CR_TN' 大于 'predicted_class' 的值设为 1
    result['predicted_class'] = 0  # 初始化 'predicted_class' 列为 0
    result.loc[(result['proba_CN_TR'] > result['proba_CR_TN']), 'predicted_class'] = 1  # 将满足条件的值设为 1
    result['target_class'] = y_test
    return result

#切分数据 运行模型
def uplift(df_model: pd.DataFrame):
    # Combine the split and Modeling function
    X_train, X_test, y_train, y_test = uplift_split(df_model)
    result = uplift_model(X_train, X_test, y_train, y_test)
    return result

# Run the uplift function
bogo_uplift = uplift(df_model_bogo)
discount_uplift = uplift(df_model_discount)



### 模型评估
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, roc_auc_score

#绘制ROC图像
def plot_roc_curve(result, title):
    fpr, tpr, thresholds = roc_curve(result['target_class'], result['proba_CN_TR'])
    auc = roc_auc_score(result['target_class'], result['proba_CN_TR'])

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label='ROC Curve (AUC = {:.2f})'.format(auc))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title + ' - ROC Curve')
    plt.legend()
    plt.show()
def plot_confusion_matrix(conf_matrix, title):
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title(title + ' - Confusion Matrix')
    plt.show()
# 绘制混淆矩阵
def evaluate_model(result, title):
    # 计算准确率
    accuracy = accuracy_score(result['target_class'], result['predicted_class'])

    # 计算混淆矩阵
    conf_matrix = confusion_matrix(result['target_class'], result['predicted_class'])

    # 计算精确率、召回率、F1 分数
    precision = precision_score(result['target_class'], result['predicted_class'])
    recall = recall_score(result['target_class'], result['predicted_class'])
    f1 = f1_score(result['target_class'], result['predicted_class'])

    # 计算 ROC 曲线和 AUC 值
    fpr, tpr, thresholds = roc_curve(result['target_class'], result['proba_CN_TR'])
    auc = roc_auc_score(result['target_class'], result['proba_CN_TR'])

    # 输出结果
    print("准确率:", accuracy)
    print("混淆矩阵:")
    print(conf_matrix)
    print("精确率:", precision)
    print("召回率:", recall)
    print("F1 分数:", f1)
    print("AUC 值:", auc)
    #绘制ROC图像和混淆矩阵
    plot_roc_curve(result,'BOGO' )
    plot_confusion_matrix(confusion_matrix(result['target_class'], result['predicted_class']), title)

# 评估 BOGO 策略的 Uplift 模型
print("BOGO 策略的 Uplift 模型评估结果:")
evaluate_model(bogo_uplift,'BOGO')

# 评估折扣策略的 Uplift 模型
print("\ndiscount策略的 Uplift 模型评估结果:")
evaluate_model(discount_uplift,'Discount')

