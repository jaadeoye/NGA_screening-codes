import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import seaborn as sns
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN
from imblearn.combine import SMOTEENN
from imblearn.combine import SMOTETomek
from sklearn.metrics import roc_auc_score
from sklearn.metrics import brier_score_loss
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_curve
import shap
from sklearn import metrics
from tabpfn import TabPFNClassifier
shap.initjs()
from sklearn.model_selection import KFold

#import data
from sklearn.preprocessing import StandardScaler
features = ['Smoking History', 'Tobacco snuff', 'Alcohol', 'Fruits', 'Veg', 'Red meat', 'Spicy food', 'CCI']
df_train = pd.read_excel('/Users/jaadeoye/Desktop/NGA_data/train.xlsx')
df_test = pd.read_excel('/Users/jaadeoye/Desktop/NGA_data/lagos_data.xlsx')
x = df_train[features]
y = df_train.O1
a = df_test[features]
b = df_test.O1
c=df_test.Category

kf = KFold(n_splits=10)

#class imbalance + CV
for fold, (train_index, test_index) in enumerate(kf.split(x,y)):
    X_train=x.iloc[train_index]
    y_train = y[train_index]
    X_test = x.iloc[test_index]
    y_test = y[test_index]
    sm = SMOTEENN(random_state = 0)
    X_train_oversampled, y_train_oversampled = sm.fit_resample(X_train, y_train)
    model=LogisticRegression(random_state=123, class_weight='balanced')
    model.fit(X_train_oversampled, y_train_oversampled)
    y_pred = model.predict(X_test)
    print(f'For fold {fold}:')
    print(f'AUC: {roc_auc_score(y_test, y_pred)}')

#train model
sm = SMOTEENN(random_state=0)
x_res, y_res = sm.fit_resample(x,y)
logreg =  LogisticRegression(random_state=123, class_weight='balanced')
logreg.fit(x_res,y_res)

#Prediction
pred=logreg.predict(a)
pred1=logreg.predict_proba(a)[:, 1]
pred2=logreg.predict_proba

#Platt scaling
y_cal=df_test.O1
x_cal=df_test.Predictions
lr = LogisticRegression()                                                       
lr.fit( x_cal.values.reshape( -1, 1 ), y_cal)     # LR needs X to be 2-dimensional
p_calibrated = lr.predict_proba(pred1.reshape( -1, 1 ))[:,1]

#confusion matrix
cnf_matrix = metrics.confusion_matrix(b, pred)
cnf_matrix

#ROC
ns_probs = [0 for _ in range(len(b))]
ns_fpr, ns_tpr, _ = roc_curve(b, ns_probs)
n_fpr, n_tpr, _ = roc_curve(b, pred1)
n1_fpr, n1_tpr, _ = roc_curve(b, c)
plt.plot(ns_fpr, ns_tpr, linestyle='--')
plt.plot(n_fpr, n_tpr, label='LR Model')
plt.plot(n1_fpr, n1_tpr, label='Crude method')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")


#print
newb = b.to_numpy()
newc = c.to_numpy()
roc = roc_auc_score(newb, pred1)
print("Accuracy:",metrics.accuracy_score(b,pred))
print("Recall:",metrics.recall_score(b,pred))
print("Precision:",metrics.precision_score(b,pred))
print("F1 score:",metrics.f1_score(b,pred))
print("BRIER:",metrics.brier_score_loss(b,pred1))
print("BRIER_CAL:",metrics.brier_score_loss(b,p_calibrated))
print(cnf_matrix)
print(roc)
roc1 = roc_auc_score(newb, newc)

#Shapley values
explainer = shap.Explainer(pred2, x_res)
shap_test = explainer(a)
print(f"Shap values length: {len(shap_test)}\n")
print(f"Sample shap value:\n{shap_test[0]}")
shap_df = pd.DataFrame(shap_test.values[:,:,1], 
                       columns=shap_test.feature_names, 
                       index=a.index)
shap.initjs()
shap_df
shap.plots.bar(shap_test[:,:,1])
shap.plots.beeswarm(shap_test[:,:,1])
shap.summary_plot(shap_test[:,:,1], max_display=8)
#shap.plots.force(explainer.expected_value, shap_test.values[0, :], a.iloc[0, :], show=False, matplotlib=True)
plt.show()



