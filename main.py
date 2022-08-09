import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from IPython.display import Image, display
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import balanced_accuracy_score

xls = pd.ExcelFile('C:/Users/m997t/Desktop/112021.xlsx')
df1 = pd.read_excel(xls, 'Patient')
df2 = pd.read_excel(xls, 'Diagnosis')
print(df1)
print(df2)

#Generate synthetic dataset

X,Y = make_classification(n_samples=12868, n_classes=2, n_features=10, random_state=0)

##Add noisy features to make the problem more difficult

random_state = np.random.RandomState(0)
n_samples, n_features = X.shape
X = np.c_[X, random_state.randn(n_samples,200 * n_features)]
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=.2,random_state=0)

## Random Forest

rf = RandomForestClassifier(max_features=5, n_estimators=500)
rf.fit(X_train,Y_train)

##Naive Bayes

nb = GaussianNB()
nb.fit(X_train,Y_train)
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)

##Prediction probabilities

r_probs = [0 for _ in range(len(Y_test))]
rf_probs = rf.predict_proba(X_test)
nb_probs = nb.predict_proba(X_test)
knn_probs = knn.predict_proba(X_test)

rf_probs = rf_probs[:, 1]
nb_probs = nb_probs[:, 1]
knn_probs = knn_probs[:, 1]

##Computing AUROC and ROC curve values

r_auc = roc_auc_score(Y_test, r_probs)
rf_auc = roc_auc_score(Y_test, rf_probs)
nb_auc =roc_auc_score(Y_test, nb_probs)

print('Random (chance) Prediction: AUROC = %.3f' % (r_auc))
print('Random Forest: AUROC = %.3f' % (r_auc))
print('Naive Bayes: AUROC = %.3f' % (r_auc))

r_fpr, r_tpr, _ = roc_curve(Y_test, r_probs)
rf_fpr, rf_tpr, _ = roc_curve(Y_test, rf_probs)
nb_fpr, nb_tpr, _ = roc_curve(Y_test, nb_probs)

plt.plot(r_fpr, r_tpr, linestyle='--', label='Random prediction (AUROC = %0.3f)' % r_auc)
plt.plot(rf_fpr, rf_tpr, marker='.', label='Random Forest (AUROC = %0.3f)' % rf_auc)
plt.plot(nb_fpr, nb_tpr, marker='.', label='Naive Bayes (AUROC = %0.3f)' % nb_auc)

#Calculate AUROC

r_auc = roc_auc_score(Y_test, r_probs)
rf_auc = roc_auc_score(Y_test, rf_probs)
nb_auc = roc_auc_score(Y_test, nb_probs)
knn_auc = roc_auc_score(Y_test, knn_probs)

#Print AUROC scores
print('Random (chance) Prediction: AUROC = %.3f' % (r_auc))
print('Random Forest: AUROC = %.3f' % (rf_auc))
print('Naive Bayes: AUROC = %.3f' % (nb_auc))
print('K Nearest Neighbors: AUROC = %.3f' % (knn_auc))

#Calculate ROC curve
r_fpr, r_tpr, _ = roc_curve(Y_test, r_probs)
rf_fpr, rf_tpr, _ = roc_curve(Y_test, rf_probs)
nb_fpr, nb_tpr, _ = roc_curve(Y_test, nb_probs)
knn_fpr, knn_tpr, _ = roc_curve(Y_test, knn_probs)

nb_fpr, nb_tpr
knn_fpr, knn_tpr

# Title
plt.title('ROC Plot')
# Axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# Show legend
plt.legend()
## Show plot
plt.show()
plt.legend()
# Save plot to files

plt.savefig('roc.pdf')
plt.savefig('roc.png')



cm = metrics.confusion_matrix(Y_test, Y_pred)
# Assigning columns names
cm_df = pd.DataFrame(cm,
            columns = ['Predicted Negative', 'Predicted Positive'],
            index = ['Actual Negative', 'Actual Positive'])
# Showing the confusion matrix
cm_df

# The codes below is partly copied from the code written by Matt Brem, Global Instructor at General Assembly.
# Imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=42)
# Instantiating a logisitic regression model
logreg = LogisiticRegression()
logreg.fit(X_train, y_train)  # model fitting
y_pred = logreg.predict(X_test)  # Predictions
# Calculating class probabilities
pred_proba = [i[1] for i in logreg.predict_proba(X_test)]
pred_df = pd.DataFrame({'true_values': y_test,
                        'pred_probs': pred_proba})
# The codes below is motly copied from the code written by Matt Brem, Global Instructor at General Assembly.
# Create figure.
plt.figure(figsize=(10, 7))
# Create threshold values.
thresholds = np.linspace(0, 1, 200)


# Define function to calculate sensitivity. (True positive rate.)
def TPR(df, true_col, pred_prob_col, threshold):
    true_positive = df[(df[true_col] == 1) & (df[pred_prob_col] >= threshold)].shape[0]
    false_negative = df[(df[true_col] == 1) & (df[pred_prob_col] < threshold)].shape[0]
    return true_positive / (true_positive + false_negative)


# Define function to calculate 1 - specificity. (False positive rate.)
def FPR(df, true_col, pred_prob_col, threshold):
    true_negative = df[(df[true_col] == 0) & (df[pred_prob_col] <= threshold)].shape[0]
    false_positive = df[(df[true_col] == 0) & (df[pred_prob_col] > threshold)].shape[0]
    return 1 - (true_negative / (true_negative + false_positive))


# Calculate sensitivity & 1-specificity for each threshold between 0 and 1.
tpr_values = [TPR(pred_df, 'true_values', 'pred_probs', prob) for prob in thresholds]
fpr_values = [FPR(pred_df, 'true_values', 'pred_probs', prob) for prob in thresholds]
# Plot ROC curve.
plt.plot(fpr_values,  # False Positive Rate on X-axis
         tpr_values,  # True Positive Rate on Y-axis
         label='ROC Curve')
# Plot baseline. (Perfect overlap between the two populations.)
plt.plot(np.linspace(0, 1, 200),
         np.linspace(0, 1, 200),
         label='baseline',
         linestyle='--')
# Label axes.
plt.title(f"ROC Curve with AUC = {round(metrics.roc_auc_score(pred_df['true_values'], pred_df['pred_probs']), 3)}",
          fontsize=22)
plt.ylabel('Sensitivity', fontsize=18)
plt.xlabel('1 - Specificity', fontsize=18)
# Create legend.
plt.legend(fontsize=16);


# Creating a function to report confusion metrics
def confusion_metrics(conf_matrix):
    # save confusion matrix and slice into four pieces
    TP = conf_matrix[1][1]
    TN = conf_matrix[0][0]
    FP = conf_matrix[0][1]
    FN = conf_matrix[1][0]
    print('True Positives:', TP)
    print('True Negatives:', TN)
    print('False Positives:', FP)
    print('False Negatives:', FN)

    # calculate accuracy
    conf_accuracy = (float(TP + TN) / float(TP + TN + FP + FN))

    # calculate mis-classification
    conf_misclassification = 1 - conf_accuracy

    # calculate the sensitivity
    conf_sensitivity = (TP / float(TP + FN))
    # calculate the specificity
    conf_specificity = (TN / float(TN + FP))

    # calculate precision
    conf_precision = (TN / float(TN + FP))
    # calculate f_1 score
    conf_f1 = 2 * ((conf_precision * conf_sensitivity) / (conf_precision + conf_sensitivity))
    print('-' * 50)
    print(f'Accuracy: {round(conf_accuracy, 2)}')
    print(f'Mis-Classification: {round(conf_misclassification, 2)}')
    print(f'Sensitivity: {round(conf_sensitivity, 2)}')
    print(f'Specificity: {round(conf_specificity, 2)}')
    print(f'Precision: {round(conf_precision, 2)}')
    print(f'f_1 Score: {round(conf_f1, 2)}')

    bal_acc = balanced_accuracy_score(y_test, y_pred)

    Accuracy = (TP + TN) / (TP + FN + FP + TN)
    Sensitivity = TP / (TP + FN)
    Specificity = TN / (TN + FP)

    #Balanaced Accuracy

    Balanced Accuracy = (RecallP + RecallQ + RecallR + RecallS) / 4.

    For class Q, RecallQ

    For class R, RecallR

    For class S,RecallS




    Balanced Accuracy = (Sensitivity + Specificity) / 2
    Accuracy = TP + TN / (TP+FP+FN+TN)
    Recallp = TP / (TP + FN)
    Binary Accuracy: Sensitivity = TP / (TP + FN)
    Specificity = TN / (TN + FP)
    Binary Accuracy: Sensitivity + Specificity / 2

    #Balanced Accuracy vs F1 Score

    F1 = 2 * ([precision * recall] / [precision + recall])
    Balanced Accuracy = (specificity + recall) / 2
