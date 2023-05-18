# AIbolit
AIbolit is a project that aims to bring together machine learning methods for disease prediction.

#AIbolit is the name of the project, SMOTE is Synthetic Minority Oversampling Technique 
#for working with unbalanced data

#Several datasets have been collected here for diseases such as lung cancer,
#stroke and stroke after oncopathology
#I will try to describe each step, let's go!

from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.metrics import classification_report, confusion_matrix

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from statistics import mean
from sklearn.utils import shuffle
import seaborn as sns
from matplotlib import pyplot as plt
import os,sys

#CANCER & STROKE
#Let's start with predictions of stroke and oncopathology
#All patients from the sick_patients sample have a documented stroke after oncopathology
healthy_patients = pd.read_excel('../input/covid19/dataset.xlsx',engine="openpyxl")
sick_patients = pd.read_excel('/kaggle/input/medicine-data/Stroke and Canser predict.xlsx',engine="openpyxl")

#We take only those parameters that are contained in the sick_patients dataset
healthy_patients = healthy_patients[['Hemoglobin', 'Hematocrit', 'Leukocytes', 'Red blood Cells', 'Mean corpuscular volume (MCV)', 'Red blood cell distribution width (RDW)', 'Platelets']]
healthy_patients = healthy_patients.dropna ()

#A little preprocessing
sick_patients = sick_patients.round(2)
sick_patients = sick_patients.fillna(sick_patients.mean())
sick_patients = (sick_patients-sick_patients.mean())/sick_patients.std()
sick_patients['Strock'] = 1

#Let's see how our parameters look on the correlation matrix
best_features_ever = ['Hemoglobin', 'Leukocytes', 'Mean corpuscular volume (MCV)', 'Red blood cell distribution width (RDW)', 'Platelets']
corr = sick_patients.loc[:, best_features_ever].corr()

mask = np.zeros_like(corr, dtype=bool)
mask[np.triu_indices_from(mask)] = True
data, ax = plt.subplots(figsize=(8, 8))
plt.title('Correlation plot')
cmap = sns.diverging_palette(260, 10, as_cmap=True)

sns.heatmap(corr, vmax=1.2, square='square', cmap=cmap, mask=mask, 
            ax=ax, annot=True, fmt='.2g',linewidths=2);
            
healthy_patients['Strock'] = 0
sick_patients = sick_patients.round(5)
healthy_patients = healthy_patients.round(5)

#Combining datasets for training
data = pd.concat([healthy_patients, sick_patients], axis=0)
data = data[['Hemoglobin', 'Leukocytes', 'Mean corpuscular volume (MCV)',
             'Red blood cell distribution width (RDW)', 'Platelets', 'Strock']]
data = shuffle(data)

X = data.drop(['Strock'], axis=1)
y = data['Strock']

#Use SMOTE to oversample the minority class
oversample = SMOTE()
over_X, over_y = oversample.fit_resample(X, y)
over_X_train, over_X_test, over_y_train, over_y_test = train_test_split(over_X, over_y, test_size=0.1, stratify=over_y)
#Build SMOTE SRF model
SMOTE_SRF = RandomForestClassifier(n_estimators=160)
#Create Stratified K-fold cross validation
c = RepeatedStratifiedKFold(n_splits=10, n_repeats=3)
s = ('f1', 'recall', 'precision')

#Evaluate SMOTE SRF model
scoring = cross_validate(SMOTE_SRF, over_X, over_y, scoring=s, cv=c)
#Get average evaluation metrics
print('Mean f1: %.3f' % mean(scoring['test_f1']))
print('Mean recall: %.3f' % mean(scoring['test_recall']))
print('Mean precision: %.3f' % mean(scoring['test_precision']))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y)
SMOTE_SRF.fit(over_X_train, over_y_train)

y_pred = SMOTE_SRF.predict(X_test)
y_pred[0]

#I prefer _proba because you can't say sick person or not without a doctor's appointment
y_pred_proba = SMOTE_SRF.predict_proba(X_test)
y_pred_proba[0]

#Create confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=SMOTE_SRF.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=SMOTE_SRF.classes_)
disp.plot()
plt.show()

#LUNG
#Let's continue working with lung cancer and do everything by analogy
lung_data = pd.read_csv("/kaggle/input/lung-cancer/survey lung cancer.csv")

#There are always fewer sick patients than healthy ones
lung_data['LUNG_CANCER'].value_counts()

encoder = LabelEncoder()
lung_data['LUNG_CANCER']=encoder.fit_transform(lung_data['LUNG_CANCER'])
lung_data['GENDER']=encoder.fit_transform(lung_data['GENDER'])
lung_data = lung_data[['GENDER', 'AGE', 'SMOKING', 'CHRONIC DISEASE', 'ALLERGY ', 'WHEEZING',
       'ALCOHOL CONSUMING', 'COUGHING', 'SHORTNESS OF BREATH',
       'SWALLOWING DIFFICULTY', 'CHEST PAIN', 'LUNG_CANCER']]

lung_data.drop_duplicates(inplace=True)

X_lung = lung_data.drop(['LUNG_CANCER'], axis=1)
y_lung = lung_data['LUNG_CANCER']

oversample_lung = SMOTE()
over_X_lung, over_y_lung = oversample_lung.fit_resample(X_lung, y_lung)
over_X_train_lung, over_X_test_lung, over_y_train_lung, over_y_test_lung = train_test_split(over_X_lung, over_y_lung, test_size=0.1, stratify=over_y_lung)
SMOTE_SRF_lung = RandomForestClassifier(n_estimators=60)
cv_lung = RepeatedStratifiedKFold(n_splits=10, n_repeats=3)

scoring_lung = cross_validate(SMOTE_SRF_lung, over_X_lung, over_y_lung, scoring=s, cv=cv_lung)
print('Mean f1: %.3f' % mean(scoring_lung['test_f1']))
print('Mean recall: %.3f' % mean(scoring_lung['test_recall']))
print('Mean precision: %.3f' % mean(scoring_lung['test_precision']))

X_train_lung, X_test_lung, y_train_lung, y_test_lung = train_test_split(X_lung, y_lung, test_size=0.1, stratify=y_lung)
SMOTE_SRF.fit(over_X_train_lung, over_y_train_lung)
y_pred_lung = SMOTE_SRF.predict_proba(X_test_lung)

#STROKE
#Using SMOTE for the third dataset
stroke_data = pd.read_csv('/kaggle/input/stroke-prediction-dataset/healthcare-dataset-stroke-data.csv')

stroke_data = stroke_data[['gender','age', 'hypertension', 'heart_disease','avg_glucose_level','smoking_status','stroke']]
#Bring the data to a convenient form
stroke_data = stroke_data.replace("Female", "0")
stroke_data = stroke_data.replace("Male", "1")
stroke_data = stroke_data.replace("Other", "1")
stroke_data = stroke_data[stroke_data["smoking_status"].str.contains("Unknown")== False]
stroke_data['smoking_status'] = encoder.fit_transform(stroke_data['smoking_status'])

X_stroke = stroke_data.drop(['stroke'], axis=1)
y_stroke = stroke_data['stroke']

oversample_stroke = SMOTE()
over_X_stroke, over_y_stroke = oversample_stroke.fit_resample(X_stroke, y_stroke)
over_X_train_stroke, over_X_test_stroke, over_y_train_stroke, over_y_test_stroke = train_test_split(over_X_stroke, over_y_stroke, test_size=0.1, stratify=over_y_stroke)
SMOTE_SRF_stroke = RandomForestClassifier(n_estimators=150)
cv_stroke = RepeatedStratifiedKFold(n_splits=10, n_repeats=3)

scoring_stroke = cross_validate(SMOTE_SRF_stroke, over_X_stroke, over_y_stroke, scoring=s, cv=cv_stroke)
print('Mean f1: %.3f' % mean(scoring_stroke['test_f1']))
print('Mean recall: %.3f' % mean(scoring_stroke['test_recall']))
print('Mean precision: %.3f' % mean(scoring_stroke['test_precision']))

#SMOTE works great for medical datasets
X_train_stroke, X_test_stroke, y_train_stroke, y_test_stroke = train_test_split(X_stroke, y_stroke, test_size=0.1, stratify=y_stroke)
SMOTE_SRF_stroke.fit(over_X_train_stroke, over_y_train_stroke)
y_pred_stroke = SMOTE_SRF_stroke.predict_proba(X_test_stroke)
y_pred_stroke[0]


