# Core libraries
import numpy as np
import pandas as pd

# Visualisation
from matplotlib import pyplot as plt

# Datasets
from sklearn.datasets import load_breast_cancer, make_classification

# Data splitting and validation
from sklearn.model_selection import (
  train_test_split,
  cross_val_score,
  RepeatedStratifiedKFold
)

# Pre-processing
from sklearn.preprocessing import LabelEncoder

#Evaluation
from sklearn.metrics import accuracy_score, f1_score

# Base learners
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

#Ensemble methods
from sklearn.ensemble import (
  AdaBoostClassifier,
  RandomForestClassifier,
  BaggingClassifier,
  VotingClassifier,
)

# Boosting with AdaBoots

# Load dataset
breast_cancer = load_breast_cancer()

# Create feature DataFrame
X = pd.DataFrame(
  breast_cancer.data,
  columns=breast_cancer.feature_names
)

# Create categorical target
y = pd.Categorical.from_codes(
  breast_cancer.target,
  breast_cancer.target_names
)

# Convert target labels to integers
encoder = LabelEncoder()
y = encoder.fit_transform(y)

print("Dataset shape:",X.shape)
print("Target classes:",encoder.classes_)

# Create train and test sets
X_train, X_test, y_train, y_test = train_test_split(
X,
 y,
 test_size=0.25,
 random_state=1,
 stratify=y #it helps maintain approximately the same class proportions in the training and test sets
)

print("Training observations:",X_train.shape[0])
print("Testing observations:",X_test.shape[0])

#Build the AdaBoost model
max_depth=1
boosting_model = AdaBoostClassifier(
 estimator=DecisionTreeCLassifier(
  max_depth=1
  random_state=1
 ),
 n_stimators=200
 random_state=1
)

boosting_model.fit(