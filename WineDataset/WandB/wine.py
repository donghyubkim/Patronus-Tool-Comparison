import wandb
import random


from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB


# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="PatronusWineDataset",
    
    # track hyperparameters and run metadata
    config={
    'model':'RandomForestClassifier',
    'n_estimators':10, 
    'maxdepth':2,
    }
)

#load and split the dataset
data=load_wine()
X=data.data
y=data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#load the model, specify the hyperparameters and fit the model on training data
model = RandomForestClassifier(n_estimators=10, max_depth=2)
model.fit(X_train, y_train)

#generate predictions
expected_y  = y_test
predicted_y = model.predict(X_test)

#log the evaluation results
report = metrics.classification_report(expected_y, predicted_y, target_names=data.target_names)
cm= metrics.confusion_matrix(expected_y, predicted_y)

wandb.log({"report": report, "ConfusionMatric": cm})
    
# [optional] finish the wandb run, necessary in notebooks
wandb.finish()