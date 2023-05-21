#import required libraries and packages
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB

# import the neptune library
import neptune
import neptune.integrations.sklearn as npt_utils
from neptune.utils import stringify_unsupported #for logging unsupported objects

run = neptune.init_run(
    project="PtronusProject/PatronusWineDataset",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJmMmE2ZmRmMy0xMTkxLTRjYjMtYTUyOS1lMTBjOTRjOTJhODEifQ==",
)  # your credentials

#load and split the dataset
data=load_wine()
X=data.data
y=data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#load the model, specify the hyperparameters and fit the model on training data
model = RandomForestClassifier(n_estimators=10, max_depth=2)
model.fit(X_train, y_train)

#log the model name and the hyperparamters 
run['Model']=stringify_unsupported(model)
params={'n_estimators':10, 'maxdepth':2}
run['Parameters']=str(params)

#generate predictions
expected_y  = y_test
predicted_y = model.predict(X_test)

#log the evaluation results
report = metrics.classification_report(expected_y, predicted_y, target_names=data.target_names)
cm= metrics.confusion_matrix(expected_y, predicted_y)
run['Report']=report
run['ConfusionMatrix']=stringify_unsupported(cm)

run.stop()