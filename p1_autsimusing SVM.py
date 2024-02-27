'''basic libraries'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

'''gather the data'''
df = pd.read_csv("D:\\TOP MENTOR\\resume projects\\autsim_SVM\\dataset\\Toddler_Autism_dataset_July_2018.csv")
print(df)
print("shape of dataset is ", df.shape)

'''checking null values'''
print(df.isnull().sum())
df = df.rename(columns={"Class/ASD Traits ": "ASD"})

'''creating feature and target variables'''
x = df.drop('Case_No', axis=1)
x = x.drop('ASD', axis=1)
y = df['ASD']

'''labelencoder to cahnge string into int'''
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y = le.fit_transform(y)

'''we do ordinal encoding to feature variables for that we can first
split the object cols and number cols'''
obj_cols = x.select_dtypes(include='object').columns
float_cols = x.select_dtypes(include='int64').columns

from sklearn.preprocessing import OrdinalEncoder

oe = OrdinalEncoder(categories=[x[i].unique() for i in obj_cols])
oe.fit(x[obj_cols])
x__obj_encoded = oe.transform(x[obj_cols])

'''scalar function on integer variables'''
from sklearn.preprocessing import MinMaxScaler

scalar = MinMaxScaler()
scalar.fit(x[float_cols])
x_float_encoded = scalar.transform(x[float_cols])

x_processed = np.hstack((x__obj_encoded, x_float_encoded))
features = np.concatenate([obj_cols, float_cols])

x = pd.DataFrame(x_processed, columns=features)

'''trying normal logistic regression'''
'''train test split'''
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=22)

'''model training'''
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(x_train, y_train)
'''predicting the model'''
y_pred = model.predict(x_test)
res = pd.DataFrame({"actual": y_test, "predicted": y_pred})
print(res)
'''checking accuracy'''
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

print("accuracy score is ", accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

'''trying with SVM'''
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

# param_grid = {"kernel": ['linear'],
#               "C": [i for i in np.arange(0.1, 10, 0.1)],
#               "gamma": [i for i in np.arange(0.1, 10, 0.1)],
#               "degree": [i for i in range(2, 10)]}
# Grid_search = GridSearchCV(SVC(), param_grid, cv=5)
# Grid_search.fit(x_train, y_train)
# best_param = Grid_search.best_params_
# best_model = Grid_search.best_estimator_
# print(best_param)
model1=SVC(kernel='linear',C=0.5,gamma=0.5)
model1.fit(x_train,y_train)
y_linear_pred=model1.predict(x_test)
print("accuracy with linear SVC is",accuracy_score(y_test,y_linear_pred))

model1=SVC(kernel='poly',degree=2)
model1.fit(x_train,y_train)
y_poly_pred=model1.predict(x_test)
print("accuracy with polynomial SVC is",accuracy_score(y_test,y_poly_pred))

model1=SVC(kernel='rbf',gamma=0.7,C=0.7)
model1.fit(x_train,y_train)
y_rbf_pred=model1.predict(x_test)
print("accuracy with rbf SVC is",accuracy_score(y_test,y_rbf_pred))

