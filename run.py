import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
import sys

print("PLEASE wait 3 mins approx")
dfd=pd.read_csv("train_data.csv")

new_dfd=dfd.drop("uuid", axis="columns")

rfr_dfd=dfd.drop(["uuid", "datasetId"], axis=1)

X = rfr_dfd.drop('HR', axis=1)
y = rfr_dfd['HR']
# Identifying categorial features
numerical_features = X.select_dtypes(include=['float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

#Using to transform different types of features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

# Creating a Random Forest Regressor model
model = make_pipeline(preprocessor, RandomForestRegressor(n_estimators=200, random_state=42))

# Train the model on the entire dataset to later test on test data provided
model.fit(X, y)

y_pred = model.predict(X)

mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

print(f'Mean Squared Error: {mse}') #printing mse
print(f'R-squared: {r2}')

n = len(sys.argv)

if(n>2 or n<0):
    print("Enter valid number of arguments")
    exit()

testdd=pd.read_csv(sys.argv[1])
uuidcol=testdd["uuid"]
new_data = testdd.drop(['uuid', 'datasetId'], axis=1)
new_predictions = model.predict(new_data)

result_df = pd.DataFrame({'uuid': uuidcol, 'HR': new_predictions})
result_df.to_csv('results.csv', index=False)

