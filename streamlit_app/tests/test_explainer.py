import shap
import sys
sys.path.append(".")
import pandas as pd
from utils.model import ChurnModel 
import matplotlib.pyplot as plt
import numpy as np

# create the churn model object
churn_model = ChurnModel()

model_trained = churn_model.load_latest_model()
df = churn_model.get_dataset(size=200)
X, y = df.drop(columns=["Exited"]), df["Exited"]

print(X.info())
print(X.head())
print(X.isna().sum())

# sample data
data = {'CreditScore': ["43743"],
        'Geography': ["Spain"],
        'Gender': ["Male"],
        'Age': ["34"],
        'Tenure': ["13"],
        'Balance': ["342"],
        'NumOfProducts': ["4"],
        'HasCrCard': ["1"],
        'IsActiveMember': ["1"],
        'EstimatedSalary': ["384972.0"]
}

features = pd.DataFrame(data)
categ_lst, numerical_cols = churn_model.get_categ_features(), churn_model.get_numerical_features()
features[categ_lst] = features[categ_lst].astype("string")
features[numerical_cols] = features[numerical_cols].astype("float")

# TEST PREDICTION
print(features.head())
print(f"The prediction of this sample is: {model_trained.predict(features)}")

# TEST EXPLAINER
explainer = shap.Explainer(model_trained.predict_proba, X)
transformed = model_trained["preprocessor"].transform(features)
transformed = pd.DataFrame(transformed, columns=df.drop(columns=["Exited"]).columns, dtype=float)

print(transformed)
shap_values = explainer(transformed)
shap.plots.waterfall(shap_values[0,:, 1], max_display = 10)
plt.show()