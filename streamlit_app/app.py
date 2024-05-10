import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import shap
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from src.components import sidebar

st.set_page_config(page_title="ByeBank!", page_icon=":heavy_dollar_sign:")

sidebar.show_sidebar()

st.title('Bye Colleague! :wave:')
st.write("""
    is a web-app where we can see the power of Streamlit!
    
    Here we trained behind this UI a simple model to predict the churn of employers of a bank.
    We leave the possibility to the user to get a single employer prediction based on changed values. 
""")

st.write('---')

# Loads the churn dataset
df_ = pd.read_csv("./streamlit_app/data/churn_dataset.csv", sep=',', on_bad_lines='skip', index_col=False, dtype='unicode')
df = df_.drop(columns=["RowNumber", "CustomerId", "Surname"])

# Removing every cell with nan values present and the duplicates
df = df.dropna()
df = df.drop_duplicates()

# Apply the schema types
numerical_cols= ['CreditScore', 'Age', 'Tenure', 'Balance','NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']
categ_lst = ["Geography", "Gender"] # encode categorical value
df[categ_lst] = df[categ_lst].astype("string")
df[numerical_cols] = df[numerical_cols].astype("float")
df['Exited'] = df['Exited'].astype("int")

# Remove outliers
df = df[(np.abs(stats.zscore(df[numerical_cols])) < 3).all(axis=1)]

# Define the pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categ_lst),
        ("num", StandardScaler(), numerical_cols)
    ]
)
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(random_state=42))
])

# split dataset and train directly
X, y = df.drop(columns=["Exited"]), df["Exited"]
model.fit(X, y)

# apply model to make prediction
def user_input_features():
    CreditScore = st.sidebar.slider('CreditScore', X.CreditScore.min(), X.CreditScore.max(), X.CreditScore.mean())
    Geography = st.sidebar.radio('Geography', ["France", "Spain", "Germany"], index=1)
    Gender = st.sidebar.radio('Gender', ["Female", "Male"], index=0)
    Age = st.sidebar.slider('Age', X.Age.min(), X.Age.max(), X.Age.mean())
    Tenure = st.sidebar.text_input('Tenure', f"Insert a number - max: {round(X.Tenure.max()+X.Tenure.std(), 0)}")
    Balance = st.sidebar.slider('Balance', X.Balance.min(), X.Balance.max(), X.Balance.mean())
    NumOfProducts = st.sidebar.text_input('NumOfProducts', f"Insert a number - max: {round(X.NumOfProducts.max()+X.NumOfProducts.std(), 0)}")
    HasCrCard = st.sidebar.radio('HasCrCard', [0, 1], index=0)
    IsActiveMember = st.sidebar.radio('IsActiveMember', [0, 1], index=1)
    EstimatedSalary = st.sidebar.slider('EstimatedSalary', X.EstimatedSalary.min(), X.EstimatedSalary.max(), X.EstimatedSalary.mean())
    
    data = {'CreditScore': CreditScore,
            'Geography': Geography,
            'Gender': Gender,
            'Age': round(Age, 0),
            'Tenure': float(Tenure.split(":")[1].strip()),
            'Balance': Balance,
            'NumOfProducts': float(NumOfProducts.split(":")[1].strip()),
            'HasCrCard': HasCrCard,
            'IsActiveMember': IsActiveMember,
            'EstimatedSalary': EstimatedSalary
    }
    features = pd.DataFrame(data, index=[0])
    return features

single_employer = user_input_features()
predict_churn = model.predict(single_employer) 

st.header("Prediction of the single employer")
st.write(predict_churn)
st.write("---")

# Explaining the model's predictions using SHAP values
# https://github.com/slundberg/shap
explainer = shap.Explainer(model.predict_proba, X)
shap_values = explainer.shap_values(X)

st.header('Feature Importance')
plt.title('Feature importance based on SHAP values')
shap.summary_plot(shap_values, X)
st.pyplot(bbox_inches='tight')
st.write('---')

plt.title('Feature importance based on SHAP values (Bar)')
shap.summary_plot(shap_values, X, plot_type="bar")
st.pyplot(bbox_inches='tight')