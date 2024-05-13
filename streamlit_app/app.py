import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import shap
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from src.components import sidebar

st.set_page_config(page_title="ByeBank!", page_icon=":heavy_dollar_sign:")
st.set_option('deprecation.showPyplotGlobalUse', False)

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

# extract categorical and numericals features from the dataset
categ_lst = ["Gender", "Geography"]
numerical_cols = list(set(df.columns) - set(["Exited", "Gender", "Geography"]))
df[categ_lst] = df[categ_lst].astype("string")
df[numerical_cols] = df[numerical_cols].astype("float")
df['Exited'] = df['Exited'].astype("int")

# Removing every cell with nan values present and the duplicates
df = df.dropna()
df = df.drop_duplicates()

# Remove outliers
df = df[(np.abs(stats.zscore(df[numerical_cols])) < 3).all(axis=1)]

# subselect the dataset to be fast in explanation
df = df[:100]

# Define the model
model = LogisticRegression(random_state=42)
X, y = df.drop(columns=["Exited"]), df["Exited"]

# preprocessing
X.Geography = X.Geography.map({'France': 0, 'Germany': 1, 'Spain': 2})
X.Gender = X.Gender.map({'Female': 0, 'Male': 1})

sscaler = StandardScaler()
X[numerical_cols] = sscaler.fit_transform(X[numerical_cols])

# apply the model
model.fit(X, y)

# apply model to make prediction
def user_input_features(categ_lst, numerical_cols):
    CreditScore = st.sidebar.slider('CreditScore', df.CreditScore.min(), df.CreditScore.max(), df.CreditScore.mean())
    Geography = st.sidebar.radio('Geography', ["France", "Spain", "Germany"], index=1)
    Gender = st.sidebar.radio('Gender', ["Female", "Male"], index=0)
    Age = st.sidebar.slider('Age', df.Age.min(), df.Age.max(), df.Age.mean())
    Tenure = st.sidebar.text_input('Tenure', f"Insert a number - max: {round(df.Tenure.max()+df.Tenure.std(), 0)}")
    Balance = st.sidebar.slider('Balance', df.Balance.min(), df.Balance.max(), df.Balance.mean())
    NumOfProducts = st.sidebar.text_input('NumOfProducts', f"Insert a number - max: {round(df.NumOfProducts.max()+df.NumOfProducts.std(), 0)}")
    HasCrCard = st.sidebar.radio('HasCrCard', [0, 1], index=0)
    IsActiveMember = st.sidebar.radio('IsActiveMember', [0, 1], index=1)
    EstimatedSalary = st.sidebar.slider('EstimatedSalary', df.EstimatedSalary.min(), df.EstimatedSalary.max(), df.EstimatedSalary.mean())
    
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
    features[categ_lst] = features[categ_lst].astype("string")
    features[numerical_cols] = features[numerical_cols].astype("float")
    return features

single_employer = user_input_features(categ_lst, numerical_cols)
st.header("Selected Sample")
st.dataframe(single_employer, use_container_width=True)

# standardize and map the categorical values
single_employer.Geography = single_employer.Geography.map({'France': 0, 'Germany': 1, 'Spain': 2})
single_employer.Gender = single_employer.Gender.map({'Female': 0, 'Male': 1})
single_employer[categ_lst] = single_employer[categ_lst].astype("float")
single_employer[numerical_cols] = sscaler.transform(single_employer[numerical_cols])
predict_churn = model.predict(single_employer) 

st.header("Prediction of the single employer")
st.write(predict_churn)
st.write("---")

# Explaining the model's predictions using SHAP values
explainer = shap.Explainer(model.predict_proba, X)
st.dataframe(single_employer, use_container_width=True)
shap_values = explainer(single_employer)

st.header('Probability of Churning')
shap.plots.waterfall(shap_values[0,:,1], max_display = 10) 
st.pyplot(bbox_inches='tight')
plt.clf()
st.write('---')

st.header('Probability of Not Churning')
shap.plots.waterfall(shap_values[0,:,0], max_display = 30) # loan not accepted by the customer
st.pyplot(bbox_inches='tight')
plt.clf()