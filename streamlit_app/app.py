import streamlit as st
import shap
import sys
sys.path.append(".")
import pandas as pd
from utils.model import ChurnModel 
import matplotlib.pyplot as plt
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

# create the churn model object
churn_model = ChurnModel()
df = churn_model.get_dataset(size=200)
X, y = df.drop(columns=["Exited"]), df["Exited"]

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

single_employer = user_input_features(churn_model.get_categ_features(), churn_model.get_numerical_features())
st.header("Selected Sample")
st.dataframe(single_employer, use_container_width=True)

# standardize and map the categorical values
model_trained = churn_model.load_latest_model()
predict_churn = model_trained.predict(single_employer) 

st.header("Prediction of the single employer")
st.write(predict_churn)
st.write("---")

# Explaining the model's predictions using SHAP values
explainer = shap.Explainer(model_trained.predict_proba, X)
print(single_employer.info())
print(single_employer.head())
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