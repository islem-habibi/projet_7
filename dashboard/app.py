import plotly.graph_objects as go
import streamlit as st
import mlflow
import pandas as pd
import requests
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler
from data_processing_module import data_processing
import shap
from PIL import Image
import io
import base64


@st.cache_data
def get_data():
    test_df =pd.read_csv('./scaled_sample_test.csv')
    return test_df

test_df = get_data()

@st.cache_data
def model_threshold():
    fbm=pd.read_csv('./final_model.csv')
    threshold=fbm['Threshold'].values
    return threshold*100

threshold=model_threshold()



# Charger le modèle depuis MLflow (assurez-vous que votre modèle est déployé)
api_url = "https://p7-api-40a72eb22034.herokuapp.com/predict_proba"


st.title("Dashboard interactif pour la prédiction de crédit")
df=pd.read_csv('training_data.csv')


st.subheader("Estimation pour les clients inscrits")

# Saisie de l'identifiant de crédit
ID = st.text_input("Identifiant de crédit: tester avec 208550 et 144092")
# use id 208550 for test
if st.button("Résultat"):
    if int(ID) in test_df['SK_ID_CURR'].values:
        # Sélectionner la ligne correspondant à l'identifiant fourni
        case = test_df.loc[test_df['SK_ID_CURR'] == int(ID), test_df.drop(columns=['SK_ID_CURR']).columns]
        # case_records = case.values.tolist()[0]#.to_dict(orient='records')[0]
        case_records = case.to_dict(orient='records')

        # Faire une requête à l'API déployée
        response = requests.post(api_url, json={"data":list(case_records[0].values()), "keys":list(case_records[0].keys() )})

        # Afficher les résultats
        if response.status_code == 200:
            result = response.json()
            
            prediction=float(result["prediction"][0])
            if  prediction> threshold:
                st.warning(f"Il y a une probabilité {prediction:.2f} % que le créditeur rencontre des difficultés de paiement.\n La demande de crédit ne peut pas etre approuvée") 
                shap_plot_base64 = result['shap_plot']
                shap_image = Image.open(io.BytesIO(base64.b64decode(shap_plot_base64.split(',')[1])))
                st.image(shap_image, caption='SHAP Summary Plot')
            else: 
                st.success(f"Il y a une probabilité de {prediction:.2f} % que le créditeur puisse rencontrer des difficultés de paiement. La demande de crédit peut être approuvée.")

            #result plot
            fig = go.Figure(go.Indicator(
            domain={'x': [0, 1], 'y': [0, 1]},
            value=prediction,
            mode="gauge+number",
            title={'text': "Jauge de score"},
            delta={'reference': 50},
            gauge={'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "black"},
                'bar': {'color': "MidnightBlue"},
                'steps': [
                    {'range': [0, 20], 'color': "Green"},
                    {'range': [20, 40], 'color': "LimeGreen"},
                    {'range': [40, 50], 'color': "Orange"},
                    {'range': [50, 100], 'color': "Red"}],
                'threshold': {'line': {'color': "brown", 'width': 4}, 'thickness': 1, 'value': 50}}))

            st.plotly_chart(fig)

            #income plot

            df0=pd.read_csv("sample_test.csv") #télechargement des données non standardisé
            income=df0.loc[df0['SK_ID_CURR']==int(ID), "AMT_INCOME_TOTAL"]
            trace_ref = go.Histogram(
            x=df['AMT_INCOME_TOTAL'],
            name='Revenus de référence',
            opacity=0.75
            )
            
            trace_client = go.Scatter(
                x=income,  
                y=[0],
                mode='markers',
                marker=dict(color='red', size=10),
                name='Revenu du client'
            )
            
            fig1 = go.Figure(data=[trace_ref, trace_client])
            fig1.update_layout(
                title='Comparaison du revenu du client avec les revenus de référence',
                xaxis_title='Revenu total',
                yaxis_title='Nombre de références',
                bargap=0.2,
            )
            st.plotly_chart(fig1)



                
        else:
            st.error(f"Erreur lors de la requête à l'API. Code de statut : {response.status_code}")
    else:
        st.warning("L'identifiant de crédit fourni n'est pas valide.")


