
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
    test_df =pd.read_csv('./sample_test.csv')
    return test_df

test_df = get_data()

@st.cache_data
def model_threshold():
    fbm=pd.read_csv('./final_model.csv')
    threshold=fbm['Threshold'].values
    return threshold*100

threshold=model_threshold()



# Charger le modèle depuis MLflow (assurez-vous que votre modèle est déployé)
api_url = "http://127.0.0.1:8000/predict_proba"


st.title("Dashboard interactif pour la prédiction de crédit")
df=pd.read_csv('training_data.csv')

st.header("Estimation pour les Nouveaux clients")

date=st.date_input("today's date", format="DD.MM.YYYY" )

NAME_CONTRACT_TYPE= st.selectbox('Identification if loan is cash or revolving', df['NAME_CONTRACT_TYPE'].unique())

CODE_GENDER=st.selectbox('Client Gender', df['CODE_GENDER'].unique())


FLAG_OWN_CAR=st.selectbox('Flag if the client owns a car', df['FLAG_OWN_CAR'].unique())

FLAG_OWN_REALTY=st.selectbox('Flag if client owns a house or flat', df['FLAG_OWN_REALTY'].unique())

CNT_CHILDREN=st.number_input('Number of children the client has', min_value= 0, value=int(df['CNT_CHILDREN'].mode().iloc[0]), step=1)

AMT_INCOME_TOTAL=st.number_input('Income of the client', min_value= 0, value=int(df['AMT_INCOME_TOTAL'].mode().iloc[0]) , step=1)

AMT_CREDIT=st.number_input('Credit amount of the loan', min_value= 0, value=int(df['AMT_CREDIT'].mode().iloc[0]) , step=1)

NAME_TYPE_SUITE=st.selectbox('Who was accompanying client when he was applying for the loan', df['NAME_TYPE_SUITE'].unique())

NAME_INCOME_TYPE=st.selectbox('Clients income type (businessman, working, maternity leave,)', df['NAME_INCOME_TYPE'].unique())

NAME_EDUCATION_TYPE=st.selectbox('Level of highest education the client achieved', df['NAME_EDUCATION_TYPE'].unique())

NAME_FAMILY_STATUS=st.selectbox('Family status of the client', df['NAME_FAMILY_STATUS'].unique())

NAME_HOUSING_TYPE=st.selectbox('What is the housing situation of the client (renting, living with parents, ...)', df['NAME_HOUSING_TYPE'].unique())

date_birth=st.date_input('Date of birth',format="DD.MM.YYYY")
DAYS_BIRTH= abs(date - date_birth).days

date_employment=st.date_input('date of starting the current employment', format="DD.MM.YYYY")
DAYS_EMPLOYED=abs(date - date_employment).days

date_REGISTRATION=st.date_input('How many days before the application did client change his registration', format="DD.MM.YYYY")
DAYS_REGISTRATION=abs(date - date_REGISTRATION).days

date_ID_PUBLISH=st.date_input('How many days before the application did client change the identity document with which he applied for the loan', format="DD.MM.YYYY")
DAYS_ID_PUBLISH=abs(date - date_ID_PUBLISH).days

FLAG_MOBIL=st.number_input('Did client provide mobile phone (1=YES, 0=NO)', min_value= 0, value='min', step=1)

FLAG_CONT_MOBILE=st.number_input('Was mobile phone reachable (1=YES, 0=NO)', min_value= 0, value='min', step=1)

FLAG_EMAIL=st.number_input('Did client provide email (1=YES, 0=NO)', min_value= 0, value='min', step=1)

OCCUPATION_TYPE=st.selectbox('What kind of occupation does the client have', df['OCCUPATION_TYPE'].unique())

ORGANIZATION_TYPE=st.selectbox('Type of organization where client works', df['ORGANIZATION_TYPE'].unique())

LIVINGAREA_AVG=st.number_input('Living area  surface in m²', min_value= 0, value=int(df['LIVINGAREA_AVG'].mean()), step=1)

OBS_30_CNT_SOCIAL_CIRCLE=st.number_input("How many observation of client's social surroundings with observable 30 DPD (days past due) default", min_value= 0, value='min', step=1)

DEF_30_CNT_SOCIAL_CIRCLE=st.number_input("How many observation of client's social surroundings defaulted on 30 DPD (days past due) ", min_value= 0, value='min', step=1)

provided_flag_documents= st.number_input("How many flag document did client provid", min_value=0, value='min', max_value= 20, step=1)
AMT_REQ_CREDIT_BUREAU_SUM= st.number_input("Total enquiries number to Credit Bureau about the client ", min_value=0, value='min', step=1)


predict_btn=st.button('Nouveau résultat')
if predict_btn:
    input_data = {
        'NAME_CONTRACT_TYPE': NAME_CONTRACT_TYPE,
        'CODE_GENDER': CODE_GENDER,
        'FLAG_OWN_CAR': FLAG_OWN_CAR,
        'FLAG_OWN_REALTY': FLAG_OWN_REALTY,
        'CNT_CHILDREN': CNT_CHILDREN,
        'AMT_INCOME_TOTAL': AMT_INCOME_TOTAL,
        'AMT_CREDIT': AMT_CREDIT,
        'NAME_TYPE_SUITE': NAME_TYPE_SUITE,
        'NAME_INCOME_TYPE': NAME_INCOME_TYPE,
        'NAME_EDUCATION_TYPE': NAME_EDUCATION_TYPE,
        'NAME_FAMILY_STATUS': NAME_FAMILY_STATUS,
        'NAME_HOUSING_TYPE': NAME_HOUSING_TYPE,
        'DAYS_BIRTH': DAYS_BIRTH,
        'DAYS_EMPLOYED': DAYS_EMPLOYED,
        'DAYS_REGISTRATION': DAYS_REGISTRATION,
        'DAYS_ID_PUBLISH': DAYS_ID_PUBLISH,
        'FLAG_MOBIL': FLAG_MOBIL,
        'FLAG_CONT_MOBILE': FLAG_CONT_MOBILE,
        'FLAG_EMAIL': FLAG_EMAIL,
        'OCCUPATION_TYPE': OCCUPATION_TYPE,
        'ORGANIZATION_TYPE': ORGANIZATION_TYPE,
        'LIVINGAREA_AVG': LIVINGAREA_AVG,
        'OBS_30_CNT_SOCIAL_CIRCLE': OBS_30_CNT_SOCIAL_CIRCLE,
        'DEF_30_CNT_SOCIAL_CIRCLE': DEF_30_CNT_SOCIAL_CIRCLE,
        'provided_flag_documents': provided_flag_documents,
        'AMT_REQ_CREDIT_BUREAU_SUM': AMT_REQ_CREDIT_BUREAU_SUM
    }

    data = pd.DataFrame([input_data])
    scaled_data=data_processing(data, "test").to_dict(orient='records')[0]
    #scaled_data=scaled_data.values.tolist()[0]
    #st.write(f"scaled_data: {scaled_data}")
    
    

    response = requests.post(api_url, json={"data":list(scaled_data.values()), "keys":list(scaled_data.keys() )})

    # Afficher les résultats
    if response.status_code == 200:
        result = response.json()
        #st.write(f"le resultat est {result}")
        prediction=float(result["prediction"][0])
        if  prediction> threshold:
            st.warning(f"Il y a une probabilité {prediction:.2f} % que le créditeur rencontre des difficultés de paiement.\n La demande de crédit ne peut pas etre approuvée") 
            shap_plot_base64 = result['shap_plot']
            shap_image = Image.open(io.BytesIO(base64.b64decode(shap_plot_base64.split(',')[1])))
            st.image(shap_image, caption='SHAP Summary Plot', use_column_width='auto')
        else: 
            st.success(f"Il y a une probabilité de {prediction:.2f} % que le créditeur puisse rencontrer des difficultés de paiement. La demande de crédit peut être approuvée.")
            shap_plot_base64 = result['shap_plot']
            shap_image = Image.open(io.BytesIO(base64.b64decode(shap_plot_base64.split(',')[1])))
            st.image(shap_image, caption='SHAP Summary Plot')
    else:
        st.error(f"Erreur lors de la requête à l'API. Code de statut : {response.status_code}")


with st.sidebar: #avec des données test deja traitées

    st.subheader("Estimation pour les clients déja inscrits")

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
                #st.write(f"le resultat est {result}")
                prediction=float(result["prediction"][0])
                if  prediction> threshold:
                    st.warning(f"Il y a une probabilité {prediction:.2f} % que le créditeur rencontre des difficultés de paiement.\n La demande de crédit ne peut pas etre approuvée") 
                    shap_plot_base64 = result['shap_plot']
                    shap_image = Image.open(io.BytesIO(base64.b64decode(shap_plot_base64.split(',')[1])))
                    st.image(shap_image, caption='SHAP Summary Plot')
                else: 
                    st.success(f"Il y a une probabilité de {prediction:.2f} % que le créditeur puisse rencontrer des difficultés de paiement. La demande de crédit peut être approuvée.")
                    
            else:
                st.error(f"Erreur lors de la requête à l'API. Code de statut : {response.status_code}")
        else:
            st.warning("L'identifiant de crédit fourni n'est pas valide.")


