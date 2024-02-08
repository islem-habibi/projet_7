import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler


def data_processing(test_df, argument):
    train_df = pd.read_csv('training_data.csv')
    target = train_df['TARGET']
    train_df = train_df.drop(columns='TARGET')

    # Sélection des colonnes de type 'object' et comptage du nombre de catégories uniques
    cat2_col = train_df.select_dtypes('object').loc[:, list(train_df.select_dtypes('object').nunique() == 2)]

    # Label encoding pour les variables avec 2 catégories différentes
    label = LabelEncoder()
    for col in cat2_col.columns:
        train_df[col] = label.fit_transform(train_df[col])
        test_df[col] = label.transform(test_df[col])

       # Sélection des colonnes catégorielles
    cat_col = train_df.select_dtypes('object').columns

    # Encodage one-hot pour les variables avec plus de 2 catégories
    encoder = OneHotEncoder(sparse=False)  # Ajout de sparse=False pour obtenir un tableau NumPy dense
    train_encoded = encoder.fit_transform(train_df[cat_col])
    columns = encoder.get_feature_names_out(cat_col)
    train_encoded_df = pd.DataFrame(train_encoded, columns=columns)
    train_df = pd.concat([train_df.drop(columns=cat_col), train_encoded_df], axis=1)

    test_encoded = encoder.transform(test_df[cat_col])
    test_encoded_df = pd.DataFrame(test_encoded, columns=columns)
    test_df = pd.concat([test_df.drop(columns=cat_col), test_encoded_df], axis=1)


    # Normalisation des données avec MinMaxScaler
    scaler = MinMaxScaler()
    cols_train = train_df.drop(columns='SK_ID_CURR').columns
    train_id = train_df['SK_ID_CURR']
    scaled_train_df = pd.DataFrame(scaler.fit_transform(train_df.drop(columns='SK_ID_CURR')), columns=cols_train)
    scaled_train_df['TARGET'] = target
    scaled_train_df['SK_ID_CURR'] = train_id

    if 'SK_ID_CURR' in test_df.columns:
        test_id = test_df['SK_ID_CURR']
        scaled_test_df = pd.DataFrame(scaler.transform(test_df.drop(columns='SK_ID_CURR')), columns=cols_train)
        scaled_test_df['SK_ID_CURR'] = test_id
    else:
        scaled_test_df = pd.DataFrame(scaler.transform(test_df), columns=cols_train)
        #scaled_testing_df = scaled_testing_df.set_index(test_id)  # Cette ligne semble inutile, car test_id n'est pas défini ici

    if argument == "train":
        return scaled_train_df
    elif argument == "test":
        return scaled_test_df
    elif argument == "both":
        return scaled_train_df, scaled_test_df


