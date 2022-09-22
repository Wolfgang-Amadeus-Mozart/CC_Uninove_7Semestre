#!/usr/bin/env python3 
#-*- coding? utf-8 -*- 
#-------------------------------------------------------------------------------------
# Created By : Filipe Queiroz de Abreu / Bacherelando em Ciência da computação Uninove
# Created Date : 15/09/2022
# Version = 1.0
#-------------------------------------------------------------------------------------

import joblib 
import pandas as pd

def pipeline_(df):

    # Pipeline de transformação das variáveis 
    df['voice_mail_plan'] = df['voice_mail_plan'].astype('category')
    df['voice_mail_plan'] = df['voice_mail_plan'].cat.codes
    df['international_plan'] = df['international_plan'].astype('category')
    df['international_plan'] = df['international_plan'].cat.codes
    df = pd.get_dummies(df, columns=['area_code'])
    df['state'] = df['state'].astype('category')
    df['state'] = df['state'].cat.codes

    # Carrega o modelo treinado
    model_trained = joblib.load_model('rf_model_churn.pkl')

    return model_trained.predict(df), model_trained.predict_proba(df[:,1])