# Data Loading
import pickle

# Data Manipulation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

#Random Forest Classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import accuracy_score
import math

from scipy.stats import mode

from domain import domain
from domain.domain import TransmodeRequest, TransmodeResponse

class TransmodeService:
    def __init__(self):
        # Load the pre-trained model and label encoders
        self.path_model = '.../artifacts/RandomForestTransportModePrediction.pkl'
        self.path_fisik_2_encoder = '.../artifacts/Fisik_2_encoder.pkl'
        self.path_fisik_3_encoder = '.../artifacts/Fisik_3_encoder.pkl'
        self.path_fisik_4_encoder = '.../artifacts/Fisik_4_encoder.pkl'
        self.path_fisik_5_encoder = '.../artifacts/Fisik_5_encoder.pkl'
        self.path_mental_1_encoder = '.../artifacts/Mental_1_encoder.pkl'
        self.path_mental_2_encoder = '.../artifacts/Mental_2_encoder.pkl'
        self.path_finansial_1_encoder = '.../artifacts/Finansial_1_encoder.pkl'
        self.path_finansial_2_encoder = '.../artifacts/Finansial_2_encoder.pkl'
        self.path_karakteristik_1_encoder = '.../artifacts/Karakteristik_1_encoder.pkl'
        self.path_karakteristik_2_encoder = '.../artifacts/Karakteristik_2_encoder.pkl'
        self.path_jadwalkeberangkatan_encoder = '.../artifacts/jadwal_keberangkatan_encoder.pkl'
        self.path_frekuensi_encoder = '.../artifacts/Frekuensi_encoder.pkl'
        self.path_kelompok_encoder = '.../artifacts/Kelompok_encoder.pkl'
        # self.path_skenario_1_encoder = 'C:/Users/akmal/transport_mode_prediction_app/artifacts/Skenario_1_encoder.pkl'
        # self.path_skenario_2_encoder = 'C:/Users/akmal/transport_mode_prediction_app/artifacts/Skenario_2_encoder.pkl'
        # self.path_skenario_3_encoder = 'C:/Users/akmal/transport_mode_prediction_app/artifacts/Skenario_3_encoder.pkl'
        # self.path_skenario_4_encoder = 'C:/Users/akmal/transport_mode_prediction_app/artifacts/Skenario_4_encoder.pkl'
        
        
        self.model = self.load_artifacts(self.path_model)
        self.le = {
            'Fisik_2_encoded': self.load_artifacts(self.path_fisik_2_encoder),
            'Fisik_3_encoded': self.load_artifacts(self.path_fisik_3_encoder),
            'Fisik_4_encoded': self.load_artifacts(self.path_fisik_4_encoder),
            'Fisik_5_encoded': self.load_artifacts(self.path_fisik_5_encoder),
            'Mental_1_encoded': self.load_artifacts(self.path_mental_1_encoder),
            'Mental_2_encoded': self.load_artifacts(self.path_mental_2_encoder),
            'Finansial_1_encoded': self.load_artifacts(self.path_finansial_1_encoder),
            'Finansial_2_encoded': self.load_artifacts(self.path_finansial_2_encoder),
            'Karakteristik_1_encoded': self.load_artifacts(self.path_karakteristik_1_encoder),
            'Karakteristik_2_encoded': self.load_artifacts(self.path_karakteristik_2_encoder),
            'jadwal_keberangkatan_encoded': self.load_artifacts(self.path_jadwalkeberangkatan_encoder),
            'Frekuensi_encoded': self.load_artifacts(self.path_frekuensi_encoder),
            'Kelompok_encoded': self.load_artifacts(self.path_kelompok_encoder)
            # 'Skenario_1_encoded': self.load_artifacts(self.path_skenario_1_encoder),
            # 'Skenario_2_encoded': self.load_artifacts(self.path_skenario_2_encoder),
            # 'Skenario_3_encoded': self.load_artifacts(self.path_skenario_3_encoder),
            # 'Skenario_4_encoded': self.load_artifacts(self.path_skenario_4_encoder)
        }

    def load_artifacts(self, path_to_artifacts):
        # Load from pickle file
        with open(path_to_artifacts, 'rb') as file:
            artifact = pickle.load(file)
        return artifact

    def preprocess_input(self, request: TransmodeRequest) -> pd.DataFrame:
        data_dict = {
            'Fisik_1': request.Fisik_1,
            'Fisik_2_encoded': request.Fisik_2_encoded,
            'Fisik_3_encoded': request.Fisik_3_encoded,
            'Fisik_4_encoded': request.Fisik_4_encoded,
            'Fisik_5_encoded': request.Fisik_5_encoded,
            'Mental_1_encoded': request.Mental_1_encoded,
            'Mental_2_encoded': request.Mental_2_encoded,
            'Finansial_1_encoded': request.Finansial_1_encoded,
            'Finansial_2_encoded': request.Finansial_2_encoded,
            'Karakteristik_1_encoded': request.Karakteristik_1_encoded,
            # 'Karakteristik_2_encoded': request.Karakteristik_2_encoded,
            'durasi_perjalanan': request.durasi_perjalanan,
            'tarif_transportasi': request.tarif_transportasi,
            'fasilitas_operator': request.fasilitas_operator,
            'jadwal_keberangkatan_encoded': request.jadwal_keberangkatan_encoded,
            'jadwal_kedatangan': request.jadwal_kedatangan,
            'Frekuensi_encoded': request.Frekuensi_encoded,
            'Kelompok_encoded': request.Kelompok_encoded
            # 'Skenario_1_encoded': request.Skenario_1_encoded,
            # 'Skenario_2_encoded': request.Skenario_2_encoded,
            # 'Skenario_3_encoded': request.Skenario_3_encoded,
            # 'Skenario_4_encoded': request.Skenario_4_encoded,
            
        }
        data_df = pd.DataFrame.from_dict([data_dict])

        data_df['Fisik_2_encoded'] = self.le['Fisik_2_encoded'].transform(data_df['Fisik_2_encoded'])
        data_df['Fisik_3_encoded'] = self.le['Fisik_3_encoded'].transform(data_df['Fisik_3_encoded'])
        data_df['Fisik_4_encoded'] = self.le['Fisik_4_encoded'].transform(data_df['Fisik_4_encoded'])
        data_df['Fisik_5_encoded'] = self.le['Fisik_5_encoded'].transform(data_df['Fisik_5_encoded'])
        data_df['Mental_1_encoded'] = self.le['Mental_1_encoded'].transform(data_df['Mental_1_encoded'])
        data_df['Mental_2_encoded'] = self.le['Mental_2_encoded'].transform(data_df['Mental_2_encoded'])
        data_df['Finansial_1_encoded'] = self.le['Finansial_1_encoded'].transform(data_df['Finansial_1_encoded'])
        data_df['Finansial_2_encoded'] = self.le['Finansial_2_encoded'].transform(data_df['Finansial_2_encoded'])
        data_df['Karakteristik_1_encoded'] = self.le['Karakteristik_1_encoded'].transform(data_df['Karakteristik_1_encoded'])
        data_df['jadwal_keberangkatan_encoded'] = self.le['jadwal_keberangkatan_encoded'].transform(data_df['jadwal_keberangkatan_encoded'])
        data_df['Frekuensi_encoded'] = self.le['Frekuensi_encoded'].transform(data_df['Frekuensi_encoded'])
        data_df['Kelompok_encoded'] = self.le['Kelompok_encoded'].transform(data_df['Kelompok_encoded'])
        # data_df['Skenario_1_encoded'] = self.le['Skenario_1_encoded'].transform(data_df['Skenario_1_encoded'])
        # data_df['Skenario_2_encoded'] = self.le['Skenario_2_encoded'].transform(data_df['Skenario_2_encoded'])
        # data_df['Skenario_3_encoded'] = self.le['Skenario_3_encoded'].transform(data_df['Skenario_3_encoded'])
        # data_df['Skenario_4_encoded'] = self.le['Skenario_4_encoded'].transform(data_df['Skenario_4_encoded'])
        return data_df
    
    def predict(self, request: TransmodeRequest) -> TransmodeResponse:
        input_df = self.preprocess_input(request)
        # Make prediction
        transmode = self.model.predict(input_df)[0]
        prediction_encoded = self.le['Karakteristik_2_encoded'].inverse_transform([transmode])[0]
        response = TransmodeResponse(Karakteristik_2_encoded=prediction_encoded)
        return response

# if __name__ == "__main__":
#     test_request = TransmodeRequest(
#         Fisik_1= 22,
#         Fisik_2_encoded= "Pria",
#         Fisik_3_encoded= "Ya",
#         Fisik_4_encoded= "Tidak",
#         Fisik_5_encoded= "Ya",
#         Mental_1_encoded= "Pelajar/Mahasiswa",
#         Mental_2_encoded= "Sarjana (S1)",
#         Finansial_1_encoded= "Rp2,000,000 - Rp4,999,999",
#         Finansial_2_encoded = "<Rp500,000",
#         Karakteristik_1_encoded = "Wisata",
#         # Karakteristik_2_encoded = "Bus AKAP Eksekutif",
#         durasi_perjalanan = 0.5,
#         tarif_transportasi = 450000,
#         fasilitas_operator= 4,
#         jadwal_keberangkatan_encoded = "Pagi hari",
#         jadwal_kedatangan = 0.3,
#         Frekuensi_encoded = "Kurang dari 1 kali per bulan",
#         Kelompok_encoded = "Sendiri"
#     )
#     transmode_service = TransmodeService()
#     res = transmode_service.predict(request = test_request)
#     print(res.Karakteristik_2_encoded)