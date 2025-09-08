from pydantic import BaseModel

class TransmodeRequest(BaseModel):
    Fisik_1: int
    Fisik_2_encoded: str
    Fisik_3_encoded: str
    Fisik_4_encoded: str
    Fisik_5_encoded: str
    Mental_1_encoded: str
    Mental_2_encoded: str
    Finansial_1_encoded: str
    Finansial_2_encoded: str
    Karakteristik_1_encoded: str
    # Karakteristik_2_encoded: str
    durasi_perjalanan: float
    tarif_transportasi: int
    fasilitas_operator: int
    jadwal_keberangkatan_encoded: str
    jadwal_kedatangan: float
    Frekuensi_encoded: str
    Kelompok_encoded: str
    # Skenario_1_encoded: str
    # Skenario_2_encoded: str
    # Skenario_3_encoded: str
    # Skenario_4_encoded: str
    

class TransmodeResponse(BaseModel):
    Karakteristik_2_encoded: str