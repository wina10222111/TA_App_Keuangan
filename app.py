import os
import re
import joblib
import pandas as pd
import streamlit as st

# Config Halaman Utama
st.set_page_config(
    page_title="Personal Financial Assistant",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS Theme
custom_css = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Fredoka:wght@400;600;700&family=Quicksand:wght@500;600;700&display=swap');

    html, body, [class*="css"], .stMarkdown, p, div {
        font-family: 'Quicksand', sans-serif !important;
        color: #3D2314 !important;
    }

    h1, h2, h3, .stTitle {
        font-family: 'Fredoka', cursive !important;
        color: #C85A32 !important;
        font-weight: 700 !important;
    }

    .stApp {
        background-color: #EBF4F6 !important;
    }

    [data-testid="stSidebar"] {
        background-color: #F8EFEA !important;
        border-right: 2px solid #F5C2A5 !important;
    }

    .stTextInput input, .stNumberInput input, .stSelectbox select, .stDateInput input {
        border-radius: 18px !important;
        background-color: #F5C2A5 !important;
        color: #3D2314 !important;
        border: 2px solid #E2A07E !important;
        font-weight: 600 !important;
        padding: 10px 15px !important;
    }

    .stButton > button {
        background: linear-gradient(135deg, #FF7F50 0%, #C85A32 100%) !important;
        color: white !important;
        font-family: 'Fredoka', cursive !important;
        font-size: 18px !important;
        font-weight: 600 !important;
        border-radius: 25px !important;
        border: 2px solid #FFFFFF !important;
        padding: 10px 25px !important;
        box-shadow: 0px 4px 10px rgba(200, 90, 50, 0.3) !important;
    }

    [data-testid="stMetric"], .stAlert, div[data-testid="stForm"] {
        background-color: #FFFFFF !important;
        border-radius: 20px !important;
        border: 2px solid #F5C2A5 !important;
        box-shadow: 0px 6px 15px rgba(0, 0, 0, 0.04) !important;
        padding: 15px !important;
    }

    [data-testid="stMetricLabel"] {
        font-family: 'Fredoka', cursive !important;
        color: #C85A32 !important;
    }

    div[role="radiogroup"] > label {
        background-color: #FFFFFF !important;
        border-radius: 15px !important;
        padding: 8px 15px !important;
        margin-bottom: 8px !important;
        border: 2px solid #F5C2A5 !important;
    }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)


def clean_text_pure(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


@st.cache_resource
def load_ml_models():
    models = {}

    def get_path(filename):
        paths_to_check = [
            os.path.join("artifacts", filename),
            filename
        ]
        for p in paths_to_check:
            if os.path.exists(p):
                return p
        return None

    path_embedder = get_path("sentence_embedder.pkl")
    path_rf_semantic = get_path("rf_model_semantic.pkl")
    path_classifier = get_path("classifier_kategori_keuangan.pkl")
    path_rf_kategori = get_path("rf_model_kategori.pkl")
    path_tfidf = get_path("tfidf_vectorizer.pkl")

    if path_embedder and path_rf_semantic:
        models["embedder"] = joblib.load(path_embedder)
        models["rf_semantic"] = joblib.load(path_rf_semantic)
        models["nlp_type"] = "semantic"
    elif path_classifier:
        models["rf_semantic"] = joblib.load(path_classifier)
        models["embedder"] = joblib.load(path_embedder) if path_embedder else None
        models["nlp_type"] = "classifier"
    elif path_rf_kategori and path_tfidf:
        models["embedder"] = None
        models["rf_semantic"] = joblib.load(path_rf_kategori)
        models["tfidf"] = joblib.load(path_tfidf)
        models["nlp_type"] = "tfidf"
    else:
        models["rf_semantic"] = None

    path_linear = get_path("model_linear_regression.pkl")
    models["linear_reg"] = joblib.load(path_linear) if path_linear else None

    path_kmeans = get_path("kmeans_model.pkl")
    path_scaler = get_path("scaler.pkl")
    models["kmeans"] = joblib.load(path_kmeans) if path_kmeans else None
    models["scaler"] = joblib.load(path_scaler) if path_scaler else None

    return models


st.session_state["ml_models"] = load_ml_models()
st.session_state["clean_text_func"] = clean_text_pure

# Safeguard key agar tidak terjadi KeyError jika ada fungsi tua yang memanggil
st.session_state["budget_limit"] = st.session_state.get("budget_limit", 0.0)

CSV_FILE_PATH = "Data_Finance_6_Bulan (1).csv"


def load_initial_data():
    if os.path.exists(CSV_FILE_PATH):
        df = pd.read_csv(CSV_FILE_PATH)
        df["Date"] = pd.to_datetime(df["Date"])
        return df[["Date", "Title", "Type", "Amount", "Category"]]
    else:
        dummy_data = {
            "Date": pd.to_datetime([
                "2026-07-01",
                "2026-07-02",
                "2026-07-05",
                "2026-07-10",
                "2026-07-15",
            ]),
            "Title": [
                "Gaji Bulanan",
                "Beli Token PLN",
                "Belanja Bulanan",
                "Makan Siang Resto",
                "Bayar Wifi",
            ],
            "Type": ["INCOME", "EXPENSE", "EXPENSE", "EXPENSE", "EXPENSE"],
            "Amount": [10000000, 200000, 1500000, 150000, 350000],
            "Category": [
                "Pendapatan",
                "Tagihan & Utility",
                "Kebutuhan Pokok",
                "Makanan & Minuman",
                "Tagihan & Utility",
            ],
        }
        return pd.DataFrame(dummy_data)


if "data_transaksi" not in st.session_state:
    st.session_state["data_transaksi"] = load_initial_data()

st.sidebar.title("🤑 Navigasi Menu")
page = st.sidebar.radio(
    "Pilih Halaman:",
    [
        "📊 Dashboard Utama",
        "📝 Pencatatan & Auto-Cat",
        "📈 Proyeksi & Profiling",
    ],
)

st.sidebar.divider()
st.sidebar.info("💡 **Tips Cute:** Catat pengeluaran harianmu agar keuangan tetap sehat! 💸")

if page == "📊 Dashboard Utama":
    from views import dashboard
    dashboard.show()
elif page == "📝 Pencatatan & Auto-Cat":
    from views import pencatatan
    pencatatan.show()
elif page == "📈 Proyeksi & Profiling":
    from views import proyeksi
    proyeksi.show()
