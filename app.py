import streamlit as st
import pandas as pd
import joblib
import warnings
warnings.filterwarnings("ignore")

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Diagnóstico de Obesidade", page_icon="⚖️", layout="centered")

# ── Load model ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("random.joblib")

model = load_model()

# ── Label mappings (LabelEncoder alphabetical order, fitted on training data) ─
GENDER_MAP       = {"Mulher": 0, "Homem": 1}                        # Female=0, Male=1
YESNO_MAP        = {"Sim": 1, "Não": 0}                             # no=0, yes=1
CAEC_MAP         = {"Sempre": 0, "Frequentemente": 1, "Às vezes": 2, "Não": 3}
CALC_MAP         = {"Sempre": 0, "Frequentemente": 1, "Às vezes": 2, "Não": 3}
MTRANS_MAP       = {
    "Carro": 0,             # Automobile
    "Bicicleta": 1,         # Bike
    "Moto": 2,              # Motorbike
    "Transporte público": 3,# Public_Transportation
    "A pé": 4               # Walking
}
FCVC_MAP  = {"Raramente": 1, "Às vezes": 2, "Sempre": 3}
NCP_MAP   = {"1": 1, "2": 2, "3": 3, "Mais de 3": 4}
CH2O_MAP  = {"Menos de 1L": 1, "1–2L": 2, "Mais de 2L": 3}
FAF_MAP   = {"Nenhuma": 0, "1–2x / semana": 1, "3–4x / semana": 2, "5x / semana ou mais": 3}
TUE_MAP   = {"0–2h / dia": 0, "3–5h / dia": 1, "Mais de 5h / dia": 2}

# ── Class labels ──────────────────────────────────────────────────────────────
CLASS_LABELS = {
    0: ("Abaixo do Peso",    "Insufficient_Weight", "🔵"),
    1: ("Peso Normal",       "Normal_Weight",       "🟢"),
    2: ("Obesidade Tipo I",  "Obesity_Type_I",      "🟠"),
    3: ("Obesidade Tipo II", "Obesity_Type_II",     "🔴"),
    4: ("Obesidade Tipo III","Obesity_Type_III",    "🔴"),
    5: ("Sobrepeso Nível I", "Overweight_Level_I",  "🟡"),
    6: ("Sobrepeso Nível II","Overweight_Level_II", "🟡"),
}

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("⚖️ Modelo de ML para Auxílio de Diagnóstico de Obesidade")
st.markdown("Preencha os dados abaixo e clique em **Classificar** para obter o diagnóstico.")

with st.form("form_paciente"):

    st.subheader("Dados Pessoais")
    genero = st.radio("Gênero", ["Homem", "Mulher"], horizontal=True)
    idade  = st.number_input("Idade", min_value=14, max_value=61, value=25)

    col1, col2 = st.columns(2)
    with col1:
        peso   = st.slider("Peso (kg)",    min_value=0.0,  max_value=200.0, value=70.0,  step=0.5)
    with col2:
        altura = st.slider("Altura (cm)",  min_value=120.0, max_value=300.0, value=170.0, step=0.5)

    st.subheader("Histórico & Hábitos Alimentares")
    historico_familiar = st.radio("Histórico familiar de obesidade", ["Sim", "Não"], horizontal=True)
    favc  = st.radio("Consumo de alimentos calóricos (FAVC)", ["Sim", "Não"], horizontal=True)
    fcvc  = st.select_slider("Consumo de vegetais (FCVC)",
                              options=["Raramente", "Às vezes", "Sempre"], value="Às vezes")
    ncp   = st.select_slider("Número de refeições principais (NCP)",
                              options=["1", "2", "3", "Mais de 3"], value="3")
    caec  = st.selectbox("Consumo de lanches entre refeições (CAEC)",
                          ["Não", "Às vezes", "Frequentemente", "Sempre"])

    st.subheader("Monitoramento & Estilo de Vida")
    smoke = st.radio("Fuma? (SMOKE)", ["Sim", "Não"], horizontal=True)
    ch2o  = st.select_slider("Consumo diário de água (CH2O)",
                              options=["Menos de 1L", "1–2L", "Mais de 2L"], value="1–2L")
    scc   = st.radio("Monitora as calorias diárias? (SCC)", ["Sim", "Não"], horizontal=True)
    faf   = st.select_slider("Frequência semanal de atividade física (FAF)",
                              options=["Nenhuma", "1–2x / semana", "3–4x / semana", "5x / semana ou mais"],
                              value="1–2x / semana")
    tue   = st.select_slider("Tempo de uso de tela por dia (TUE)",
                              options=["0–2h / dia", "3–5h / dia", "Mais de 5h / dia"], value="3–5h / dia")
    calc  = st.selectbox("Consumo de álcool (CALC)",
                          ["Não", "Às vezes", "Frequentemente", "Sempre"])

    st.subheader("Transporte Habitual")
    mtrans = st.radio("Meio de transporte principal (MTRANS)",
                       ["Carro", "Moto", "Bicicleta", "Transporte público", "A pé"],
                       horizontal=True)

    submitted = st.form_submit_button("🔍 Classificar", use_container_width=True)

# ── Prediction ────────────────────────────────────────────────────────────────
if submitted:
    # Convert altura de cm para metros (modelo foi treinado em metros)
    altura_m = altura / 100.0

    input_data = pd.DataFrame([[
        GENDER_MAP[genero],
        float(idade),
        altura_m,
        float(peso),
        YESNO_MAP[historico_familiar],
        YESNO_MAP[favc],
        float(FCVC_MAP[fcvc]),
        float(NCP_MAP[ncp]),
        float(CAEC_MAP[caec]),
        YESNO_MAP[smoke],
        float(CH2O_MAP[ch2o]),
        YESNO_MAP[scc],
        float(FAF_MAP[faf]),
        float(TUE_MAP[tue]),
        float(CALC_MAP[calc]),
        float(MTRANS_MAP[mtrans]),
    ]], columns=model.feature_names_in_)

    pred_idx  = int(model.predict(input_data)[0])
    proba     = model.predict_proba(input_data)[0]

    label_pt, label_en, emoji = CLASS_LABELS[pred_idx]

    st.divider()
    st.subheader("Resultado da Classificação")

    col_res, col_conf = st.columns([2, 1])
    with col_res:
        st.metric("Diagnóstico", f"{emoji} {label_pt}")
        st.caption(f"Classe técnica: `{label_en}`")
    #with col_conf:
    #    st.metric("Confiança", f"{proba[pred_idx]*100:.1f}%")
