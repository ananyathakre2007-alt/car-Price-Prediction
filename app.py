import streamlit as st
import joblib
import numpy as np

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚗",
    layout="centered",
)

# ── Load model & scaler ────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    scaler = joblib.load("scaler.pkl")
    model  = joblib.load("ridge_model.pkl")
    return scaler, model

scaler, model = load_artifacts()

# ── UI ─────────────────────────────────────────────────────────────────────────
st.title("🚗 Car Price Predictor")
st.markdown("Fill in the car details below and click **Predict Price** to get an estimate.")

st.markdown("---")

# ── Section 1 : Numeric features ──────────────────────────────────────────────
st.subheader("📐 Numeric Features")

col1, col2, col3 = st.columns(3)

with col1:
    symboling     = st.number_input("symboling",     value=0,    step=1,
                                    help="Risk factor: -3 (safe) to +3 (risky)")
    wheelbase     = st.number_input("wheelbase",     value=98.0, step=0.1,
                                    help="Distance between axles (inches)")
    carlength     = st.number_input("carlength",     value=168.0, step=0.1,
                                    help="Length of the car (inches)")

with col2:
    carwidth      = st.number_input("carwidth",      value=64.0, step=0.1,
                                    help="Width of the car (inches)")
    curbweight    = st.number_input("curbweight",    value=2500, step=10,
                                    help="Weight of the car without passengers/cargo (lbs)")
    enginesize    = st.number_input("enginesize",    value=120,  step=1,
                                    help="Engine displacement (cubic inches)")

with col3:
    horsepower    = st.number_input("horsepower",    value=100,  step=1,
                                    help="Engine power output (hp)")
    citympg       = st.number_input("citympg",       value=25,   step=1,
                                    help="Fuel efficiency in city driving (mpg)")

st.markdown("---")

# ── Section 2 : Categorical features ──────────────────────────────────────────
st.subheader("🔧 Categorical Features")

col4, col5 = st.columns(2)

with col4:
    carbody = st.selectbox(
        "carbody",
        options=["convertible", "hardtop", "hatchback", "sedan", "wagon"],
        index=3,
        help="Body style of the car"
    )

    drivewheel = st.selectbox(
        "drivewheel",
        options=["4wd", "fwd", "rwd"],
        index=1,
        help="Drive wheel configuration"
    )

    enginelocation = st.selectbox(
        "enginelocation",
        options=["front", "rear"],
        index=0,
        help="Engine position"
    )

with col5:
    enginetype = st.selectbox(
        "enginetype",
        options=["dohc", "dohcv", "l", "ohc", "ohcf", "ohcv", "rotor"],
        index=3,
        help="Engine type"
    )

    cylindernumber = st.selectbox(
        "cylindernumber",
        options=["two", "three", "four", "five", "six", "eight", "twelve"],
        index=2,
        help="Number of cylinders"
    )

st.markdown("---")

# ── One-hot encoding helpers ────────────────────────────────────────────────────
def encode_inputs():
    # carbody dummies (reference = convertible)
    carbody_hardtop  = 1 if carbody == "hardtop"  else 0
    carbody_hatchback= 1 if carbody == "hatchback" else 0
    carbody_sedan    = 1 if carbody == "sedan"     else 0
    carbody_wagon    = 1 if carbody == "wagon"     else 0

    # drivewheel dummies (reference = 4wd)
    drivewheel_fwd   = 1 if drivewheel == "fwd" else 0
    drivewheel_rwd   = 1 if drivewheel == "rwd" else 0

    # enginelocation dummy (reference = front)
    enginelocation_rear = 1 if enginelocation == "rear" else 0

    # enginetype dummies (reference = dohc)
    enginetype_dohcv  = 1 if enginetype == "dohcv"  else 0
    enginetype_l      = 1 if enginetype == "l"       else 0
    enginetype_ohc    = 1 if enginetype == "ohc"     else 0
    enginetype_ohcf   = 1 if enginetype == "ohcf"    else 0
    enginetype_ohcv   = 1 if enginetype == "ohcv"    else 0
    enginetype_rotor  = 1 if enginetype == "rotor"   else 0

    # cylindernumber dummies (reference = eight)
    cylindernumber_five   = 1 if cylindernumber == "five"   else 0
    cylindernumber_four   = 1 if cylindernumber == "four"   else 0
    cylindernumber_six    = 1 if cylindernumber == "six"    else 0
    cylindernumber_three  = 1 if cylindernumber == "three"  else 0
    cylindernumber_twelve = 1 if cylindernumber == "twelve" else 0
    cylindernumber_two    = 1 if cylindernumber == "two"    else 0

    # Assemble in the exact order the scaler was fitted on
    feature_vector = [
        symboling, wheelbase, carlength, carwidth, curbweight,
        enginesize, horsepower, citympg,
        carbody_hardtop, carbody_hatchback, carbody_sedan, carbody_wagon,
        drivewheel_fwd, drivewheel_rwd,
        enginelocation_rear,
        enginetype_dohcv, enginetype_l, enginetype_ohc,
        enginetype_ohcf, enginetype_ohcv, enginetype_rotor,
        cylindernumber_five, cylindernumber_four, cylindernumber_six,
        cylindernumber_three, cylindernumber_twelve, cylindernumber_two,
    ]
    return np.array(feature_vector).reshape(1, -1)

# ── Predict button ─────────────────────────────────────────────────────────────
if st.button("🔍 Predict Price", use_container_width=True, type="primary"):
    X_raw    = encode_inputs()
    X_scaled = scaler.transform(X_raw)
    price    = model.predict(X_scaled)[0]

    if price < 0:
        st.warning("The model returned a negative value — the input combination may be outside the training distribution.")
        price = abs(price)

    st.success(f"### Estimated Car Price: **${price:,.2f}**")

    with st.expander("📊 Feature vector sent to the model"):
        feature_names = list(scaler.feature_names_in_)
        import pandas as pd
        df = pd.DataFrame(X_raw, columns=feature_names).T
        df.columns = ["Value"]
        st.dataframe(df, use_container_width=True)