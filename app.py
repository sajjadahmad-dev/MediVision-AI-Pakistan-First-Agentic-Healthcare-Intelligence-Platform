import streamlit as st

st.set_page_config(page_title="MediVision AI", page_icon="🩺", layout="wide")

st.title("🩺 MediVision AI - Your Health Navigator")
st.markdown("""
**Instant Analysis, Expert Guidance, Immediate Action**

Navigate to different features using the sidebar.
""")

# Navigation
page = st.sidebar.selectbox("Choose a page", ["Home", "Image Analysis", "Eye Conjunctiva", "Chat", "Medicines"])

if page == "Home":
    st.switch_page("pages/home.py")
elif page == "Image Analysis":
    st.switch_page("pages/image_analysis.py")
elif page == "Eye Conjunctiva":
    st.switch_page("pages/eye_conjunctiva.py")
elif page == "Chat":
    st.switch_page("pages/chat.py")
elif page == "Medicines":
    st.switch_page("pages/medicines.py")
