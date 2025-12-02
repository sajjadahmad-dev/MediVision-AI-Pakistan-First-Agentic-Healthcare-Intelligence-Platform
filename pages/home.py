import streamlit as st

st.set_page_config(page_title="MediVision AI - Home", page_icon="🩺")

st.title("🩺 MediVision AI - Your Health Navigator")
st.markdown("""
**Instant Analysis, Expert Guidance, Immediate Action**

Welcome to MediVision AI! Choose a feature from the sidebar to get started.

- **Image Analysis**: Upload medical images for AI-powered detection and description.
- **Chat with AI**: Ask health-related questions and get advice.
- **Medicines**: Manage your medicines and get information.
""")

st.header("Quick Actions")
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("📸 Analyze Image"):
        st.switch_page("pages/image_analysis.py")

with col2:
    if st.button("💬 Chat with AI"):
        st.switch_page("pages/chat.py")

with col3:
    if st.button("💊 Manage Medicines"):
        st.switch_page("pages/medicines.py")
