import streamlit as st
from agent import agent_graph, process_prescription_image, db
from langchain.schema import HumanMessage
import base64
from PIL import Image
import io

st.set_page_config(page_title="MediVision AI", page_icon="🩺", layout="wide")

st.title("🩺 MediVision AI - Your Health Navigator")
st.markdown("""
**Instant Analysis, Expert Guidance, Immediate Action**

Upload a prescription image or enter medicine details manually. Chat with our AI agent for health advice, find nearby pharmacies, doctors, and telemedicine options.
""")

# User ID (for demo, use a default; in production, use authentication)
user_id = st.sidebar.text_input("User ID", value="demo_user", key="user_id")

# Initialize session state for chat
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Image Upload Section
st.header("📸 Upload Prescription Image (Optional)")
uploaded_file = st.file_uploader("Choose a prescription image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Prescription", use_column_width=True)

    if st.button("Analyze Prescription"):
        with st.spinner("Analyzing image..."):
            image_bytes = io.BytesIO()
            image.save(image_bytes, format="JPEG")
            image_bytes = image_bytes.getvalue()

            result = process_prescription_image(image_bytes, uploaded_file.name)

            if result["status"] == "success":
                st.success("Prescription analyzed successfully!")
                medicines = result.get("medicines", [])
                st.subheader("Extracted Medicines:")
                for med in medicines:
                    name = med.get("name", "Unknown")
                    dosage = med.get("dosage", "Unknown")
                    instructions = med.get("instructions", "Not specified")
                    st.write(f"- **{name}** ({dosage}) - {instructions}")

                if medicines and st.button("Add All Medicines to Database"):
                    for med in medicines:
                        name = med.get("name", "Unknown")
                        dosage = med.get("dosage", "500mg")
                        time_str = "08:00,14:00,20:00"  # Default times
                        try:
                            response = agent_graph.invoke({
                                "messages": [HumanMessage(content=f"add_medicine: name={name}, dosage={dosage}, time={time_str}, user_id={user_id}")],
                                "user_id": user_id
                            })
                            st.write(f"Added {name}: {response['messages'][-1].content}")
                        except Exception as e:
                            st.error(f"Error adding {name}: {str(e)}")
                    st.success("Medicines added to database!")

                    # After adding, provide comprehensive info for each medicine
                    st.subheader("Comprehensive Medicine Information:")
                    for med in medicines:
                        medicine_name = med.get("name", "Unknown")
                        query = f"Tell me about {medicine_name}"
                        try:
                            response = agent_graph.invoke({
                                "messages": [HumanMessage(content=query)],
                                "user_id": user_id
                            })
                            st.write(f"**{medicine_name}:**")
                            st.write(response["messages"][-1].content)
                        except Exception as e:
                            st.error(f"Error getting info for {medicine_name}: {str(e)}")
            elif result["status"] == "info":
                st.info(result["analysis"])
            else:
                st.error(f"Analysis failed: {result['message']}")

# Manual Medicine Entry
st.header("💊 Add Medicine Manually")
with st.form("add_med_form"):
    med_name = st.text_input("Medicine Name")
    med_dosage = st.text_input("Dosage (e.g., 500mg)")
    med_times = st.text_input("Times (e.g., 08:00, 14:00, 20:00)")
    submitted = st.form_submit_button("Add Medicine")

    if submitted and med_name:
        response = agent_graph.invoke({
            "messages": [HumanMessage(content=f"add_medicine: name={med_name}, dosage={med_dosage}, time={med_times}, user_id={user_id}")],
            "user_id": user_id
        })
        st.write(response["messages"][-1].content)

# Chat Interface
st.header("💬 Chat with MediVision AI")
st.markdown("Ask about symptoms, health advice, find pharmacies, doctors, etc.")

user_input = st.text_input("Your message:", key="user_input")

if st.button("Send") and user_input:
    # Add user message to history
    st.session_state.chat_history.append(HumanMessage(content=user_input))

    # Invoke agent
    with st.spinner("Thinking..."):
        response = agent_graph.invoke({
            "messages": st.session_state.chat_history,
            "user_id": user_id
        })

    # Add agent response to history
    agent_response = response["messages"][-1].content
    st.session_state.chat_history.append(response["messages"][-1])

    # Display response
    st.write("**MediVision AI:**")
    st.write(agent_response)

# Display Chat History
if st.session_state.chat_history:
    st.header("Chat History")
    for msg in st.session_state.chat_history:
        if isinstance(msg, HumanMessage):
            st.write(f"**You:** {msg.content}")
        else:
            st.write(f"**MediVision AI:** {msg.content}")

# Show All Medicines
st.header("📋 Your Medicines")
if st.button("Show All Medicines"):
    response = agent_graph.invoke({
        "messages": [HumanMessage(content=f"show_all_medicines: user_id={user_id}")],
        "user_id": user_id
    })
    st.write(response["messages"][-1].content)
