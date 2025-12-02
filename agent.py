import os
from dotenv import load_dotenv
import json
from datetime import datetime
from typing import TypedDict, Annotated, Sequence
import operator
from langgraph.graph import StateGraph, END
from pymongo import MongoClient
from bson.objectid import ObjectId
from langchain_openai import ChatOpenAI
from langchain.tools import Tool, StructuredTool
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
import requests
from openai import OpenAI  # AIMLAPI uses OpenAI-compatible client

# Load environment variables
load_dotenv()
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")
AIMLAPI_KEY = os.getenv("AIMLAPI_KEY")

# Initialize AIMLAPI client
aiml_client = OpenAI(
    api_key=AIMLAPI_KEY,
    base_url="https://api.aimlapi.com/v1"
)

# Backend base URL for HTTP requests
BACKEND_BASE_URL = os.getenv("BACKEND_BASE_URL", "http://localhost:8000")

# ============= DATABASE SETUP =============
class MedicineDatabase:
    def __init__(self):
        MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
        DB_NAME = os.getenv("DB_NAME", "medicine_dispenser")
        self.client = MongoClient(MONGODB_URI)
        self.db = self.client[DB_NAME]
        self.medicines = self.db["medicines"]
        self.sessions = self.db["sessions"]
        self.prescriptions = self.db["prescriptions"]

    def add_medicine(self, name, dosage, times, user_id):
        doc = {
            "user_id": user_id,
            "name": name,
            "dosage": dosage,
            "times": times,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
        result = self.medicines.insert_one(doc)
        return str(result.inserted_id)

    def get_medicine(self, medicine_id, user_id):
        result = self.medicines.find_one({"_id": ObjectId(medicine_id), "user_id": user_id})
        return result

    def get_all_medicines(self, user_id):
        results = list(self.medicines.find({"user_id": user_id}).sort("created_at", -1))
        return results

    def update_medicine(self, medicine_id, user_id, name=None, dosage=None, times=None):
        updates = {}
        if name:
            updates["name"] = name
        if dosage:
            updates["dosage"] = dosage
        if times is not None:
            updates["times"] = times
        updates["updated_at"] = datetime.utcnow()
        result = self.medicines.update_one(
            {"_id": ObjectId(medicine_id), "user_id": user_id},
            {"$set": updates}
        )
        return result.modified_count > 0

    def delete_medicine(self, medicine_id, user_id):
        result = self.medicines.delete_one({"_id": ObjectId(medicine_id), "user_id": user_id})
        return result.deleted_count > 0

db = MedicineDatabase()

# ============= TOOLS SETUP =============
def add_medicine_tool(name: str, dosage: str, time: str, user_id: str) -> str:
    try:
        times = [t.strip() for t in time.replace(" and ", ",").split(",") if t.strip()]
        valid_times = []
        for t in times:
            try:
                dt = datetime.strptime(t, "%H:%M")
                valid_times.append(dt.strftime("%I:%M %p"))
            except Exception:
                return f"Invalid time format: '{t}'. Use 24-hour format (e.g., 14:30)."

        existing_med = db.medicines.find_one({
            "user_id": user_id,
            "name": name,
            "dosage": dosage,
            "status": "active"
        })

        if existing_med:
            existing_times = set(existing_med.get("times", []))
            new_times = set(valid_times)
            merged_times = sorted(existing_times.union(new_times))
            db.medicines.update_one(
                {"_id": existing_med["_id"]},
                {"$set": {"times": merged_times, "updated_at": datetime.utcnow()}}
            )
            return f"Medicine '{name}' ({dosage}) updated successfully. Times: {', '.join(merged_times)}"
        else:
            med_doc = {
                "user_id": user_id,
                "name": name,
                "dosage": dosage,
                "times": valid_times,
                "status": "active",
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            }
            result = db.medicines.insert_one(med_doc)
            return f"Medicine '{name}' ({dosage}) added successfully. Times: {', '.join(valid_times)}"

    except Exception as e:
        return f"Error adding medicine: {str(e)}"

def show_all_medicines_tool(user_id: str) -> str:
    try:
        medicines = list(db.medicines.find({"user_id": user_id, "status": "active"}))
        if not medicines:
            return "No medicines found."
        result = "Your medicines:\n"
        for med in medicines:
            result += f"- {med['name']} ({med['dosage']}) at {', '.join(med['times'])}\n"
        return result
    except Exception as e:
        return f"Error showing medicines: {str(e)}"

def medicine_pricing_tool(query: str) -> str:
    try:
        if not SERPAPI_API_KEY:
            return "Pricing search not available - API key not set."
        params = {
            "engine": "google",
            "q": f"{query} price in Pakistan",
            "api_key": SERPAPI_API_KEY
        }
        response = requests.get("https://serpapi.com/search", params=params)
        data = response.json()
        if "organic_results" in data:
            results = data["organic_results"][:3]
            pricing_info = "Pricing Information:\n"
            for result in results:
                title = result.get("title", "N/A")
                link = result.get("link", "N/A")
                snippet = result.get("snippet", "N/A")
                pricing_info += f"- {title}\n  {snippet}\n  Link: {link}\n\n"
            return pricing_info
        else:
            return "No pricing information found."
    except Exception as e:
        return f"Error searching prices: {str(e)}"

def find_nearby_pharmacies_tool(location: str) -> str:
    try:
        if not SERPAPI_API_KEY:
            return "Pharmacy search not available - API key not set."
        params = {
            "engine": "google",
            "q": f"pharmacies near {location}",
            "api_key": SERPAPI_API_KEY
        }
        response = requests.get("https://serpapi.com/search", params=params)
        data = response.json()
        if "local_results" in data and "places" in data["local_results"]:
            results = data["local_results"]["places"][:5]
            pharmacy_info = "Nearby Pharmacies:\n"
            for result in results:
                title = result.get("title", "N/A")
                address = result.get("address", "N/A")
                rating = result.get("rating", "N/A")
                phone = result.get("phone", "N/A")
                pharmacy_info += f"- {title}\n  Address: {address}\n  Rating: {rating}\n  Phone: {phone}\n\n"
            return pharmacy_info
        else:
            return "No pharmacies found."
    except Exception as e:
        return f"Error finding pharmacies: {str(e)}"

def find_doctors_tool(specialty: str, location: str) -> str:
    try:
        if not SERPAPI_API_KEY:
            return "Doctor search not available - API key not set."
        params = {
            "engine": "google",
            "q": f"{specialty} doctors near {location}",
            "api_key": SERPAPI_API_KEY
        }
        response = requests.get("https://serpapi.com/search", params=params)
        data = response.json()
        if "local_results" in data and "places" in data["local_results"]:
            results = data["local_results"]["places"][:5]
            doctor_info = f"Nearby {specialty} Doctors:\n"
            for result in results:
                title = result.get("title", "N/A")
                address = result.get("address", "N/A")
                rating = result.get("rating", "N/A")
                phone = result.get("phone", "N/A")
                doctor_info += f"- {title}\n  Address: {address}\n  Rating: {rating}\n  Phone: {phone}\n\n"
            return doctor_info
        else:
            return "No doctors found."
    except Exception as e:
        return f"Error finding doctors: {str(e)}"

def find_hospitals_tool(location: str) -> str:
    try:
        if not SERPAPI_API_KEY:
            return "Hospital search not available - API key not set."
        params = {
            "engine": "google",
            "q": f"hospitals near {location}",
            "api_key": SERPAPI_API_KEY
        }
        response = requests.get("https://serpapi.com/search", params=params)
        data = response.json()
        if "local_results" in data and "places" in data["local_results"]:
            results = data["local_results"]["places"][:5]
            hospital_info = "Nearby Hospitals:\n"
            for result in results:
                title = result.get("title", "N/A")
                address = result.get("address", "N/A")
                rating = result.get("rating", "N/A")
                phone = result.get("phone", "N/A")
                hospital_info += f"- {title}\n  Address: {address}\n  Rating: {rating}\n  Phone: {phone}\n\n"
            return hospital_info
        else:
            return "No hospitals found."
    except Exception as e:
        return f"Error finding hospitals: {str(e)}"

def find_telemedicine_services_tool() -> str:
    try:
        if not SERPAPI_API_KEY:
            return "Telemedicine search not available - API key not set."
        params = {
            "engine": "google",
            "q": "telemedicine services in Pakistan",
            "api_key": SERPAPI_API_KEY
        }
        response = requests.get("https://serpapi.com/search", params=params)
        data = response.json()
        if "organic_results" in data:
            results = data["organic_results"][:5]
            telemedicine_info = "Telemedicine Services:\n"
            for result in results:
                title = result.get("title", "N/A")
                link = result.get("link", "N/A")
                snippet = result.get("snippet", "N/A")
                telemedicine_info += f"- {title}\n  {snippet}\n  Link: {link}\n\n"
            return telemedicine_info
        else:
            return "No telemedicine services found."
    except Exception as e:
        return f"Error finding telemedicine services: {str(e)}"

def health_advisor_tool(query: str) -> str:
    try:
        response = aiml_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a knowledgeable health advisor. Be accurate and remind users to consult doctors."},
                {"role": "user", "content": query}
            ],
            temperature=0.7,
            max_tokens=512
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error getting health advice: {str(e)}"

def symptom_checker_tool(symptoms: str, duration: str, severity: str) -> str:
    try:
        response = aiml_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a symptom checker. Suggest possible causes and when to see a doctor."},
                {"role": "user", "content": f"Symptoms: {symptoms}\nDuration: {duration}\nSeverity: {severity}"}
            ],
            temperature=0.5,
            max_tokens=512
        )
        return "Symptom Checker Result:\n\n" + response.choices[0].message.content
    except Exception as e:
        return f"Error in symptom checker: {str(e)}"

def drug_interaction_checker_tool(medicines: str) -> str:
    try:
        response = aiml_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a pharmacology expert. Analyze drug interactions."},
                {"role": "user", "content": f"Check interactions for: {medicines}"}
            ],
            temperature=0.3,
            max_tokens=512
        )
        return "Drug Interaction Analysis:\n\n" + response.choices[0].message.content
    except Exception as e:
        return f"Error checking interactions: {str(e)}"

def medicine_information_tool(medicine_name: str) -> str:
    try:
        if not SERPAPI_API_KEY:
            return "Medicine information search not available - API key not set."
        params = {
            "engine": "google",
            "q": f"{medicine_name} uses side effects contraindications",
            "api_key": SERPAPI_API_KEY
        }
        response = requests.get("https://serpapi.com/search", params=params)
        data = response.json()
        if "organic_results" in data:
            results = data["organic_results"][:3]
            info = f"Information on {medicine_name}:\n"
            for result in results:
                title = result.get("title", "N/A")
                snippet = result.get("snippet", "N/A")
                link = result.get("link", "N/A")
                info += f"- {title}\n  {snippet}\n  Link: {link}\n\n"
            return info
        else:
            return "No information found."
    except Exception as e:
        return f"Error searching medicine information: {str(e)}"

def emergency_services_tool(location: str) -> str:
    try:
        if not SERPAPI_API_KEY:
            return "Emergency services search not available - API key not set."
        params = {
            "engine": "google",
            "q": f"emergency services near {location}",
            "api_key": SERPAPI_API_KEY
        }
        response = requests.get("https://serpapi.com/search", params=params)
        data = response.json()
        if "local_results" in data and "places" in data["local_results"]:
            results = data["local_results"]["places"][:5]
            emergency_info = f"Emergency Services near {location}:\n"
            for result in results:
                title = result.get("title", "N/A")
                address = result.get("address", "N/A")
                phone = result.get("phone", "N/A")
                emergency_info += f"- {title}\n  Address: {address}\n  Phone: {phone}\n\n"
            return emergency_info
        else:
            return "No emergency services found."
    except Exception as e:
        return f"Error finding emergency services: {str(e)}"

def nutrition_advisor_tool(query: str) -> str:
    try:
        response = aiml_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a nutrition expert. Provide balanced, healthy advice. Remind users to consult professionals."},
                {"role": "user", "content": query}
            ],
            temperature=0.7,
            max_tokens=512
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error getting nutrition advice: {str(e)}"

def vaccine_information_tool(vaccine_name: str) -> str:
    try:
        if not SERPAPI_API_KEY:
            return "Vaccine information search not available - API key not set."
        params = {
            "engine": "google",
            "q": f"{vaccine_name} vaccine information schedule side effects",
            "api_key": SERPAPI_API_KEY
        }
        response = requests.get("https://serpapi.com/search", params=params)
        data = response.json()
        if "organic_results" in data:
            results = data["organic_results"][:3]
            info = f"Information on {vaccine_name} vaccine:\n"
            for result in results:
                title = result.get("title", "N/A")
                snippet = result.get("snippet", "N/A")
                link = result.get("link", "N/A")
                info += f"- {title}\n  {snippet}\n  Link: {link}\n\n"
            return info
        else:
            return "No vaccine information found."
    except Exception as e:
        return f"Error searching vaccine information: {str(e)}"

# ============= TOOL LIST =============
tools = [
    StructuredTool.from_function(add_medicine_tool, name="add_medicine", description="Add medicine. Args: name, dosage, time, user_id"),
    Tool(name="show_all_medicines", func=show_all_medicines_tool, description="Show all medicines. Args: user_id"),
    StructuredTool.from_function(health_advisor_tool, name="health_advisor", description="Get health advice. Args: query"),
    StructuredTool.from_function(medicine_pricing_tool, name="medicine_pricing", description="Search medicine prices. Args: query"),
    StructuredTool.from_function(find_nearby_pharmacies_tool, name="find_pharmacies", description="Find pharmacies. Args: location"),
    StructuredTool.from_function(find_doctors_tool, name="find_doctors", description="Find doctors. Args: specialty, location"),
    StructuredTool.from_function(find_hospitals_tool, name="find_hospitals", description="Find hospitals. Args: location"),
    StructuredTool.from_function(find_telemedicine_services_tool, name="find_telemedicine_services", description="Find telemedicine services"),
    StructuredTool.from_function(symptom_checker_tool, name="symptom_checker", description="Check symptoms. Args: symptoms, duration, severity"),
    StructuredTool.from_function(drug_interaction_checker_tool, name="check_drug_interactions", description="Check drug interactions. Args: medicines"),
    StructuredTool.from_function(medicine_information_tool, name="medicine_information", description="Get detailed information on a medicine. Args: medicine_name"),
    StructuredTool.from_function(emergency_services_tool, name="emergency_services", description="Find emergency services. Args: location"),
    StructuredTool.from_function(nutrition_advisor_tool, name="nutrition_advisor", description="Get nutrition advice. Args: query"),
    StructuredTool.from_function(vaccine_information_tool, name="vaccine_information", description="Get vaccine information. Args: vaccine_name"),
]

# ============= LANGGRAPH AGENT =============
class AgentState(TypedDict):
    messages: Annotated[Sequence[HumanMessage], operator.add]
    user_id: str

def call_agent(state: AgentState):
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0.7,
        openai_api_key=AIMLAPI_KEY,
        openai_api_base="https://api.aimlapi.com/v1"
    ).bind_tools(tools)

    system_message = SystemMessage(content="""You are a medical assistant. For medicine queries (like 'panadol', 'aspirin'), call all relevant tools: medicine_pricing, find_pharmacies, find_doctors, find_hospitals, find_telemedicine_services, medicine_information. For other queries, call appropriate tools. Always provide comprehensive information.""")

    messages = [system_message] + list(state["messages"])
    response = llm.invoke(messages)

    if response.tool_calls:
        tool_results = []
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool = next((t for t in tools if t.name == tool_name), None)
            if tool:
                try:
                    result = tool.invoke(tool_args)
                    tool_results.append(f"{tool_name} result: {result}")
                except Exception as e:
                    tool_results.append(f"Error in {tool_name}: {str(e)}")
            else:
                tool_results.append(f"Tool {tool_name} not found.")
        output = "\n".join(tool_results)
    else:
        output = response.content

    return {"messages": [AIMessage(content=output)]}

def should_continue(state: AgentState):
    return END

graph = StateGraph(AgentState)
graph.add_node("agent", call_agent)
graph.set_entry_point("agent")
graph.add_edge("agent", END)

agent_graph = graph.compile()

# ============= IMAGE PROCESSING =============
def process_prescription_image(image_bytes, filename: str = None):
    try:
        if not AIMLAPI_KEY:
            return {"status": "error", "message": "AIMLAPI_KEY not set."}

        name_hint = (filename or "").lower()
        if any(word in name_hint for word in ["xray", "mri", "ct", "scan", "ultrasound", "ecg", "echo"]):
            return {
                "status": "info",
                "type": "non_prescription",
                "analysis": "This is a diagnostic image (X-ray, MRI, etc.). Automated medicine extraction not supported. Consult a doctor.",
            }

        import base64
        encoded_image = base64.b64encode(image_bytes).decode("utf-8")

        response = aiml_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Extract medicine names, dosages, and instructions from prescription. Respond ONLY with valid JSON in this format: {\"medicines\": [{\"name\": \"medicine_name\", \"dosage\": \"dosage\", \"instructions\": \"instructions\"}]} "},
                {"role": "user", "content": [
                    {"type": "text", "text": "Analyze this prescription image and extract all medicines."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}}
                ]}
            ],
            max_tokens=512
        )

        content = response.choices[0].message.content.strip()
        import json, re

        # Try to extract JSON from the response
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            try:
                result = json.loads(json_str)
                return {"status": "success", **result}
            except json.JSONDecodeError:
                # Try to fix common JSON issues
                json_str = re.sub(r',\s*\}', '}', json_str)  # Remove trailing commas
                json_str = re.sub(r',\s*\]', ']', json_str)
                try:
                    result = json.loads(json_str)
                    return {"status": "success", **result}
                except json.JSONDecodeError:
                    pass

        # If JSON parsing fails, extract medicine names manually
        medicine_names = re.findall(r'\b(?:panadol|aspirin|ibuprofen|paracetamol|amoxicillin|azithromycin|omeprazole|metformin|lisinopril|atorvastatin|simvastatin|levothyroxine|albuterol|prednisone|warfarin|clopidogrel|insulin|hydrochlorothiazide|furosemide|gabapentin|sertraline|citalopram|fluoxetine|escitalopram|tramadol|oxycodone|morphine|codeine|diazepam|lorazepam|alprazolam|zopiclone|zolpidem|amlodipine|atenolol|bisoprolol|carvedilol|enalapril|ramipril|losartan|candesartan|valsartan|amlodipine|felodipine|nicardipine|diltiazem|verapamil|digoxin|amiodarone|propafenone|flecainide|sotalol|procainamide|quinidine|warfarin|heparin|enoxaparin|dalteparin|fondaparinux|aspirin|clopidogrel|ticagrelor|prasugrel|abciximab|eptifibatide|tirofiban|streptokinase|alteplase|tenecteplase|reteplase|urokinase)\b', content.lower())

        if medicine_names:
            medicines = [{"name": name.title(), "dosage": "Not specified", "instructions": "Consult prescription"} for name in set(medicine_names)]
            return {"status": "success", "medicines": medicines}

        return {"status": "error", "message": "Could not extract medicines from image.", "raw": content}

    except Exception as e:
        return {"status": "error", "message": str(e)}

# Export
__all__ = ['agent_graph', 'process_prescription_image', 'db']
