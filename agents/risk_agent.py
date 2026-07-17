"""
Risk Assessment Agent.
Takes fraud probability + key features, returns a risk level with reasoning.
"""
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.2
)

RISK_PROMPT = ChatPromptTemplate.from_template("""
You are a telecom fraud risk analyst. Given the model output and features below,
classify the risk level as LOW, MEDIUM, or HIGH, and give a 2-3 sentence reason
based on the specific feature values provided. Be concise and analytical.

Fraud Probability: {fraud_probability}
Call Duration (sec): {call_duration_sec}
Calls Per Day: {calls_per_day}
Night Call Ratio: {night_call_ratio}
SIM Age (days): {sim_age_days}
Unique Contacts: {unique_contacts}
Location Changes: {location_changes}

Respond in this exact format:
Risk Level: <LOW/MEDIUM/HIGH>
Reason: <your reasoning>
""")

def assess_risk(record: dict) -> str:
    chain = RISK_PROMPT | llm
    response = chain.invoke(record)
    return response.content

if __name__ == "__main__":
    sample = {
        "fraud_probability": 0.91,
        "call_duration_sec": 2,
        "calls_per_day": 420,
        "night_call_ratio": 0.92,
        "sim_age_days": 2,
        "unique_contacts": 380,
        "location_changes": 9
    }
    print(assess_risk(sample))