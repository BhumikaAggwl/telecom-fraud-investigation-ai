"""
Investigation Agent.
Takes fraud prediction + risk assessment, infers likely fraud type
and recommends investigation steps.
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

INVESTIGATION_PROMPT = ChatPromptTemplate.from_template("""
You are a telecom fraud investigator. Based on the risk assessment and features
below, identify the MOST LIKELY fraud type from this list:
[Spam Campaign, Robocalling, SIM Box Fraud, Identity Fraud, Subscription Fraud, Unclear]

Also give a confidence level (Low/Medium/High) and 2-3 recommended checks.

Risk Assessment: {risk_assessment}
Calls Per Day: {calls_per_day}
Night Call Ratio: {night_call_ratio}
SIM Age (days): {sim_age_days}
Unique Contacts: {unique_contacts}
Device Changes: {device_changes}

Respond in this exact format:
Likely Fraud Type: <type>
Confidence: <Low/Medium/High>
Recommended Checks:
- <check 1>
- <check 2>
- <check 3>
""")

def investigate(record: dict) -> str:
    chain = INVESTIGATION_PROMPT | llm
    response = chain.invoke(record)
    return response.content

if __name__ == "__main__":
    sample = {
        "risk_assessment": "Risk Level: HIGH. Reason: Extremely short calls, high frequency, new SIM.",
        "calls_per_day": 420,
        "night_call_ratio": 0.92,
        "sim_age_days": 2,
        "unique_contacts": 380,
        "device_changes": 4
    }
    print(investigate(sample))