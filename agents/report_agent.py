"""
Report Agent.
Combines risk assessment + investigation findings into a final,
structured investigation report.
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

REPORT_PROMPT = ChatPromptTemplate.from_template("""
You are a telecom fraud analyst writing a formal investigation report.
Combine the information below into a clean, structured report.

Customer ID: {customer_id}
Fraud Probability: {fraud_probability}
Risk Assessment: {risk_assessment}
Investigation Findings: {investigation_findings}

Respond in this exact markdown format:

# Fraud Investigation Report

**Customer ID:** {customer_id}
**Fraud Score:** {fraud_probability}

## Risk Level
<extract from risk assessment>

## Supporting Observations
<2-3 bullet points synthesizing the evidence>

## Likely Fraud Type
<extract from investigation findings>

## Recommended Actions
<bullet list from investigation findings>

## Final Recommendation
<1-2 sentence closing analyst recommendation>
""")

def generate_report(record: dict) -> str:
    chain = REPORT_PROMPT | llm
    response = chain.invoke(record)
    return response.content

if __name__ == "__main__":
    sample = {
        "customer_id": "CUST003886",
        "fraud_probability": 0.91,
        "risk_assessment": "Risk Level: HIGH. Reason: Extremely short calls, high frequency, new SIM, high night activity.",
        "investigation_findings": "Likely Fraud Type: SIM Box Fraud. Confidence: High. Checks: Verify SIM, blacklist number, freeze account."
    }
    print(generate_report(sample))