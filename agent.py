import json
import os
from groq import Groq

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

with open("metrics.json", "r") as f:
    BENCHMARK_DATA = json.load(f)

SYSTEM_PROMPT = f"""
You are an expert Credit Risk AI Advisor presenting model selection results to non-technical financial stakeholders.

Benchmark Data Context:
- Winning Production Model: {BENCHMARK_DATA['winning_model']}
- Performance Metrics: {json.dumps(BENCHMARK_DATA['metrics'])}
- Key Risk Factors (Top Drivers): {', '.join(BENCHMARK_DATA['top_features'])}

Instructions:
1. Explain clearly why XGBoost outperformed the PyTorch MLP (higher AUC score, faster inference time, superior interpretability on tabular credit data).
2. Answer non-technical stakeholder questions about credit risk, model trade-offs, and metric meanings.
3. Keep answers clear, direct, and focused on business value.
"""

def ask_stakeholder_agent(user_question, chat_history):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    
    for msg in chat_history:
        messages.append(msg)
        
    messages.append({"role": "user", "content": user_question})
    
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=0.3
    )
    return response.choices[0].message.content
