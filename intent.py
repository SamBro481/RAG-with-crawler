import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

INTENTS = {
    "summarization": {
        "description": "User wants a summary or overview of content",
        "examples": ["what did the homepage say?", "summarize the page", "what is this site about?"],
        "top_k": 6,      
        "chunk_priority": "broad"
    },
    "fact_lookup": {
        "description": "User wants a specific fact, name, date, or value",
        "examples": ["what is Sameer's full name?", "what is the price?", "when was it founded?"],
        "top_k": 3,    
        "chunk_priority": "precise"
    },
    "explanation": {
        "description": "User wants to understand how something works",
        "examples": ["how does X work?", "explain the process", "what is machine learning?"],
        "top_k": 5,           
        "chunk_priority": "sequential"
    },
    "comparison": {
        "description": "User wants to compare two or more things",
        "examples": ["compare X and Y", "what's the difference between A and B?"],
        "top_k": 6,     
        "chunk_priority": "broad"
    },
    "instruction": {
        "description": "User wants steps or instructions on how to do something",
        "examples": ["how do I set up X?", "what are the steps to Y?", "how to install Z?"],
        "top_k": 5,
        "chunk_priority": "sequential"
    },
    "opinion": {
        "description": "User wants recommendations or opinions",
        "examples": ["what do you recommend?", "which is better?", "what should I use?"],
        "top_k": 4,
        "chunk_priority": "broad"
    }
}


def detect_intent(query: str, history: list[dict] = []) -> dict:
    intent_list = "\n".join([
        f"- {name}: {data['description']} (e.g. {data['examples'][0]})"
        for name, data in INTENTS.items()
    ])

    messages = [
        {
            "role": "system",
            "content": (
                f"You are an intent classifier for a RAG search system.\n"
                f"Given a user query, classify it into exactly one of these intents:\n\n"
                f"{intent_list}\n\n"
                f"Respond ONLY with a JSON object in this exact format:\n"
                f'{{"intent": "intent_name", "confidence": 0.95, "reasoning": "brief reason"}}\n'
                f"No other text, no markdown, just the JSON object."
            )
        },
        *history[-4:], 
        {"role": "user", "content": query}
    ]

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )

    raw = response.choices[0].message.content.strip()

    try:
        result = json.loads(raw)
        intent_name = result.get("intent", "fact_lookup")
        if intent_name not in INTENTS:
            intent_name = "fact_lookup"

        print(f"🎯 Intent: {intent_name} (confidence: {result.get('confidence', '?')})")
        print(f"   Reason: {result.get('reasoning', '')}")

        return {
            "intent": intent_name,
            "confidence": result.get("confidence", 0.8),
            "reasoning": result.get("reasoning", ""),
            **INTENTS[intent_name]   
        }

    except json.JSONDecodeError:
        print("⚠️ Intent detection failed, falling back to fact_lookup")
        return {"intent": "fact_lookup", **INTENTS["fact_lookup"]}