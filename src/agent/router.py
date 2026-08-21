"""Semantic routing agent to filter noise and filler words."""

import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

# Using a lightweight Flash model for minimum Time-To-First-Token latency
ROUTER_MODEL = "gemini-1.5-flash-8b"

def is_meaningful_intent(transcript: str) -> bool:
    """
    Evaluates if the transcript contains a deliberate user action or request.
    Returns True for valid inputs, False for filler words, stutters, or noise.
    """
    if not transcript.strip():
        return False
        
    prompt = f"""Evaluate the following transcript from a voice assistant user.
Does it contain a meaningful conversational intent, command, or partial thought? 
Or is it just a filler word (e.g., um, uh), background noise, or a clearing of the throat?

Transcript: "{transcript}"

Output ONLY the exact word "True" if meaningful, or "False" if it is noise/filler. No punctuation or explanations."""

    try:
        model = genai.GenerativeModel(ROUTER_MODEL)
        # Using low temperature for strict deterministic output
        response = model.generate_content(prompt, generation_config={"temperature": 0.0})
        
        result = response.text.strip().lower()
        return result == "true"
        
    except Exception as e:
        print(f"[Router Error] {e}")
        # Default to True if the router fails, to avoid ignoring valid user input
        return True