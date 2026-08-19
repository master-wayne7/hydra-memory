import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import llm_client as llm

DATA_PATH = os.path.join("data", "sessions.json")
OUT_PATH = os.path.join("data", "baseline_comparison.json")

def precompute():
    with open(DATA_PATH, encoding="utf-8") as fh:
        data = json.load(fh)
    
    # Sort sessions by index
    sessions = sorted(data["sessions"], key=lambda s: s.get("index", 0))
    
    # Build full concatenated transcript
    full_transcript = []
    for s in sessions:
        full_transcript.append(f"--- Session {s['id']} ({s['date']}) ---")
        for m in s["messages"]:
            full_transcript.append(f"{m['role']}: {m['text']}")
    context = "\n".join(full_transcript)
    
    questions = [
        "Where does the user currently live?",
        "Where does the user work?",
        "What is the user's favorite food?",
        "What are the user's travel plans?",
        "What is the user's pet's name?",
        "What is the user's favorite color?",
        "Where did the user previously live?"
    ]
    
    comparison = {}
    for q in questions:
        print(f"Running baseline for: {q}")
        system_prompt = "You are a helpful assistant. Answer the user's question using ONLY the provided conversation history. If the information is not in the history, say 'I don't have that information in memory.' Be concise."
        user_prompt = f"Conversation History:\n{context}\n\nQuestion: {q}"
        try:
            ans = llm.chat_text(system_prompt, user_prompt)
            comparison[q] = ans.strip()
        except Exception as e:
            print(f"Error: {e}")
            comparison[q] = "I don't have that information in memory."
            
    with open(OUT_PATH, "w", encoding="utf-8") as fw:
        json.dump(comparison, fw, indent=2)
    print("Done precomputing baseline comparison!")

if __name__ == "__main__":
    precompute()
