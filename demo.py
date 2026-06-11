"""
demo.py
End-to-end demonstration of the Multi-Modal Field Service RAG Assistant.
Usage: python3 demo.py  (requires app.py running in another terminal)
"""

import requests
import time

BASE_URL = "http://localhost:5000"

def separator(title):
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)

def print_answer(response: dict):
    # Print answer block (already contains sources inside)
    answer = response.get("answer", "No answer returned")
    # Strip the embedded Sources block — we print sources separately with scores
    if "\nSources:" in answer:
        answer = answer[:answer.index("\nSources:")]
    print(answer)
    print("\nSources (with similarity scores):")
    for s in response.get("sources", []):
        print(f"  - {s['source']} | chunk {s['chunk_id']} | score {s['score']:.4f}")

def demo_health():
    separator("HEALTH CHECK")
    r = requests.get(f"{BASE_URL}/health")
    data = r.json()
    print(f"  Status  : {data['status']}")
    print(f"  Vectors : {data['vectors']}")
    print(f"  Chunks  : {data['chunks']}")

def demo_text():
    separator("MODE 1 — TEXT QUERY")
    query = "What should I check during transformer inspection?"
    print(f"  Query: {query}\n")
    r = requests.post(f"{BASE_URL}/query",
                      json={"query": query},
                      headers={"Content-Type": "application/json"})
    print_answer(r.json())

def demo_voice():
    separator("MODE 2 — VOICE QUERY (Whisper)")
    audio_path = "data/test_query.mp3"
    print(f"  Audio file: {audio_path}")
    with open(audio_path, "rb") as f:
        r = requests.post(f"{BASE_URL}/query/voice",
                          files={"file": ("test_query.mp3", f, "audio/mpeg")})
    data = r.json()
    print(f"  Transcript: {data.get('transcript', 'N/A')}\n")
    print_answer(data)

def demo_ocr():
    separator("MODE 3 — EQUIPMENT LABEL (EasyOCR)")
    image_path = "data/test_label.jpg"
    print(f"  Label image: {image_path}")
    with open(image_path, "rb") as f:
        r = requests.post(f"{BASE_URL}/query/ocr",
                          files={"file": ("test_label.jpg", f, "image/jpeg")})
    data = r.json()
    print(f"  OCR Text: {data.get('ocr_text', 'N/A')}\n")
    print_answer(data)

def demo_image():
    separator("MODE 4 — EQUIPMENT PHOTO (BLIP)")
    image_path = "data/test_equipment.jpg"
    print(f"  Image: {image_path}")
    with open(image_path, "rb") as f:
        r = requests.post(f"{BASE_URL}/query/image",
                          files={"file": ("test_equipment.jpg", f, "image/jpeg")})
    data = r.json()
    print(f"  BLIP Caption: {data.get('caption', 'N/A')}\n")
    print_answer(data)

if __name__ == "__main__":
    print("\n" + "*"*60)
    print("  Multi-Modal AI Field Service Assistant — DEMO")
    print("  AMD Developer Cloud | RAG + BLIP + Whisper + OCR")
    print("*"*60)

    try:
        demo_health()
        time.sleep(1)
        demo_text()
        time.sleep(1)
        demo_voice()
        time.sleep(1)
        demo_ocr()
        time.sleep(1)
        demo_image()
        print("\n" + "*"*60)
        print("  DEMO COMPLETE — All 4 input modes working")
        print("*"*60 + "\n")
    except requests.exceptions.ConnectionError:
        print("\nERROR: Flask server not running.")
        print("Start it first with: python3 app.py\n")
