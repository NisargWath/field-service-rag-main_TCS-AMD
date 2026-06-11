"""
image_analyzer.py
Uses BLIP to generate a caption from an equipment image,
then feeds that caption into the RAG pipeline for a grounded answer.
"""

import sys
import os
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch

_processor = None
_model = None

def _load_blip():
    global _processor, _model
    if _processor is None:
        print("Loading BLIP model (first run, will cache)...")
        _processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        _model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        )
        _model.eval()
    return _processor, _model

def caption_image(image_path: str) -> str:
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    processor, model = _load_blip()
    image = Image.open(image_path).convert("RGB")
    inputs = processor(image, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=60)
    caption = processor.decode(output[0], skip_special_tokens=True)
    return caption

def analyze_equipment_image(image_path: str) -> dict:
    caption = caption_image(image_path)

    # Better query templates based on caption keywords
    caption_lower = caption.lower()
    if any(w in caption_lower for w in ["transformer", "pole", "electrical", "power"]):
        rag_query = f"Technician inspecting {caption}. What are the safety checks and inspection steps?"
    elif any(w in caption_lower for w in ["wire", "cable", "line"]):
        rag_query = f"Technician sees {caption}. What electrical hazards and PPE requirements apply?"
    elif any(w in caption_lower for w in ["oil", "leak", "fluid"]):
        rag_query = f"Technician observes {caption}. What are the procedures for oil leak inspection?"
    else:
        rag_query = f"Field technician is inspecting equipment: {caption}. What safety procedures and checks apply?"

    return {
        "image_path": image_path,
        "caption": caption,
        "rag_query": rag_query,
    }

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 rag/image_analyzer.py <path_to_image>")
        sys.exit(1)
    image_path = sys.argv[1]
    print(f"\nAnalyzing image: {image_path}")
    result = analyze_equipment_image(image_path)
    print(f"\nBLIP Caption:\n  {result['caption']}")
    print(f"\nGenerated RAG Query:\n  {result['rag_query']}")
