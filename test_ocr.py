import base64, requests

with open("maize.jpeg", "rb") as f:
    b64 = base64.b64encode(f.read()).decode()

payload = {
    "model": "chandra-ocr-2.Q8_0.gguf",
    "max_tokens": 200,
    "messages": [{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            {"type": "text", "text": "OCR this image to HTML"}
        ]
    }]
}

r = requests.post("http://localhost:8000/v1/chat/completions", json=payload)
print(r.json())