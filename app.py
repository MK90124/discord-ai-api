from flask import Flask, request, jsonify
from transformers import pipeline
from huggingface_hub import login
import os

# 用环境变量登录 Hugging Face Token
HF_TOKEN = os.environ.get("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")
login(HF_TOKEN)

app = Flask(__name__)

model = pipeline("text2text-generation", model="google/flan-t5-base")

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    prompt = data.get("prompt", "")
    result = model(prompt, max_new_tokens=100)[0]["generated_text"]
    return jsonify({"response": result})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
