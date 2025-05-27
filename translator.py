from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

app = Flask(__name__)

# Cargar modelo y tokenizer
model_name = "facebook/nllb-200-distilled-600M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

@app.route("/", methods=["GET"])
def home():
    return "Translator running"

# Mapa simple de idiomas a códigos
lang_code_map = {
    "es": "spa_Latn",
    "en": "eng_Latn",
    "fr": "fra_Latn",
    "de": "deu_Latn",
    "it": "ita_Latn",
    "pt": "por_Latn",
    "ja": "jpn_Jpan",
    "zh": "zho_Hans"
}

@app.route("/translate", methods=["POST"])
def translate():
    data = request.get_json()
    text = data.get("text")
    src_lang = data.get("source_lang")
    tgt_lang = data.get("target_lang")

    if not text or not src_lang or not tgt_lang:
        return jsonify({"error": "Missing 'text', 'source_lang' or 'target_lang'"}), 400

    try:
        src_code = lang_code_map[src_lang]
        tgt_code = lang_code_map[tgt_lang]

        tokenizer.src_lang = src_code
        encoded = tokenizer(text, return_tensors="pt")
        generated_tokens = model.generate(
            **encoded,
            forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_code),
            max_length=256
        )
        translated = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
        return jsonify({"translation": translated})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
