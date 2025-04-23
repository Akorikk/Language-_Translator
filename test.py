# app.py
import streamlit as st
from transformers import AutoTokenizer, TFAutoModelForSeq2SeqLM

# Load model and tokenizer
model_path = "tf_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = TFAutoModelForSeq2SeqLM.from_pretrained(model_path)

# Set language codes (for M2M100 or similar models)
source_lang = "en"
target_lang = "hi"

st.title("English to Hindi Translator 🌐")

# Input text box
input_text = st.text_area("Enter English text:", "")

if st.button("Translate"):
    if input_text.strip() == "":
        st.warning("Please enter some text to translate.")
    else:
        # Tokenize input
        inputs = tokenizer(input_text, return_tensors="tf", padding=True, truncation=True, src_lang=source_lang)
        
        # Generate output
        translated = model.generate(**inputs, forced_bos_token_id=tokenizer.lang_code_to_id[target_lang])
        output = tokenizer.batch_decode(translated, skip_special_tokens=True)[0]

        st.subheader("Translated Text (Hindi):")
        st.success(output)
