import streamlit as st
from gpt4all import GPT4All
from deep_translator import GoogleTranslator
from langdetect import detect, DetectorFactory
from gtts import gTTS
import speech_recognition as sr
import tempfile
import base64
import os

DetectorFactory.seed = 0

MODEL_PATH = "gpt4all-falcon-newbpe-q4_0.gguf"
model = GPT4All(model_name=MODEL_PATH)

def detect_language(text):
    try:
        return detect(text)
    except:
        return "en" 

def translate_text(text, dest_lang="en"):
    try:
        return GoogleTranslator(source='auto', target=dest_lang).translate(text)
    except Exception as e:
        st.error(f"Translation error: {e}")
        return text  

def ask_health_bot(topic, prompt):
    system_prompt = (
        f"You are a multilingual healthcare assistant specialized in {topic}.\n"
        "Provide simple, educational health advice. Remind the user that this is not a substitute for professional medical care.\n"
    )
    full_prompt = f"{system_prompt}User: {prompt}\nAssistant:"
    with model.chat_session():
        response = model.generate(full_prompt, max_tokens=512)
    return response.strip()

def speak_text(text, lang="en"):
    try:
        tts = gTTS(text=text, lang=lang)
        tmp_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tts.save(tmp_path.name)
        return tmp_path.name
    except Exception as e:
        st.error(f"TTS error: {e}")
        return None

def record_and_transcribe():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("🎙️ Speak your question now...")
        audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
    try:
        st.info("🧠 Transcribing...")
        return recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        st.warning("Could not understand audio.")
        return ""
    except sr.RequestError as e:
        st.error(f"STT request error: {e}")
        return ""


st.set_page_config(page_title="Anemia & Pregnancy Healthcare Chatbot", page_icon="🩺")
st.title("Anemia & Pregnancy Healthcare Chatbot")
st.markdown("Ask your question in **any language**, by **voice or text**!")

# Topic selection
topic = st.selectbox("Choose your support topic:", ["Anemia", "Pregnancy Care"])

use_voice = st.checkbox(" Use Voice Input")
user_input = ""

if use_voice:
    if st.button(" Record Question"):
        user_input = record_and_transcribe()
        st.text_input("Transcribed Text:", user_input, key="voice_input")
else:
    user_input = st.text_input("Type your Question:")

# When user submits question
if user_input.strip():
    user_lang = detect_language(user_input)
    translated_input = translate_text(user_input, dest_lang="en")

    with st.spinner(" Thinking..."):
        response_en = ask_health_bot(topic, translated_input)

    response_user_lang = translate_text(response_en, dest_lang=user_lang)

    st.markdown(f"**Bot:** {response_user_lang}")
    with st.expander(" English Version"):
        st.markdown(response_en)


    if st.button("Speak Answer"):
        audio_path = speak_text(response_user_lang, lang=user_lang)
        if audio_path:
            audio_bytes = open(audio_path, "rb").read()
            st.audio(audio_bytes, format="audio/mp3", start_time=0)
        # os.remove(audio_path)


    st.info("⚠️ This chatbot provides educational health guidance. Always consult a doctor for medical concerns.")
