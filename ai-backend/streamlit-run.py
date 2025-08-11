import main  # Import fungsi dari main.py
import streamlit as st
import time
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import os
import json
import hashlib
from IPython.display import display, Markdown

# =============== Konfigurasi Gemini + Chatbot ===============
API_KEY = "AIzaSyAXMr24XVP1ohfCO29GdM-9nm1IpBF_A_o"
try:
    genai.configure(api_key=API_KEY)
    print("✅ API Key configured!")
except Exception as e:
    print(f"❌ Gagal mengkonfigurasi API Key: {e}")
    exit()

my_generation_config = {
    "temperature": 0.5,
    "max_output_tokens": 4096,
    "top_p": 0.6
}
my_safety_settings = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
}
model = genai.GenerativeModel(
    model_name="models/gemini-2.5-pro",
    generation_config=my_generation_config,
    safety_settings=my_safety_settings
)


combined_txt_path = "source-chatbot.txt"
print("\n" + "="*50)
chatbot = main.TxtChatbot(model=model)
success = chatbot.load_from_combined_txt(combined_txt_path)
st.session_state.chatbot = chatbot


# =============== UI STREAMLIT ===============
st.set_page_config(page_title="Tanya DTSEN", page_icon="🤖", layout="wide")

# --- CSS untuk hover effect ---
st.markdown("""
<style>
a.doc-link, a.doc-link:visited {
    color: white !important;         /* Default putih */
    text-decoration: none !important; /* Hilangkan underline */
}
a.doc-link:hover {
    color: #2563eb !important;       /* Biru saat hover */
    text-decoration: underline !important; /* Underline saat hover */
}
</style>
""", unsafe_allow_html=True)

# --- Sidebar untuk daftar dokumen ---
st.sidebar.title("📂 Daftar Dokumen")
pdf_folder = os.path.abspath(os.path.join("..", "bahan-chatbot", "pdf"))
if os.path.exists(pdf_folder):
    pdf_files = [f for f in os.listdir(pdf_folder) if f.lower().endswith(".pdf")]
    if pdf_files:
        for pdf in sorted(pdf_files):
            display_name = os.path.splitext(pdf)[0].replace("_", " ").title()
            pdf_path = "file:../bahan-chatbot/pdf/" + os.path.join(pdf_folder, pdf).replace("\\", "/")
            st.sidebar.markdown(
                f"""
                <div style="margin-bottom:4px; font-size:14px;">
                    <a href="{pdf_path}" target="_blank" class="doc-link">
                        {display_name}
                    </a>
                </div>
                """,
                unsafe_allow_html=True
            )
    else:
        st.sidebar.info("Tidak ada dokumen PDF ditemukan.")
else:
    st.sidebar.error("Folder dokumen tidak ditemukan.")

# --- Garis pembatas ---
st.sidebar.markdown("---")

# --- State untuk konfirmasi reset ---
if "show_confirm_reset" not in st.session_state:
    st.session_state.show_confirm_reset = False

# --- CSS untuk tombol merah ---
st.markdown("""
<style>
div[data-testid="stSidebar"] button[kind="secondary"] {
    background-color: #dc2626 !important; /* Merah */
    color: white !important;
    border: none !important;
    border-radius: 6px !important;
    padding: 6px 12px !important;
}
div[data-testid="stSidebar"] button[kind="secondary"]:hover {
    background-color: #b91c1c !important; /* Merah lebih gelap saat hover */
}
</style>
""", unsafe_allow_html=True)

# --- Tombol reset awal ---
if not st.session_state.show_confirm_reset:
    if st.sidebar.button("🔄 Reset Chat"):
        st.session_state.show_confirm_reset = True
        st.rerun()
else:
    st.sidebar.warning("Yakin ingin menghapus semua riwayat chat?")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("✅ Ya"):
            st.session_state.messages = []
            st.session_state.show_confirm_reset = False
            st.rerun()
    with col2:
        if st.button("❌ Batal"):
            st.session_state.show_confirm_reset = False
            st.rerun()

# --- Main content ---
st.title("🤖 Tanya DTSEN")
st.markdown("""
Selamat datang di **Tanya DTSEN!**  
Aku adalah asisten virtual yang akan menjawab pertanyaanmu seputar **DTSEN**.  
""")

# --- CSS untuk bubble chat ---
st.markdown("""
<style>
.user-bubble {
    background-color: #2563eb !important; /* Biru */
    color: white !important;
    padding: 10px 14px !important;
    border-radius: 12px !important;
    max-width: 70% !important;
    margin-left: auto !important; /* Dorong ke kanan */
    margin-bottom: 8px !important;
    word-wrap: break-word !important;
}
.bot-bubble {
    background-color: #e5e7eb !important; /* Abu-abu muda */
    color: black !important;
    padding: 10px 14px !important;
    border-radius: 12px !important;
    max-width: 70% !important;
    margin-right: auto !important; /* Dorong ke kiri */
    margin-bottom: 8px !important;
    word-wrap: break-word !important;
}
</style>
""", unsafe_allow_html=True)

# Simpan riwayat chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Tampilkan riwayat chat
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f'<div class="user-bubble">{msg["text"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="bot-bubble">{msg["text"]}</div>', unsafe_allow_html=True)

# Input pertanyaan
if question := st.chat_input("Tulis pertanyaan kamu..."):
    # Simpan pertanyaan user
    st.session_state.messages.append({"role": "user", "text": question})
    st.markdown(f'<div class="user-bubble">{question}</div>', unsafe_allow_html=True)

    # Placeholder untuk animasi bot berpikir
    placeholder = st.empty()

    loading_text = "🤖 Sedang berpikir"
    for i in range(6):  # loop animasi titik berjalan
        dots = "." * (i % 4)
        placeholder.markdown(f'<div class="bot-bubble">{loading_text}{dots}</div>', unsafe_allow_html=True)
        time.sleep(0.4)

    # Ambil jawaban bot
    answer = st.session_state.chatbot.get_response(question)

    # Streaming teks per paragraf
    streamed_text = ""
    for paragraph in answer.split("\n\n"):
        streamed_text += paragraph + "\n\n"
        placeholder.markdown(f'<div class="bot-bubble">{streamed_text}</div>', unsafe_allow_html=True)
        time.sleep(0.4)

    # Simpan jawaban
    st.session_state.messages.append({"role": "assistant", "text": streamed_text.strip()})
