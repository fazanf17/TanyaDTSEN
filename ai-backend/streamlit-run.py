import main  # Import fungsi dari main.py
import streamlit as st
import time
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import os
import json
import hashlib
from IPython.display import display, Markdown

if "chatbot" not in st.session_state:
    st.session_state.chatbot = main.init_chatbot()


# =============== UI STREAMLIT ===============
st.set_page_config(page_title="Tanya DTSEN", page_icon="🤖", layout="wide")

# --- CSS untuk hover effect ---
st.markdown("""
<style>
a.doc-link, a.doc-link:visited {
    text-decoration: none !important; /* Hilangkan underline */
}
a.doc-link:hover {
    color: #2563eb !important;       /* Biru saat hover */
    text-decoration: underline !important; /* Underline saat hover */
}
</style>
""", unsafe_allow_html=True)

# --- Sidebar untuk daftar dokumen ---
st.sidebar.markdown("Tanya DTSEN adalah chatbot AI berbasis RAG yang membantu memahami segala hal tentang DTSEN (Data Tunggal Sosial Ekonomi Nasional)")
st.sidebar.title("📂 Daftar Dokumen")

# Daftar dokumen sudah terdefinisi
dokumen_list = [
    ("Bahan Ajar Groundcheck DTSEN", "https://drive.google.com/file/d/1ffTqe2HK7uZms4DSAiTRBfexc5I3vIrg/view?usp=sharing"),
    ("Bahan Kemensos Panel", "https://drive.google.com/file/d/1WI496xHu2ILiSri0TniC9I2STjMJwyBG/view?usp=sharing"),
    ("Briefing Groundcheck Provinsi", "https://drive.google.com/file/d/1ZWc60P_LZ6402OQGLeSKXbskaD8r6Jjt/view?usp=sharing"),
    ("Inpres Nomor 4 Tahun 2025", "https://drive.google.com/file/d/1susFuD75VrK0hWWon1UO60h6VSXYV8fS/view?usp=sharing"),
    ("Kesimpulan Rapat Koordinasi Nasional Penggunaan DTSEN", "https://drive.google.com/file/d/1J-b80FKCD654vtmYI24BKH4sh-TKYq-T/view?usp=sharing"),
    ("Paparan Kebijakan DTSEN Rakornas", "https://drive.google.com/file/d/1U2wLFEQoDrNSI7eha_MP9tEZ_mIhNpZS/view?usp=sharing"),
    ("Pemeringkatan DTSEN Kemensos", "https://drive.google.com/file/d/1sqsyoFk3Rvy_wvq8-6ABlX7iHeay5e7D/view?usp=sharing"),
    ("Permensos No 3 Tahun 2025", "https://drive.google.com/file/d/1KFUB2N-KshkB_wwcJzqPDEAEy93rlWMO/view?usp=sharing"),
    ("Rakornas Dinsos", "https://drive.google.com/file/d/1RFuHz8jFyB9Oe-dQKF_0nY82d4kruOc1/view?usp=sharing"),
    ("Rakornas Kemensos", "https://drive.google.com/file/d/1OFDpc3_dpLwPSDwpo8HTp6JEYrLWJEez/view?usp=sharing"),
]


for nama, link in dokumen_list:
    st.sidebar.markdown(
        f"""
        <div style="margin-bottom:4px; font-size:14px;">
            <a href="{link}" target="_blank" class="doc-link">
                {nama}
            </a>
        </div>
        """,
        unsafe_allow_html=True
    )

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

