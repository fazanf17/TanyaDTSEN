import streamlit as st
import time
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import os
import json
import hashlib
from IPython.display import display, Markdown  # Untuk Jupyter/Colab


PROMPT_TEMPLATES = {
    "single_chunk_qa": """Anda adalah Asisten AI Analis Dokumen yang sangat teliti.
                        Aturan utama Anda:
                        1. JAWAB HANYA berdasarkan informasi dari <dokumen> yang diberikan.
                        2. JANGAN menambahkan informasi, asumsi, atau pengetahuan eksternal.
                        3. Jawaban harus dalam Bahasa Indonesia yang ringkas dan jelas.
                        4. Batasi jawaban Anda MAKSIMAL 200 Kata saja.
                        5. Jika informasi tidak ditemukan dalam dokumen, jawab dengan: "Informasi tidak ditemukan dalam sumber yang dimiliki"

                        <dokumen>pip
                        {chunk}
                        </dokumen>

                        Pertanyaan: {user_question}

                        Jawaban Langsung dan Ringkas:""",

    "extractor": """Anda adalah Asisten AI yang handal dalam mengekstraksi dokumen yang diberikan. Dari bagian dokumen berikut ekstrak semua informasi yang relevan dengan
                    pertanyaan: "{user_question}". Fokus hanya pada informasi yang ada pada dokumen dan menjawab pertanyaan. Jika tidak ada yang relevan katakan "Tidak ada informasi yang relevan".
                    Kemudian jika pertanyaan: "{user_question}" berupa slang indonesia seperti ucapan terimakasih atau tidak ada keterkaitannya dengan informasi pada sumber maka cukup katakan
                    "Saya sulit memahami pertanyaan anda".

                    <dokumen_bagian>
                    {chunk}
                    </dokumen_bagian>

                    Informasi Relevan:""",

    "synthesizer": """Anda adalah asisten AI yang ahli dalam merangkum informasi. Berdasarkan kumpulan informasi berikut, maka rangkum informasi tersebut sehingga menjawab pertanyaan pengguna.
                    Perlu diingat Aturan utama yang harus anda penuhi :
                    1. Gabungkan informasi yang relevan dan ringkas untuk memberikan jawaban yang padu dan relevan dengan pertanyaan.
                    2. Jawaban harus dalam Bahasa Indonesia yang jelas dan menyesuaikan gaya bahasa pengguna.
                    3. JAWAB SECARA LANGSUNG dan SINGKAT, hindari menggunakan kalimat pembuka atau penutup yang tidak perlu.
                    4. Batasi jawaban anda tidak lebih dari 200 kata, kecuali diminta.
                        <informasi_terkumpul>
                        {combined_info}
                        </informasi_terkumpul>
                        Pertanyaan Pengguna: {user_question}
                        Jawaban Akhir yang Ringkas:"""
}

# ======== chatbot ========
class TxtChatbot:
    def __init__(self, model):
        """Inisialisasi Chatbot yang membaca dari file teks."""
        self.model = model
        self.source_text = None
        self.data_source_name = None
        print(f"✅ TxtChatbot berhasil diinisialisasi dengan model '{model.model_name}'!")

    def load_from_combined_txt(self, combined_txt_path):
        """Memuat seluruh teks dari satu file .txt gabungan."""
        self.data_source_name = os.path.basename(combined_txt_path)
        print(f"📂 Membaca sumber data utama dari: '{self.data_source_name}'")
        try:
            with open(combined_txt_path, 'r', encoding='utf-8') as f:
                self.source_text = f.read()
            if not self.source_text.strip():
                print("⚠️ Peringatan: File sumber data kosong.")
                return False
            
            print("✅ Sumber data berhasil dimuat.")
            return True
        except FileNotFoundError:
            print(f"❌ File sumber data tidak ditemukan. Jalankan proses pembaruan terlebih dahulu.")
            return False

    def get_info(self):
        """Menampilkan statistik dari teks yang dimuat."""
        if not self.source_text:
            print("❌ Belum ada data yang dimuat.")
            return
        lines = self.source_text.count('\n') + 1
        words = len(self.source_text.split())
        chars = len(self.source_text)
        info = (f"**📊 INFORMASI SUMBER DATA**\n"
                f"- 📄 **Sumber:** {self.data_source_name}\n"
                f"- 📝 **Total karakter:** {chars:,}\n"
                f"- 🗣️ **Total kata:** {words:,}\n"
                f"- 📄 **Total baris:** {lines:,}")
        try:
            display(Markdown(info))
        except NameError:
            print(info.replace('**', ''))

    def chunk_text(self, text, max_length=100000):
        """Memecah teks menjadi beberapa bagian jika terlalu panjang."""
        if len(text) <= max_length:
            return [text]
        
        chunks, words = [], text.split()
        current_chunk, current_length = [], 0
        for word in words:
            word_length = len(word) + 1
            if current_length + word_length > max_length:
                if current_chunk: chunks.append(" ".join(current_chunk))
                current_chunk, current_length = [word], word_length
            else:
                current_chunk.append(word)
                current_length += word_length
        if current_chunk: chunks.append(" ".join(current_chunk))
        print(f"📝 Teks sumber terlalu besar, dibagi menjadi {len(chunks)} bagian untuk dianalisis.")
        return chunks
    
    def _call_model(self, prompt: str) -> str:
        """Fungsi helper terpusat untuk memanggil model dan menangani respons/error."""
        try:
            response = self.model.generate_content(prompt)
            # Menggunakan short-circuiting untuk mengembalikan teks atau pesan error
            return response.text if response.parts else "❌ Respons diblokir oleh filter keamanan."
        except Exception as e:
            # Menangani error API atau lainnya saat pemanggilan model
            return f"❌ Terjadi kesalahan saat memanggil model: {e}"

    def get_response(self, user_question: str) -> str:
        """Menghasilkan jawaban berdasarkan teks yang dimuat menggunakan prompt templates."""
        if not self.source_text:
            return "❌ Belum ada data yang dimuat. Harap jalankan `load_from_combined_txt` terlebih dahulu."
        
        print(f"🤖 Memproses pertanyaan: {user_question}")
        chunks = self.chunk_text(self.source_text)
        
        # Kasus 1: Teks cukup pendek (hanya 1 chunk)
        if len(chunks) == 1:
            prompt = PROMPT_TEMPLATES["single_chunk_qa"].format(
                chunk=chunks[0], 
                user_question=user_question
            )
            return self._call_model(prompt)

        # Kasus 2: Teks panjang (beberapa chunk), gunakan strategi Map-Reduce
        else:
            relevant_info = []
            print(f"📊 Menganalisis {len(chunks)} bagian teks...")
            for i, chunk in enumerate(chunks):
                print(f"⏳ Mengekstrak info dari bagian {i+1}/{len(chunks)}...", end='\r')
                extract_prompt = PROMPT_TEMPLATES["extractor"].format(
                    user_question=user_question, 
                    chunk=chunk
                )
                response_text = self._call_model(extract_prompt)
                
                # Tambahkan hanya jika respons berisi dan bukan pesan 'tidak relevan'
                if response_text and "tidak ada informasi relevan" not in response_text.lower():
                    relevant_info.append(response_text)
            
            print("\n✅ Ekstraksi selesai.")

            if not relevant_info:
                return "Informasi yang relevan dengan pertanyaan Anda tidak ditemukan di dalam dokumen."
            
            # Gabungkan (Reduce) informasi relevan dan buat jawaban akhir
            combined_info = "\n\n---\n\n".join(relevant_info)
            synthesis_prompt = PROMPT_TEMPLATES["synthesizer"].format(
                combined_info=combined_info, 
                user_question=user_question
            )
            
            print("✍️  Merangkum informasi untuk jawaban akhir...")
            return self._call_model(synthesis_prompt)
        
if __name__ == "__main__":

    # Konfigurasi Gemini Models
    API_KEY = "AIzaSyD_2j6h_4TcXk8roZPeEE5NW0bPaQRsnQo"
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
    chatbot = TxtChatbot(model=model)
    success = chatbot.load_from_combined_txt(combined_txt_path)
