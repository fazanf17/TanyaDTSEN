import streamlit as st
import time
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from google.api_core import exceptions as google_exceptions
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

    "extractor": """Anda AI Ahli Dalam Ekstraksi Dokumen, ringkas dokumen yang saya berikan dengan fokus pada informasi yang akan menjawab {user_question}
                    dan HANYA bersumber pada dokumen yang diberikan. 
                    Jika jawaban tidak ditemukan atau pertanyaan tidak valid (seperti sapaan/slang), cukup jawab: "Tidak ada informasi yang relevan". 
                    Jangan pernah menambahkan informasi atau membuat asumsi apa pun di luar konteks dokumen ini.

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
    def __init__(self, model_names: list, generation_config: dict, safety_settings: dict):
        """
        Inisialisasi Chatbot dengan daftar nama model untuk fallback.
        """
        self.model_names = model_names
        self.generation_config = generation_config
        self.safety_settings = safety_settings
        self.models = self._initialize_models()  # Buat instance semua model
        self.current_model_index = 0
        self.source_text = None
        self.data_source_name = None
        
        if self.models:
            print(f"✅ TxtChatbot berhasil diinisialisasi dengan model utama: '{self.get_current_model().model_name}'!")
        else:
            print("❌ Gagal menginisialisasi model. Pastikan nama model dan API key valid.")

    def _initialize_models(self) -> list:
        """Membuat instance model generatif untuk setiap nama model yang diberikan."""
        models = []
        for name in self.model_names:
            try:
                model = genai.GenerativeModel(
                    model_name=name,
                    generation_config=self.generation_config,
                    safety_settings=self.safety_settings
                )
                models.append(model)
                print(f"   - Model '{name}' berhasil dimuat.")
            except Exception as e:
                print(f"⚠️ Peringatan: Gagal memuat model '{name}'. Error: {e}")
        return models

    def get_current_model(self):
        """Mengembalikan objek model yang sedang aktif."""
        if not self.models:
            return None
        return self.models[self.current_model_index]

    def _switch_to_next_model(self) -> bool:
        """
        Beralih ke model berikutnya dalam daftar. Mengembalikan True jika berhasil,
        False jika semua model sudah dicoba.
        """
        next_index = self.current_model_index + 1
        if next_index < len(self.models):
            self.current_model_index = next_index
            print(f"🔄 Beralih ke model fallback: {self.get_current_model().model_name}")
            return True
        else:
            print("❌ Semua model dalam daftar telah dicoba dan gagal.")
            return False
        
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

    def chunk_text(self, text, max_length=10000):
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
        """
        Fungsi terpusat untuk memanggil model. Menangani error dan mencoba
        model fallback secara rekursif jika terjadi kegagalan.
        """
        if not self.models:
            return "❌ Tidak ada model yang bisa digunakan."
        

        current_model = self.get_current_model()
        print(f"🧠 Mencoba menghasilkan respons dengan model: {current_model.model_name}...")

        try:
            response = current_model.generate_content(prompt)
            return response.text if response.parts else "❌ Respons diblokir oleh filter keamanan."

        except google_exceptions.ResourceExhausted as e:
            print(f"⚠️ Model '{current_model.model_name}' mencapai limit penggunaan.")
            if self._switch_to_next_model():
                return self._call_model(prompt)  # Coba lagi dengan model baru
            else:
                return "Maaf, semua model sedang mencapai batas penggunaan. Silakan coba lagi nanti."

        except (google_exceptions.InternalServerError, google_exceptions.ServiceUnavailable) as e:
            print(f"❌ Terjadi gangguan server pada model '{current_model.model_name}'.")
            if self._switch_to_next_model():
                return self._call_model(prompt)
            else:
                return "Maaf, layanan sedang mengalami gangguan teknis pada semua model. Silakan coba lagi nanti."

        except Exception as e:
            print(f"❌ Terjadi kesalahan tak terduga dengan model '{current_model.model_name}': {e}")
            if self._switch_to_next_model():
                 return self._call_model(prompt)
            else:
                return f"Maaf, terjadi kesalahan yang tidak dapat diatasi setelah mencoba semua model yang tersedia."

    def get_response(self, user_question: str) -> str:
        """Menghasilkan jawaban berdasarkan teks yang dimuat menggunakan prompt templates."""
        if not self.source_text:
            return "❌ Belum ada data yang dimuat. Harap jalankan `load_from_combined_txt` terlebih dahulu."
        
        #reset model
        self.current_model_index = 0
        print(f"\n🤖 Memproses pertanyaan: '{user_question}' (Mulai dengan model: {self.get_current_model().model_name})")
        
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
        
API_KEY = "AIzaSyAXMr24XVP1ohfCO29GdM-9nm1IpBF_A_o"

AVAILABLE_MODELS = [
    "models/gemini-2.5-pro",
    "models/gemini-2.5-flash",
    "models/gemini-2.0-flash",
    "models/gemini-2.0-flash-001",
    "models/gemini-2.0-flash-lite-001",
    "models/gemini-2.0-flash-lite",
    "models/gemini-1.5-flash", 
    "models/gemini-1.5-pro",  
    "models/gemini-1.5-flash-latest"
]

# def init_model():
#     try:
#         genai.configure(api_key=API_KEY)
#         print("✅ API Key configured!")
#     except Exception as e:
#         print(f"❌ Gagal mengkonfigurasi API Key: {e}")
#         exit()

#     my_generation_config = {
#         "temperature": 0.5,
#         "max_output_tokens": 4096,
#         "top_p": 0.6
#     }
#     my_safety_settings = {
#         HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
#         HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
#         HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
#         HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
#     }

#     return genai.GenerativeModel(
#         model_name=AVAILABLE_MODELS,
#         generation_config=my_generation_config,
#         safety_settings=my_safety_settings
#     )

def init_chatbot():
    """
    Fungsi ini mengkonfigurasi API key dan menginisialisasi TxtChatbot
    dengan daftar model dan pengaturannya.
    """
    try:
        genai.configure(api_key=API_KEY)
        print("✅ API Key configured!")
    except Exception as e:
        print(f"❌ Gagal mengkonfigurasi API Key: {e}")
        return None # Return None jika gagal

    # Definisikan konfigurasi di sini
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

    BASE_DIR = os.path.dirname(__file__)
    combined_txt_path = os.path.join(BASE_DIR, "source-chatbot.txt")
    print("Looking for:", combined_txt_path, "exists:", os.path.exists(combined_txt_path))

    # Inisialisasi TxtChatbot dengan benar
    # Berikan daftar nama model, bukan objek model
    cb = TxtChatbot(
        model_names=AVAILABLE_MODELS, 
        generation_config=my_generation_config,
        safety_settings=my_safety_settings
    )
    
    success = cb.load_from_combined_txt(combined_txt_path)
    print("load_from_combined_txt returned:", success)

    return cb

if __name__ == "__main__":
    # Pastikan chatbot berhasil diinisialisasi sebelum digunakan
    chatbot = init_chatbot()
    if chatbot:
        # Lakukan sesuatu dengan chatbot, misalnya:
        chatbot.get_info()
        # response = chatbot.get_response("Apa itu PDB?")
        # print(response)