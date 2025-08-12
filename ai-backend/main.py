import streamlit as st
import time
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from google.api_core import exceptions as google_exceptions
import os
import json
import hashlib
from IPython.display import display, Markdown  # Untuk Jupyter/Colab
import concurrent.futures
from typing import Union
import numpy as np
from numpy.linalg import norm

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
                    dan HANYA bersumber pada dokumen yang diberikan. Hindari penggunaan kata-kata dengan konotasi negatif atau kekerasan.
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
                        
                        <riwayat_percakapan>
                        {conversation_history}
                        </riwayat_percakapan>
                        
                        <informasi_terkumpul_untuk_pertanyaan_baru>
                        {combined_info}
                        </informasi_terkumpul_untuk_pertanyaan_baru>

                        Pertanyaan Baru Pengguna: {user_question}
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

        #inisiasi cache & history
        self.semantic_cache = []
        self.history= []

        #define threshold kemiripan (rekomendasi tuning 0.9 - 0.98)
        self.SIMILARITY_THRESHOLD = 0.96
        
        if self.models:
            print(f"✅ TxtChatbot berhasil diinisialisasi dengan model utama: '{self.get_current_model().model_name}'!")
        else:
            print("❌ Gagal menginisialisasi model. Pastikan nama model dan API key valid.")

    #fungsi cosine similarity between 2 vectors
    def _get_cosine_similarity(self, vec1, vec2):
        """Menghitung kemiripan kosinus antara dua vektor embedding."""
        return np.dot(vec1, vec2) / (norm(vec1) * norm(vec2))
    
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
        
    #metode baru ini menambahkan model dan menangani fallback dengan model rotation/ distribution
    def _extract_info_from_chunk(self, chunk_index: int, chunk: str, user_question: str) -> Union[str, None]:
        """
        Fungsi pekerja yang memanggil model untuk satu chunk dengan fallback.
        """
        extract_prompt = PROMPT_TEMPLATES["extractor"].format(
            user_question=user_question, 
            chunk=chunk
        )
        
        # Coba dengan model yang ditetapkan, lalu fallback jika gagal
        # Model rotation/distribution happens here
        model_index = chunk_index % len(self.models) 
        
        for i in range(len(self.models)):
            current_model_index = (model_index + i) % len(self.models)
            model_to_try = self.models[current_model_index]
            try:
                response_text = self._call_model(extract_prompt, model_to_try)
                if response_text and "tidak ada informasi relevan" not in response_text.lower():
                    return response_text
                return None # Berhasil tapi tidak relevan
            except (google_exceptions.ResourceExhausted, google_exceptions.InternalServerError, google_exceptions.ServiceUnavailable):
                print(f"🔄 Gagal dengan {model_to_try.model_name}, mencoba model fallback berikutnya...")
                continue # Lanjut ke model berikutnya dalam loop
                
        print(f"❌ Gagal mengekstrak info dari chunk setelah mencoba semua model.")
        return None

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
    
    def _call_model(self, prompt: str, model) -> str:
        """Fungsi terpusat untuk memanggil model yang spesifik."""
        print(f"🧠 Mencoba menghasilkan respons dengan model: {model.model_name}...")
        try:
            response = model.generate_content(prompt)
            return response.text if response.parts else "❌ Respons diblokir oleh filter keamanan."
        except google_exceptions.ResourceExhausted as e:
            print(f"⚠️ Model '{model.model_name}' mencapai limit penggunaan. Ini akan ditangani oleh fallback.")
            raise e # Lemparkan kembali error agar logika pemanggil bisa menanganinya
        except (google_exceptions.InternalServerError, google_exceptions.ServiceUnavailable) as e:
            print(f"❌ Terjadi gangguan server pada model '{model.model_name}'.")
            raise e
        except Exception as e:
            print(f"❌ Terjadi kesalahan tak terduga dengan model '{model.model_name}': {e}")
            raise e
        
    def _call_model_with_fallback(self, prompt: str) -> str:
        """
        Versi _call_model yang menggunakan dan memodifikasi state class (untuk single chunk & synthesizer)
        """
        if not self.models:
            return "❌ Tidak ada model yang bisa digunakan."

        current_model = self.get_current_model()
        try:
            # Panggil versi 'stateless'
            return self._call_model(prompt, current_model)
        except (google_exceptions.ResourceExhausted, google_exceptions.InternalServerError, google_exceptions.ServiceUnavailable):
            if self._switch_to_next_model():
                return self._call_model_with_fallback(prompt)  # Coba lagi dengan model baru
            else:
                return "Maaf, semua model sedang bermasalah atau mencapai batas penggunaan. Coba lagi nanti."
        except Exception as e:
            if self._switch_to_next_model():
                return self._call_model_with_fallback(prompt)
            else:
                return f"Maaf, terjadi kesalahan yang tidak dapat diatasi: {e}"


    def get_response(self, user_question: str) -> str:
        """
        Menghasilkan respons dengan alur kerja lengkap:
        1. Pengecekan Semantic Cache untuk pertanyaan serupa.
        2. Jika cache miss, jalankan proses RAG (ekstraksi-sintesis).
        3. Simpan hasil baru ke cache dan history percakapan.
        """
        if not self.source_text:
            return "❌ Belum ada data yang dimuat."

        print(f"\n🤖 Memproses pertanyaan: '{user_question}'")
        question_embedding = None # Inisialisasi variabel embedding

        # --- LANGKAH 1: PENGECEKAN SEMANTIC CACHE ---
        try:
            # Buat embedding untuk pertanyaan baru menggunakan model embedding khusus.
            question_embedding = genai.embed_content(
                model="models/text-embedding-004",
                content=user_question,
                task_type="RETRIEVAL_QUERY"
            )["embedding"]

            # Cari di cache semantik dengan iterasi
            for cached_embedding, cached_question, cached_answer in self.semantic_cache:
                similarity = self._get_cosine_similarity(question_embedding, cached_embedding)
                
                # Jika kemiripan melebihi ambang batas, anggap sebagai cache hit
                if similarity >= self.SIMILARITY_THRESHOLD:
                    print(f"✅ Semantic cache HIT! (Kemiripan: {similarity:.2f})")
                    print(f"   L Merespons dengan jawaban untuk pertanyaan serupa: '{cached_question}'")
                    
                    # Tetap simpan ke history percakapan agar konteks tidak hilang
                    self.history.append((user_question, cached_answer))
                    if len(self.history) > 5:
                        self.history.pop(0) # Batasi history
                    return cached_answer

            print("... Cache miss. Memproses sebagai pertanyaan baru.")

        except Exception as e:
            print(f"⚠️ Gagal melakukan pengecekan semantic cache: {e}. Melanjutkan tanpa cache.")
        
        # --- LANGKAH 2: PROSES RAG JIKA CACHE MISS ---
        chunks = self.chunk_text(self.source_text)
        
        # Kasus 1: Teks cukup pendek (hanya 1 chunk), tidak perlu RAG kompleks
        if len(chunks) == 1:
            print("   (Analisis single-chunk...)")
            self.current_model_index = 0 
            prompt = PROMPT_TEMPLATES["single_chunk_qa"].format(chunk=chunks[0], user_question=user_question)
            answer = self._call_model_with_fallback(prompt)

        # Kasus 2: Teks panjang (beberapa chunk), gunakan strategi Map-Reduce (ekstraksi-sintesis)
        else:
            relevant_info = []
            print(f"📊 Menganalisis {len(chunks)} bagian teks secara paralel dengan distribusi model...")
            
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future_to_chunk = {
                    executor.submit(self._extract_info_from_chunk, i, chunk, user_question): chunk 
                    for i, chunk in enumerate(chunks)
                }
                
                for i, future in enumerate(concurrent.futures.as_completed(future_to_chunk)):
                    print(f"⏳ Menyelesaikan ekstraksi bagian {i+1}/{len(chunks)}...", end='\r')
                    try:
                        result = future.result()
                        if result:
                            relevant_info.append(result)
                    except Exception as exc:
                        print(f'\n❌ Chunk menghasilkan exception: {exc}')

            print("\n✅ Ekstraksi paralel selesai.")

            if not relevant_info:
                return "Informasi yang relevan dengan pertanyaan Anda tidak ditemukan di dalam dokumen."
            
            # Format riwayat percakapan untuk dimasukkan ke dalam prompt synthesizer
            history_str = "\n".join([f"Pengguna: {q}\nAsisten: {a}" for q, a in self.history])
            if not history_str:
                history_str = "Tidak ada riwayat percakapan sebelumnya."

            combined_info = "\n\n---\n\n".join(relevant_info)
            synthesis_prompt = PROMPT_TEMPLATES["synthesizer"].format(
                conversation_history=history_str,
                combined_info=combined_info, 
                user_question=user_question
            )
            
            print("✍️ Merangkum informasi untuk jawaban akhir dengan konteks...")
            self.current_model_index = 0
            answer = self._call_model_with_fallback(synthesis_prompt)
        
        # --- LANGKAH 3: SIMPAN HASIL BARU KE CACHE & HISTORY ---
        # Hanya simpan ke cache jika proses pembuatan embedding di awal berhasil
        if question_embedding:
            print("✅ Jawaban berhasil dibuat. Menyimpan ke semantic cache.")
            self.semantic_cache.append((question_embedding, user_question, answer))
            # Batasi ukuran cache agar tidak membengkak
            if len(self.semantic_cache) > 20: 
                self.semantic_cache.pop(0)

        # Selalu simpan ke history percakapan
        self.history.append((user_question, answer))
        if len(self.history) > 5:
            self.history.pop(0)

        return answer
        
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
    # if chatbot:
    #     chatbot.get_info()