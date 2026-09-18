import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

# .env dosyasındaki ortam değişkenlerini yüklüyoruz
load_dotenv()

# API anahtarını çevre değişkeninden alıyoruz
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("Hata: GEMINI_API_KEY bulunamadi! Lutfen .env dosyasini kontrol edin.")
    exit()

# Gemini istemcisini baslatiyoruz
client = genai.Client(api_key=api_key)

# Botun rolunu ve davranisini belirleyen sistem komutu
system_prompt = (
    "Sen yardimsever ve samimi bir yazilim mentorusun. "
    "Sorulara anlasilir, net ve orneklerle cevap ver."
)

# Sohbet gecmisi ve baglam penceresi ayari
# Token tasarrufu icin yalnizca son 6 mesaji (3 soru, 3 cevap) hafizada tutuyoruz
chat_history = []
MAX_MESSAGES = 6

def ask_bot(user_message):
    global chat_history

    # Kullanici mesajini gecmise ekle
    chat_history.append(
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=user_message)]
        )
    )

    # Baglam penceresi yonetimi (Context window management - Truncation)
    if len(chat_history) > MAX_MESSAGES:
        chat_history = chat_history[-MAX_MESSAGES:]

    # API cagrisi ve hata yakalama (Error handling)
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=chat_history,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.7,
            )
        )

        bot_reply = response.text

        # Botun cevabini da hafizaya ekliyoruz
        chat_history.append(
            types.Content(
                role="model",
                parts=[types.Part.from_text(text=bot_reply)]
            )
        )

        return bot_reply

    except Exception as error:
        # Hata olursa son eklenen kullanici mesajini listeden cikarip uyaralim
        chat_history.pop()
        return f"Bir hata olustu: {error}"

def main():
    print("--- Yapay Zeka Sohbet Botu Baslatildi ---")
    print("Cikmak icin 'q' veya 'exit' yazabilirsiniz.\n")

    while True:
        try:
            user_input = input("Sen: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ["exit", "q", "quit"]:
                print("Gorusmek uzere!")
                break

            cevap = ask_bot(user_input)
            print(f"Bot: {cevap}\n")

        except KeyboardInterrupt:
            print("\nProgram sonlandirildi.")
            break

if __name__ == "__main__":
    main()
