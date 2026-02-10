import os
import sys
import io
from dotenv import load_dotenv
from google import genai


os.environ["PYTHONIOENCODING"] = "utf-8"
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

load_dotenv()
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

try:
    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents="Gere um roteiro de 1 dia sobre música no Brasil. Responda com acentos."
    )
    
    print("\n" + "="*50)
    print(response.text)
    print("="*50)

except Exception as e:
    if "503" in str(e):
        print("Servidor do Google lotado. Tente de novo em 10 segundos!")
    else:
        print(f"Erro: {e}")
        