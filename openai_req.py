import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

numero_dias = 7
numero_criancas = 2
atividade = "praia"

modelo_de_prompt = PromptTemplate(
    template="""
    Crie um roteiro de viagem de {dias} dias,
    para uma família com {numero_criancas} crianças,
    que gostam de {atividade}
    """
)

prompt = modelo_de_prompt.format(
    dias = numero_dias,
    numero_criancas = numero_criancas,
    atividade = atividade
)

print("Prompt : \n", prompt)

modelo = ChatGoogleGenerativeAI(
    model="gemini-3-flash-preview",
    temperature=0.5,
    api_key=api_key
)

resposta = modelo.invoke(prompt)
print(resposta.content)