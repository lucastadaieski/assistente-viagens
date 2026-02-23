import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

modelo = ChatGoogleGenerativeAI(
    model="gemini-3-flash-preview", 
    temperature=0.5,
    google_api_key=api_key
)

prompt_consultor = ChatPromptTemplate.from_messages(
    [
        ("system", "Você é um consultor de viagens especialista em Brasil."),
        ("human", "{query}")
    ]
)

assistente = prompt_consultor | modelo | StrOutputParser()

print(assistente.invoke({"query": "Quero férias em praias no Brasil."}))