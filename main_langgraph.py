import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

modelo = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,            
    google_api_key=api_key
)

prompt_consultor_praia = ChatPromptTemplate.from_messages(
    [
        ("system", "Apresente-se como Sra Praia. Você é uma especialista em viangens com destinos para praia no Brasil."),
        ("human", "{query}")
    ]
)

prompt_consultor_montanha = ChatPromptTemplate.from_messages(
    [
        ("system", "Apresente-se como Sr Montanha. Você é uma especialista em viangens com destinos para montanhas e atividades radicais."),
        ("human", "{query}")
    ]
)

cadeia_praia = prompt_consultor_praia | modelo | StrOutputParser()
cadeia_montanha = prompt_consultor_montanha | modelo | StrOutputParser()

class Rota(BaseModel):
    destino: Literal["praia", "montanha"] = Field(description="O destino escolhido pelo usuário")

prompt_roteador = ChatPromptTemplate.from_messages([
    ("system", "Você é um classificador. Responda se o usuário quer praia ou montanha."),
    ("human","{query}")
])

roteador = prompt_roteador | modelo.with_structured_output(Rota)

def responda(pergunta: str):
    resultado = roteador.invoke({"query": pergunta})
    rota = resultado.destino
    
    print(f"--- Rota: {rota} ---")
    
    if rota == "praia":
        return cadeia_praia.invoke({"query": pergunta})
    return cadeia_montanha.invoke({"query": pergunta})

if __name__ == "__main__":
    print(responda("Quero surfar em um lugar quente"))