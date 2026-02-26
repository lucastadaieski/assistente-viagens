import os
import asyncio
from dotenv import load_dotenv
from typing import Literal, TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableConfig

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

modelo = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,            
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# --- Definições de Tipo ---
class Rota(TypedDict):
    destino: Literal["praia", "montanha"]

class Estado(TypedDict):
    query: str
    destino: Rota
    resposta: str

# --- Esquema para o Gemini ---
esquema_rota = {
    "title": "roteador_viagem", 
    "type": "object",
    "properties": {
        "destino": {"type": "string", "enum": ["praia", "montanha"]}
    },
    "required": ["destino"]
}

# --- 1. PROMPTS DOS ESPECIALISTAS ---
prompt_praia = ChatPromptTemplate.from_messages([
    ("system", "Você é a Sra. Praia. Especialista em mar e sol. Se pedirem montanha, brinque que prefere o mar."),
    ("human", "{query}")
])

prompt_montanha = ChatPromptTemplate.from_messages([
    ("system", "Você é o Sr. Montanha. Especialista em frio, altitude e escalada radical. NUNCA sugira praias."),
    ("human", "{query}")
])

cadeia_praia = prompt_praia | modelo | StrOutputParser()
cadeia_montanha = prompt_montanha | modelo | StrOutputParser()

# --- 2. PROMPT DO ROTEADOR (VERSÃO XEQUE-MATE) ---
prompt_roteador = ChatPromptTemplate.from_messages([
    ("system", """Você é um sistema lógico de classificação. Não tente ser amigável.
    
    REGRAS DE CLASSIFICAÇÃO:
    1. Se a pergunta mencionar: 'escalar', 'montanha', 'pico', 'altitude', 'frio', 'neve', 'rocha' -> RETORNE 'montanha'.
    2. Se a pergunta mencionar: 'mar', 'praia', 'surf', 'onda', 'calor', 'areia', 'sol' -> RETORNE 'praia'.
    
    Pense passo a passo: O usuário quer subir algo ou quer ir ao nível do mar?
    Responda apenas com o JSON."""),
    ("human", "{query}")
])

# Criando o roteador estruturado
roteador = prompt_roteador | modelo.with_structured_output(esquema_rota)

# --- 3. NÓS DO GRAFO ---
async def no_roteador(estado: Estado, config: RunnableConfig):
    res = await roteador.ainvoke({"query": estado["query"]}, config)
    if not res or "destino" not in res:
        res = {"destino": "praia"}
    return {"destino": res}

async def no_praia(estado: Estado, config: RunnableConfig):
    res = await cadeia_praia.ainvoke({"query": estado["query"]}, config)
    return {"resposta": res}

async def no_montanha(estado: Estado, config: RunnableConfig):
    # CORRIGIDO: Agora chamando a cadeia de MONTANHA
    res = await cadeia_montanha.ainvoke({"query": estado["query"]}, config)
    return {"resposta": res}

def escolher_no(estado: Estado):
    decisao = estado["destino"]["destino"]
    print(f"\n[DEBUG] O Roteador classificou como: {decisao.upper()}")
    return decisao

# --- 4. CONSTRUÇÃO DO GRAFO ---
grafo = StateGraph(Estado)
grafo.add_node("rotear", no_roteador)
grafo.add_node("praia", no_praia)
grafo.add_node("montanha", no_montanha)

grafo.add_edge(START, "rotear")
grafo.add_conditional_edges("rotear", escolher_no)
grafo.add_edge("praia", END)
grafo.add_edge("montanha", END)

app = grafo.compile()

# --- 5. EXECUÇÃO ---
async def main():
    pergunta = "Quero escalar montanhas radicais"
    print(f"Pergunta: {pergunta}")
    
    resultado = await app.ainvoke({"query": pergunta})
    print(f"\nRESPOSTA FINAL:\n{resultado['resposta']}")

if __name__ == "__main__":
    asyncio.run(main())