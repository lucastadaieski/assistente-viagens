import os
import time
from dotenv import load_dotenv

# Importações do LangChain e Google
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. Configurações Iniciais
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
NOME_BANCO_LOCAL = "banco_faiss_seguros"

# Configuração do Modelo de Chat (Gemini 1.5 Flash é o mais estável para RAG)
modelo = ChatGoogleGenerativeAI(
    model="models/gemini-2.5-flash", 
    temperature=0, 
    api_key=api_key
)

# Configuração dos Embeddings (Modelo que confirmamos estar disponível na sua conta)
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001", 
    google_api_key=api_key
)

# 2. Lógica de Carregamento e Processamento Inteligente
if os.path.exists(NOME_BANCO_LOCAL):
    print("✓ Banco de vetores local encontrado. Carregando...")
    vectorstore = FAISS.load_local(
        NOME_BANCO_LOCAL, 
        embeddings, 
        allow_dangerous_deserialization=True
    )
else:
    print("! Banco não encontrado. Iniciando leitura dos PDFs...")
    arquivos = [
        "documentos/GTB_standard_Nov23.pdf",
        "documentos/GTB_gold_Nov23.pdf",
        "documentos/GTB_platinum_Nov23.pdf"
    ]
    
    # Carregando os documentos
    docs_carregados = []
    for arquivo in arquivos:
        try:
            loader = PyPDFLoader(arquivo)
            docs_carregados.extend(loader.load())
            print(f"  - {arquivo} carregado.")
        except Exception as e:
            print(f"  - Erro ao carregar {arquivo}: {e}")

    # Divisão em pedaços (Chunks)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500, 
        chunk_overlap=150
    )
    pedacos = text_splitter.split_documents(docs_carregados)
    print(f"Total de pedaços gerados: {len(pedacos)}")

    # 3. Criação do Banco com Controle de Cota (Batching)
    print("Iniciando vetorização (Embedding) em lotes para respeitar o limite de 100 RPM...")
    
    tamanho_lote = 40  # Enviamos 40 por vez para ter margem de segurança
    # Criar o banco com o primeiro lote
    vectorstore = FAISS.from_documents(pedacos[:tamanho_lote], embeddings)
    
    # Adicionar os outros lotes com pausa
    for i in range(tamanho_lote, len(pedacos), tamanho_lote):
        print(f" -> Processando pedaços {i} até {i+tamanho_lote}...")
        lote_atual = pedacos[i : i + tamanho_lote]
        vectorstore.add_documents(lote_atual)
        
        print("    Aguardando 20 segundos para resetar cota do Google...")
        time.sleep(20) # Pausa necessária para o plano Free

    # Salva para uso futuro
    vectorstore.save_local(NOME_BANCO_LOCAL)
    print("✓ Banco de vetores criado e salvo com sucesso!")

# 4. Configuração do Retriever (Buscador)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 5. Prompt e Chain
template = ChatPromptTemplate.from_messages([
    ("system", "Você é um especialista em seguros. Responda APENAS com base no contexto fornecido."),
    ("human", "Pergunta: {query}\n\nContexto:\n{contexto}")
])

chain = template | modelo | StrOutputParser()

# 6. Função de Execução
def responder_seguro(pergunta: str):
    print(f"\nBuscando resposta para: {pergunta}")
    documentos_proximos = retriever.invoke(pergunta)
    contexto_texto = "\n\n".join([d.page_content for d in documentos_proximos])
    
    resposta = chain.invoke({
        "query": pergunta,
        "contexto": contexto_texto
    })
    return resposta

# Execução do Teste
if __name__ == "__main__":
    print("-" * 30)
    pergunta_teste = "Como devo proceder caso tenha um item comprado roubado e caso eu tenha o cartão gold?"
    try:
        resultado = responder_seguro(pergunta_teste)
        print(f"\nRESPOSTA:\n{resultado}")
    except Exception as e:
        print(f"\nErro ao gerar resposta: {e}")