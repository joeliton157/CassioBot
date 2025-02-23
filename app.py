import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import MessagesPlaceholder
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

# --- Carregamento de Variáveis de Ambiente ---
# Use st.secrets se estiver no Streamlit Cloud, senão, load_dotenv
if 'OPENAI_API_KEY' in st.secrets:
    #Já no Streamlit Cloud
    openai_api_key = st.secrets['OPENAI_API_KEY']
else:
    #Local ou se a chave não estiver em st.secrets (teste)
    load_dotenv()
    openai_api_key = os.environ.get("OPENAI_API_KEY")


# Configuração do Streamlit
st.set_page_config(page_title="Cassio Bot 🤖", page_icon="🤖")
st.title("Cassio Bot 🤖")

# --- Barra Lateral para Seleção do Modelo ---
with st.sidebar:
    st.header("Configurações do Modelo")
    model_class = st.selectbox(
        "Escolha o provedor do modelo:",
        ("openai", "ollama")
    )

    if model_class == "openai":
        model_name = st.selectbox(
            "Escolha o modelo OpenAI:",
            ("gpt-4o", "gpt-3.5-turbo-0125", "gpt-3.5-turbo")  #Removi o gpt-4o-mini
        )
    elif model_class == "ollama":
        model_name = st.selectbox(
            "Escolha o modelo Ollama:",
            ("phi3", "llama3", "mistral", "deepseek-coder")
        )

    temperature = st.slider("Temperatura:", min_value=0.0, max_value=1.0, value=0.1, step=0.01)


# Funções de Criação de Modelo
def model_openai(model_name, temperature):
    # Usar a chave diretamente, obtida de st.secrets ou .env
    llm = ChatOpenAI(model=model_name, temperature=temperature, streaming=True, openai_api_key=openai_api_key)
    return llm

def model_ollama(model_name, temperature):
    llm = ChatOllama(model=model_name, temperature=temperature)
    return llm

def model_response(user_query, chat_history, model_class, model_name, temperature):
    # Carregamento da LLM
    if model_class == "openai":
        llm = model_openai(model_name, temperature)
    elif model_class == "ollama":
        llm = model_ollama(model_name, temperature)
    else:
        raise ValueError(f"model_class inválido: {model_class}")

    # Prompts
    system_prompt = """
        Seu nome é Cássio, um assistente virtual 
        prestativo e está respondendo perguntas 
        relacionadas a Logística e Supply Chain. Responda em {language}.
        Não responda nada fora do contexto logístico.
    """
    language = "português"

    # Template do Prompt
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}")
    ])

    # Criação da Chain
    chain = (
        {
            "input": RunnablePassthrough(),
            "chat_history": lambda x: x['chat_history'],
            "language": lambda x: x['language']
        }
        | prompt_template
        | llm
        | StrOutputParser()
    )
    # Retorna o Stream
    return chain.stream({
            "input": user_query,
            "chat_history": chat_history,
            "language": language
        })


if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = [
        AIMessage(content="Olá, eu sou o Cássio! Como posso te ajudar?")
    ]

for message in st.session_state["chat_history"]:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.markdown(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.markdown(message.content)

user_query = st.chat_input("Digite sua mensagem aqui...")
if user_query:
    st.session_state["chat_history"].append(HumanMessage(content=user_query))

    with st.chat_message("Human"):
        st.markdown(user_query)

    with st.chat_message("AI"):
        full_response = ""
        placeholder = st.empty()
        for chunk in model_response(user_query, st.session_state["chat_history"], model_class, model_name, temperature):
            full_response += chunk
            placeholder.markdown(full_response + "▌")
        placeholder.markdown(full_response)
        st.session_state["chat_history"].append(AIMessage(content=full_response))