import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import MessagesPlaceholder
from langchain_community.llms import HuggingFaceHub
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

# Configuração do Streamlit
st.set_page_config(page_title="Cassio Bot 🤖", page_icon="🤖")
st.title("Cassio Bot 🤖")

# --- Barra Lateral para Seleção do Modelo ---
with st.sidebar:
    st.header("Configurações do Modelo")
    model_class = st.selectbox(
        "Escolha o provedor do modelo:",
        ("openai")
    )

    if model_class == "openai":
        model_name = st.selectbox(
            "Escolha o modelo OpenAI:",
            ("gpt-4o-mini", "gpt-3.5-turbo-0125", "gpt-3.5-turbo")
        )
        openai_api_token = st.text_input("OpenAI API Key:", type="password")

    temperature = st.slider("Temperatura:", min_value=0.0, max_value=1.0, value=0.1, step=0.01)

def model_openai(model_name, temperature, api_key):  # Added api_key parameter
    llm = ChatOpenAI(model=model_name, temperature=temperature, streaming=True, api_key=api_key)  # Pass api_key
    return llm


def model_response(user_query, chat_history, model_class, model_name, temperature, api_key=None):
    if model_class == "openai":
        if not api_key:  # Check if API key is provided
            raise ValueError("Para conversar comigo você precisa informar a chave API")
        llm = model_openai(model_name, temperature, api_key)  # Pass api_key
    else:
        raise ValueError(f"model_class inválido: {model_class}")

    # Prompts (Added explicit Portuguese instruction)
    system_prompt = """
        Seu nome é Cássio, um assistente virtual prestativo especializado em códigos e depuração.
        Responda TODAS as perguntas em mineirês, com dialetos mineiros sempre, inclusive o sotaque
        mineiro. Se alguém perguntar de onde veio, fale de onde você é. Não utilize muito o R, cortando
        isso no final.
    """
    language = "português-br" # More specific

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
            "language": lambda x: x['language']  # Keep this for consistency, even though it's in the system prompt
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

# Initialize chat history (outside the conditional)
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
        try:
            # Pass the API key to model_response
            for chunk in model_response(user_query, st.session_state["chat_history"], model_class, model_name, temperature, api_key=openai_api_token if model_class == 'openai' else None):
                full_response += chunk
                placeholder.markdown(full_response + "▌")
            placeholder.markdown(full_response)
        except ValueError as ve:  # Catch specific ValueError for missing API key
            st.error(str(ve))  # Display the ValueError message
        except Exception as e:
            st.error(f"Ocorreu um erro: {e}")  # More general error handling

        st.session_state["chat_history"].append(AIMessage(content=full_response))
