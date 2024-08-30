import os
import dotenv
import random
from typing import Annotated, TypedDict, Literal, Any, List
from typing_extensions import TypedDict

from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from azure.core.credentials import AzureKeyCredential

import streamlit as st
from langchain_community.callbacks.streamlit import (
    StreamlitCallbackHandler,
)
from openai import AzureOpenAI
from langchain_openai import AzureChatOpenAI
from langchain_openai import AzureOpenAIEmbeddings
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.runnables import RunnableLambda, RunnableWithFallbacks
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.graph import CurveStyle, MermaidDrawMethod, NodeStyles
from langchain_core.prompts import PromptTemplate
from langchain.agents import AgentExecutor, create_structured_chat_agent, create_react_agent
from langchain import agents

from langgraph.graph.message import add_messages
from langgraph.graph import END, StateGraph, START
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.prebuilt import ToolNode

from azure.search.documents import SearchClient
from azure.search.documents.models import (
    VectorizedQuery
)

from IPython.display import Image

dotenv.load_dotenv()

st.set_page_config(
    page_title="AI agentic bot that to recommend products"
)

st.title("💬 AI shopping copilot")
st.caption("🚀 A Bot that can recommend the right products for you")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.markdown(message.content)
    elif isinstance(message, ToolMessage):
        with st.chat_message("Tool"):
            st.markdown(message.content)
    else:
        with st.chat_message("Agent"):
            st.markdown(message.content)

llm: AzureChatOpenAI = None
client: AzureOpenAI = None
embeddings_model: AzureOpenAIEmbeddings = None
embedding_model = os.getenv("AZURE_OPENAI_EMBEDDING_MODEL")
if "AZURE_OPENAI_API_KEY" in os.environ:
    llm = AzureChatOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_deployment=os.getenv("AZURE_OPENAI_COMPLETION_DEPLOYMENT_NAME"),
        openai_api_version=os.getenv("AZURE_OPENAI_VERSION"),
        temperature=0,
        streaming=True
    )
    embeddings_model = AzureOpenAIEmbeddings(    
        azure_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"),
        openai_api_version = os.getenv("AZURE_OPENAI_VERSION"),
        model= os.getenv("AZURE_OPENAI_EMBEDDING_MODEL"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY")
    )
    client = AzureOpenAI(
        api_key = os.getenv("AZURE_OPENAI_API_KEY"),  
        api_version = os.getenv("AZURE_OPENAI_VERSION"),
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    )
else:
    token_provider = get_bearer_token_provider(DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default")
    llm = AzureChatOpenAI(
        azure_ad_token_provider=token_provider,
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        azure_deployment=os.getenv("AZURE_OPENAI_COMPLETION_DEPLOYMENT_NAME"),
        openai_api_version=os.getenv("AZURE_OPENAI_VERSION"),
        temperature=0,
        openai_api_type="azure_ad",
        streaming=True
    )
    embeddings_model = AzureOpenAIEmbeddings(    
        azure_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"),
        openai_api_version = os.getenv("AZURE_OPENAI_VERSION"),
        model= os.getenv("AZURE_OPENAI_EMBEDDING_MODEL"),
        azure_ad_token_provider = token_provider
    )
    client = AzureOpenAI(
        azure_ad_token_provider=token_provider,
        api_version = os.getenv("AZURE_OPENAI_VERSION"),
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    )

credential = AzureKeyCredential(os.environ["AZURE_AI_SEARCH_KEY"]) if len(os.environ["AZURE_AI_SEARCH_KEY"]) > 0 else DefaultAzureCredential()

search_client = SearchClient(
    endpoint=os.environ["AZURE_AI_SEARCH_ENDPOINT"], 
    index_name=os.environ["AZURE_AI_SEARCH_INDEX"],
    credential=credential
)

def get_session_id() -> str:
    id = random.randint(0, 1000000)
    return "00000000-0000-0000-0000-" + str(id).zfill(12)

if "session_id" not in st.session_state:
    st.session_state["session_id"] = get_session_id()
    print("started new session: " + st.session_state["session_id"])
    st.write("You are running in session: " + st.session_state["session_id"])


def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])

# use an embeddingsmodel to create embeddings
def get_embedding(text, model=embedding_model):
    if len(text) == 0:
        return client.embeddings.create(input = "no description", model=model).data[0].embedding
    return client.embeddings.create(input = [text], model=model).data[0].embedding

@tool
def search_for_product(question: str) -> str:
    """This will return more detailed information about the products from the product repository Returns top 5 results."""
    # create a vectorized query based on the question
    vector = VectorizedQuery(vector=get_embedding(question), k_nearest_neighbors=5, fields="vector")

    found_docs = list(search_client.search(
        search_text=None,
        query_type="semantic", query_answer="extractive",
        query_answer_threshold=0.8,
        semantic_configuration_name="products-semantic-config",
        vector_queries=[vector],
        select=["id", "name", "description", "category", "price"],
        top=12
    ))

    print(found_docs)
    found_docs_as_text = " "
    for doc in found_docs:   
        print(doc) 
        found_docs_as_text += " "+ "Name: {}".format(doc["name"]) +" "+ "Description: {}".format(doc["description"]) +" "+ "Price: {}".format(doc["price"]) + "Category: {}".format(doc["category"]) +" "

    return found_docs_as_text

@tool
def get_last_purchases(user_id: Annotated[int, "the user id, which is int. Example 105"]) -> List[str]:
    "Returns last 5 purchases of a customer, as List of strings. Call this tool with the user_id which is only a number."
    print("Getting last purchases for userId: ", user_id)
    return ["Lego City Police Station","Ultra-Thin Mechanical Keyboard"]

@tool
def get_user_info(jwtToken: str) -> str:
    "Returns current user/customers information. Name, Address is returned seperated by semi-colon. This function needs the current jwt from the user as parameter."
    return "Name: Dennis; Address: Microsoft Street 1."

@tool
def get_user_id(jwtToken: str) -> int:
    "Returns current user/customers user_id. This function needs the current jwt from the user as parameter."
    return 1234

@tool
def get_jwt() -> int:
    "Returns current user/customers jwt."
    return " ABC1345"

tools = [get_jwt, get_user_id, get_user_info, search_for_product, get_last_purchases]


prompt_template_v1 = """\
Assistant is designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. As a language model, Assistant is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.

You are talking to customers. Users and customers are the same thing for you.

Assistant is able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions. Additionally, Assistant is able to generate its own text based on the input it receives, allowing it to engage in discussions and provide explanations and descriptions on a wide range of topics.

Overall, Assistant is a powerful tool that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. 

TOOLS:

------

Assistant has access to the following tools:

{tools}

To use a tool, please use the following format:

```

Thought: Do I need to use a tool? Yes

Action: the action to take, should be one of [{tool_names}]

Action Input: the input to the action

Observation: the result of the action

```

When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

```

Thought: Do I need to use a tool? No

Final Answer: [your response here]

```

Begin!

Previous conversation history:

{chat_history}

New input: {input}

{agent_scratchpad}
"""

prompt = PromptTemplate.from_template(prompt_template_v1)
agent = create_react_agent(llm, tools, prompt)

agent_executor = agents.AgentExecutor(
        name="Tools Agent",
        agent=agent, tools=tools, verbose=True, handle_parsing_errors=True, max_iterations=10, return_intermediate_steps=True, 
        # handle errors
        error_message="I'm sorry, I couldn't understand that. Please try again.",
    )
 

human_query = st.chat_input()

if human_query is not None and human_query != "":

    st.session_state.chat_history.append(HumanMessage(human_query))

    with st.chat_message("assistant"):
        st_callback = StreamlitCallbackHandler(st.container())
        response = agent_executor.invoke(
            {"input": human_query, "chat_history": st.session_state.chat_history}, {"callbacks": [st_callback]}, 
        )

        ai_response = st.write(response["output"])
