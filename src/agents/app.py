import os
import dotenv
import pymongo
import random
from typing import Annotated, TypedDict, Literal, Any, List, Dict
from pydantic import BaseModel, Field
from typing import  Literal
from typing_extensions import TypedDict

from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from azure.core.credentials import AzureKeyCredential

import streamlit as st

import tiktoken
from langchain_core.callbacks import BaseCallbackHandler
from openai import AzureOpenAI
from langchain_openai import AzureChatOpenAI
from langchain_openai import AzureOpenAIEmbeddings
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, BaseMessage
from langchain_core.runnables import RunnableLambda, RunnableWithFallbacks
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.graph import CurveStyle, MermaidDrawMethod, NodeStyles
from langchain_core.prompts import PromptTemplate
from langchain.agents import AgentExecutor, create_structured_chat_agent, create_react_agent
from langchain import agents

from langgraph.graph import MessagesState
from langgraph.graph.message import add_messages
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import AnyMessage, add_messages


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

def num_tokens_from_messages(messages: List[str]) -> int:
    '''
    Calculate the number of tokens in a list of messages. This is a somewhat naive implementation that simply concatenates 
    the messages and counts the tokens in the resulting string. A more accurate implementation would take into account the 
    fact that the messages are separate and should be counted as separate sequences.
    If available, the token count should be taken directly from the model response.
    '''
    encoding = tiktoken.encoding_for_model("gpt-4o")
    num_tokens = 0
    content = ' '.join(messages)
    num_tokens += len(encoding.encode(content))

    return num_tokens

class TokenCounterCallback(BaseCallbackHandler):
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def on_llm_new_token(self, token: str, **kwargs: Any) -> Any:
        self.completion_tokens += 1

    def on_chat_model_start(self, serialized: Dict[str, Any], messages: List[List[BaseMessage]], **kwargs: Any) -> Any:
        self.prompt_tokens += num_tokens_from_messages( [message.content for message in messages[0]])
         

callback = TokenCounterCallback()

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
openai: AzureOpenAI = None
embeddings_model: AzureOpenAIEmbeddings = None
mongodb_client = None

embedding_model = os.getenv("AZURE_OPENAI_EMBEDDING_MODEL")
if "AZURE_OPENAI_API_KEY" in os.environ:
    llm = AzureChatOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_deployment=os.getenv("AZURE_OPENAI_COMPLETION_DEPLOYMENT_NAME"),
        openai_api_version=os.getenv("AZURE_OPENAI_VERSION"),
        temperature=0,
        streaming=True,
        callbacks=[callback]
    )
    embeddings_model = AzureOpenAIEmbeddings(    
        azure_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"),
        openai_api_version = os.getenv("AZURE_OPENAI_VERSION"),
        model= os.getenv("AZURE_OPENAI_EMBEDDING_MODEL"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY")
    )
    openai = AzureOpenAI(
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
        streaming=True,
        callbacks=[callback]
    )
    embeddings_model = AzureOpenAIEmbeddings(    
        azure_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"),
        openai_api_version = os.getenv("AZURE_OPENAI_VERSION"),
        model= os.getenv("AZURE_OPENAI_EMBEDDING_MODEL"),
        azure_ad_token_provider = token_provider
    )
    openai = AzureOpenAI(
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

mongodb_client = pymongo.MongoClient(os.environ["AZURE_COSMOS_DB_CONNECTIONSTRING"])
mongodb_database = os.environ["AZURE_COSMOS_DB_DATABASE_NAME"]
db = mongodb_client[mongodb_database]
if mongodb_database not in mongodb_client.list_database_names():
    print("Created db '{}' with shared throughput.\n".format(mongodb_database))
else:
    print("Using database: '{}'.\n".format(mongodb_database))

def get_session_id() -> str:
    id = random.randint(0, 1000000)
    return "00000000-0000-0000-0000-" + str(id).zfill(12)

if "session_id" not in st.session_state:
    st.session_state["session_id"] = get_session_id()
    print("started new session: " + st.session_state["session_id"])
    st.write("You are running in session: " + st.session_state["session_id"])


class RecommendationResponse(BaseModel):
    """Response from the recommendation API"""
    name: str = Field(description="The name of the product")
    category: str = Field(description="The category of the product")
    description: str = Field(description="The description of the product")
    personalization: str = Field(description="The personalization of the product recommendation")
    price: float = Field(description="Your price of the product")
    question: str = Field(description="The question asked to the user for additional recommendations")

class AgentState(MessagesState):
    final_response: RecommendationResponse

def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])

# use an embeddingsmodel to create embeddings
def get_embedding(text, model=embedding_model):
    if len(text) == 0:
        return openai.embeddings.create(input = "no description", model=model).data[0].embedding
    return openai.embeddings.create(input = [text], model=model).data[0].embedding

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

model_with_tools = llm.bind_tools(tools)
model_with_structured_output = llm.with_structured_output(RecommendationResponse)

# Force the model to use tools by passing tool_choice="any"    
model_with_response_tool = llm.bind_tools(tools,tool_choice="auto")


# Define the function that calls the model
def call_model(state: AgentState):
    response = model_with_tools.invoke(state['messages'])
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}

# Define the function that responds to the user
def respond(state: AgentState):
    # We call the model with structured output in order to return the same format to the user every time
    # state['messages'][-2] is the last ToolMessage in the convo, which we convert to a HumanMessage for the model to use
    # We could also pass the entire chat history, but this saves tokens since all we care to structure is the output of the tool
    response = model_with_structured_output.invoke([HumanMessage(content=state['messages'][-2].content)])
    # We return the final answer
    return {"final_response": response}

# Define the function that determines whether to continue or not
def should_continue(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    # If there is no function call, then we respond to the user
    if not last_message.tool_calls:
        return "respond"
    # Otherwise if there is, we continue
    else:
        return "continue"

# Define a new graph
workflow = StateGraph(AgentState)

# Define the two nodes we will cycle between
workflow.add_node("agent", call_model)
workflow.add_node("respond", respond)
workflow.add_node("tools", ToolNode(tools))

# Set the entrypoint as `agent`
# This means that this node is the first one called
workflow.set_entry_point("agent")

# We now add a conditional edge
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "tools",
        "respond": "respond",
    },
)

workflow.add_edge("tools", "agent")
workflow.add_edge("respond", END)
graph = workflow.compile()

human_query = st.chat_input()

if human_query is not None and human_query != "":

    st.session_state.chat_history.append(HumanMessage(human_query))

    inputs = {
        "messages": [
            ("user", human_query),
        ]
    }

    for event in graph.stream(inputs):  
        for value in event.values():
            print(value)

            if ( "messages" in value):
                # check if there is a message in the response
                message = value["messages"][-1]
                if ( isinstance(message, AIMessage) ):
                    print("AI:", message.content)
                    with st.chat_message("Agent"):
                        if (message.content == ''):
                            toolusage = ''
                            for tool in message.tool_calls:
                                print(tool)
                                toolusage += "name: " + tool["name"] + "  \n\n"
                            st.write("Using the following tools: \n", toolusage)
                        else:
                            st.write(message.content)
                
                if ( isinstance(message, ToolMessage) ):
                    print("Tool:", message.content)
                    with st.chat_message("Tool"):
                        st.write(message.content.replace('\n\n', ''))

        st.write("The total number of tokens used in this conversation was: ", callback.completion_tokens + callback.prompt_tokens)
