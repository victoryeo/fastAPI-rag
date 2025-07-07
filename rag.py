from dotenv import load_dotenv
import os
#from langchain_huggingface import HuggingFaceEndpoint
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
import bs4
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
from langsmith import Client
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from validation import QueryInput
from pydantic import ValidationError

load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

print("token",os.getenv("GROQ_API_KEY"))

# Define state for application
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

# chat model
"""llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-v0.2",
    max_length=64,
    temperature=0.1,
    task="text-generation",  #HuggingFaceEndpoint Requires Explicit task Argument
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
)"""
llm = ChatGroq(temperature=0.7, model_name="llama-3.3-70b-versatile")

# embeddings model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# vector store
vector_store = InMemoryVectorStore(embeddings)

vector_store_ready = False

def vector_store_exists():
    return vector_store_ready

def build_vector_store():
    # Load and chunk contents of the blog
    loader = WebBaseLoader(
        web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_=("post-content", "post-title", "post-header")
            )
        ),
    )
    # load documents
    docs = loader.load()    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    # split documents
    all_splits = text_splitter.split_documents(docs)

    # stored indexed chunks
    _ = vector_store.add_documents(documents=all_splits)

    vector_store_ready = True

def answer_query(query):
    try:
        user_input = QueryInput(question=query)
    except ValidationError as e:
        result = {}
        result["answer"] = str(e)
        result["sources"] = 'Validation'
        return result
    # Define prompt for question-answering
    #client = Client(api_key=os.getenv("LANGSMITH_API_KEY"))
    #prompt = client.pull_prompt("rlm/rag-prompt", include_model=True)
    #docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    #messages = prompt.invoke({"question": state["question"], "context": docs_content})

    # Define prompt for question-answering
    template = """Question: {question}

    Answer: Let's think step by step."""
    prompt = PromptTemplate.from_template(template)      

    # Define application steps
    def retrieve(state: State):
        retrieved_docs = vector_store.similarity_search(state["question"])
        return {"context": retrieved_docs}

    def generate(state: State):
        llm_chain = prompt | llm
        response = llm_chain.invoke(state["question"])
        return {"answer": response}

    # retrieval and generation
    graph_builder = StateGraph(State).add_sequence([retrieve, generate])
    graph_builder.add_edge(START, "retrieve")
    # Compile application 
    graph = graph_builder.compile()

    print("query", user_input.question)
    # invoke using langgraph
    response = graph.invoke({"question": user_input.question})
    print(response["answer"])
    result = {}
    # Extract 'content' if present
    answer = response["answer"]
    if hasattr(answer, 'content'):
        result["answer"] = answer.content
    else:
        result["answer"] = answer
    result["sources"] = 'LLMBot'

    return result