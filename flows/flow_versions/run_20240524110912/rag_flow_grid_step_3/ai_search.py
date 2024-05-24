from promptflow import tool
from langchain_community.vectorstores.azuresearch import AzureSearch
from langchain_openai import AzureOpenAIEmbeddings, OpenAIEmbeddings
from promptflow.connections import CustomConnection, CognitiveSearchConnection, AzureOpenAIConnection



@tool
def my_python_tool(
                question: str, 
                search_index_name:str,
                embedding_deployment:str,
                openai_conn: AzureOpenAIConnection, 
                search_conn:CognitiveSearchConnection) -> str:
    def format_doc(doc: dict):
        return f"Content: {doc.page_content}\n Source: {doc.metadata.get('source')}"
    # Perform a similarity search
    embeddings: AzureOpenAIEmbeddings = AzureOpenAIEmbeddings(
    azure_deployment=embedding_deployment,
    openai_api_version="2024-02-01",
    azure_endpoint=openai_conn.api_base,
    api_key=openai_conn.api_key,
    )


    vector_store: AzureSearch = AzureSearch(
        azure_search_endpoint=search_conn.api_base,
        azure_search_key=search_conn.api_key,
        index_name=search_index_name,
        embedding_function=embeddings.embed_query,
    )
    
    docs = vector_store.similarity_search(
    query=question,
    k=3,
    search_type="similarity"
    )

    doc_string = "\n\n".join([format_doc(doc) for doc in docs])
    
    return doc_string
