from promptflow.core import tool
from promptflow.connections import CustomConnection
from langchain_community.chat_models.azureml_endpoint import (
    AzureMLEndpointApiType,
    CustomOpenAIChatContentFormatter,
)
from langchain_community.chat_models.azureml_endpoint import AzureMLChatOnlineEndpoint
from langchain_core.messages.human import HumanMessage
from langchain_core.messages.system import SystemMessage




@tool
def my_python_tool(question: str, context: str, conn: CustomConnection) -> str:
    #model_family = myconn.model_family
    endpoint_url = conn.endpoint_url
    api_key = conn.endpoint_api_key

    chat= AzureMLChatOnlineEndpoint(
    endpoint_url=endpoint_url+'/v1/chat/completions',
    endpoint_api_type=AzureMLEndpointApiType.serverless,
    endpoint_api_key=api_key,
    content_formatter=CustomOpenAIChatContentFormatter()
    )

    system_message="""
    You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. 
    If you don't know the answer, just say that you don't know.
    """
    template=f"""
    Question: {question}
    Context: {context}
    Answer:"""
    try:
        response = chat.invoke(
            [   SystemMessage(content=system_message),
                HumanMessage(content=template)]
        )
        return response.content
    
    except Exception as e:
        return str(e)