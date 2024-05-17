from promptflow.core import tool
from promptflow.connections import CustomConnection
from langchain_community.chat_models.azureml_endpoint import (
    AzureMLEndpointApiType,
    CustomOpenAIChatContentFormatter,
)
from langchain_community.chat_models.azureml_endpoint import AzureMLChatOnlineEndpoint
from langchain_core.messages.human import HumanMessage


@tool
def my_python_tool(message: str, conn: CustomConnection) -> str:
    #model_family = myconn.model_family
    endpoint_url = conn.endpoint_url
    api_key = conn.endpoint_api_key

    chat= AzureMLChatOnlineEndpoint(
    endpoint_url=endpoint_url+'/v1/chat/completions',
    endpoint_api_type=AzureMLEndpointApiType.serverless,
    endpoint_api_key=api_key,
    content_formatter=CustomOpenAIChatContentFormatter(),
)
    try:
        response = chat.invoke(
            [HumanMessage(content=message)]
        )
        return response.content
    
    except Exception as e:
        return str(e)