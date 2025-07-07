from langchain.chains import RetrievalQA
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from retrieval import load_retriever
import json

print('ok1')

llm = OllamaLLM(model="llama3.1")
retriever = load_retriever()

prompt_template = PromptTemplate.from_template("""
You are a geospatial assistant. Based on the user's query and the reference documents, extract the following:

- `location`: e.g., "Naples"
- `start_date`: e.g., "2023-07-01"
- `end_date`: optional
- `frequencies`: list of relevant bands, indices (e.g., ["NDVI", "B04", "B08"])
- `cloud_cover`: optional (default 20)

                                               
DOCUMENTS:
{context}

QUERY:
{query}

RESPONSE FORMAT:
                                               
[Explanation and summary, backed by theory.]

Extracted parameters in JSON format (without any additional text):                        
{{
  "location": "...",
  "start_date": "...",
  "end_date": "...",
  "frequencies": [...],
  "cloud_cover": ...
}}
""")

def parse_query_with_rag(user_query):
    docs = retriever.invoke(user_query)
    context = "\n\n".join([d.page_content for d in docs[:2]])
    prompt = prompt_template.format(context=context, query=user_query)
    response = llm.invoke(prompt)

    # Isolate the JSON part in response, The user is given text while the json is sent to the backend
    json_start = response.find('{')
    json_end = response.rfind('}') + 1
    if json_start == -1 or json_end == -1 or json_start >= json_end:
        json_response = None
    
    json_response = response[json_start:json_end].strip()
    text_response = response[:json_start].strip() + response[json_end:].strip()

    try:
        return json.loads(json_response), text_response, None
    except:
        return None, None, "❌ Could not parse response but the following information was retrieved:\n" + response

print('ok')