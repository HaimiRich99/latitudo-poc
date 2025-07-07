import json
#from retrieval import load_retriever
from pinecone_retrieval import load_retriever
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate


local = False
if local:
    from langchain_ollama import OllamaLLM
    print('✅ Using Ollama backend')
    llm = OllamaLLM(model="llama3.1")
else:
    from langchain_groq import ChatGroq
    print('✅ Using Groq backend')
    groq_api_key = "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"
    if not groq_api_key:
        raise ValueError("❌ GROQ_API_KEY not found in environment variables")
    llm = ChatGroq(model_name="llama3-8b-8192", api_key=groq_api_key)

# Pinecone api key 
# pcsk_6EaKWM_KZFEZQz97UgaRaRux3pMMGxqyEVq7gukSd6hY8FUS889J7Bn6qpWBrxyHmf1mAZ

retriever = load_retriever()
print('✅ Vector store loaded')

prompt_template = PromptTemplate.from_template("""
You are a geospatial assistant. Based on the user's query and the reference documents, extract the following:

- `location`: e.g., "Naples"
- `start_date`: e.g., "2023-07-01"
- `end_date`: optional
- `time_retrieval_type`: wether to retrieve a single date, a time-lapse or compare two dates ("single", "time-lapse", "compare")
- `frequencies`: list of relevant bands, asset key (NOTE: you need to write the asset names provided in the context docs, not generic band names)
- `frequency_operation`: wath to do when multiple bands are provided (None, "stack" for composites, "combine" for normaslized indexes)
- `cloud_cover`: optional (default 10)

DOCUMENTS:
{context}

QUERY:
{query}

RESPONSE FORMAT:

[Explanation and summary, backed by theory. Explain how you derived the parameters from the query and documents.]

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
    if local:
        response = llm.invoke(prompt)
    else:
        response = llm.invoke(prompt).content

    # Extract JSON from response
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

print("🟢 Groq LLM backend ready")
