# re-usuable pipeline
from embed_utils import get_embedding
from mongo_utils import hybrid_vector_search
import openai

openai.api_key = "your_openai_key"

def generate_rag_response(user_query, collection_name="demo_rag", category=None, keyword=None):
    query_embedding = get_embedding(user_query)
    docs = hybrid_vector_search(collection_name, query_embedding, category, keyword)
    
    if not docs:
        return "No relevant info found."

    context = "\n".join([f"- {d['text']}" for d in docs])
    prompt = f"""User asked: {user_query}\n\nRelevant info:\n{context}\n\nGive a concise helpful answer."""

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200
    )

    return response.choices[0].message['content'].strip()
