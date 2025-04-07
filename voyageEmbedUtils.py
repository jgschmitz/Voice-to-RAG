from voyageai import Client as VoyageClient

voyage_client = VoyageClient(api_key="your_key")

def get_embedding(text, model="voyage-lite-02-instruct"):
    return voyage_client.embed([text], model=model).embeddings[0]

def batch_embed(docs, field="text"):
    return [
        {**doc, "embedding": get_embedding(doc[field])}
        for doc in docs
    ]
