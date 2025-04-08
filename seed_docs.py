# seed_docs.py
from pymongo import MongoClient
from voyageai import Client as VoyageClient
import os

client = MongoClient(os.environ["MONGODB_URI"])
db = client["your_db"]
collection = db["your_collection"]

voyage = VoyageClient(api_key=os.environ["VOYAGE_API_KEY"])

health_docs = [
    {"text": "High blood pressure often presents with headaches...", "category": "hypertension"},
    # add more docs...
]

for doc in health_docs:
    embedding = voyage.embed([doc["text"]], model="voyage-lite-02-instruct").embeddings[0]
    collection.insert_one({"text": doc["text"], "embedding": embedding, "category": doc["category"]})
