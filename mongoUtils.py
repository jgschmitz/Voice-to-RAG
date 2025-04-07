from pymongo import MongoClient

client = MongoClient("your_mongo_uri")
db = client.voyagenew

def insert_document(collection_name, doc):
    return db[collection_name].insert_one(doc)

def hybrid_vector_search(collection_name, vector, category=None, keyword=None, top_k=3):
    pipeline = [
        {
            "$vectorSearch": {
                "queryVector": vector,
                "path": "embedding",
                "numCandidates": 50,
                "limit": top_k,
                "index": "vector_index"
            }
        }
    ]
    if keyword or category:
        match_filter = {}
        if keyword:
            match_filter["text"] = {"$regex": keyword, "$options": "i"}
        if category:
            match_filter["category"] = category
        pipeline.append({"$match": match_filter})
    
    return list(db[collection_name].aggregate(pipeline))
