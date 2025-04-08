#### 🧠 Voice-to-RAG (Multimodal Healthcare Assistant) 

### How it works
How It Works <br>
User speaks a question <br>
Audio is transcribed using Whisper <br>
The transcribed query is embedded via Voyage AI <br>
Embedding is used to run a hybrid search in MongoDB Atlas <br>
Top results are passed to OpenAI GPT-4 to generate the response <br>

Final answer is returned — and can be spoken back via TTS (optional)

``` python
!pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```
#### ---Then install Whisper---
``` python
!pip install -q git+https://github.com/openai/whisper.git
```
#### ---Make sure ffmpeg is available---
``` python
!apt-get update -y && apt-get install -y ffmpeg
```

#### --- Imports ---
``` python
import pymongo
from voyageai import Client as VoyageClient
import openai
import whisper
import datetime
import uuid
import logging
```

#### --- Setup Logging ---
``` python
logging.basicConfig(level=logging.INFO)
```

#### --- MongoDB Setup ---
``` python
mongo_client = pymongo.MongoClient("your_mongo_uri")  # Replace with your URI
db = mongo_client.voyagenew
collection = db.demo_rag
history_collection = db.voice_query_history
```
#### --- API Keys ---
``` python
voyage_client = VoyageClient(api_key="your_voyage_api_key")
openai.api_key = "your_openai_api_key"
```

#### --- Whisper Setup ---
``` python
whisper_model = whisper.load_model("base")  # can also try 'tiny' for faster inference
```
#### --- Step 1: Transcribe Audio ---
``` python
def transcribe_audio(file_path):
    result = whisper_model.transcribe(file_path)
    return result["text"]

# --- Step 2: Vector + Filter Search ---
def search_similar_docs(query_embedding, keyword=None, category=None, top_k=3):
    pipeline = [
        {
            "$vectorSearch": {
                "queryVector": query_embedding,
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

    try:
        return list(collection.aggregate(pipeline))
    except Exception as e:
        logging.error(f"Error during hybrid search: {e}")
        return []
```

#### --- Step 3: RAG Response Generation ---
``` python
def generate_rag_response(user_query, keyword=None, category=None):
    try:
        query_response = voyage_client.embed([user_query], model="voyage-lite-02-instruct")
        query_embedding = query_response.embeddings[0]

        retrieved_docs = search_similar_docs(query_embedding, keyword, category)
        if not retrieved_docs:
            return "No relevant info found.", []

        retrieved_texts = [doc["text"] for doc in retrieved_docs]
        retrieved_summary = "\n".join([f"- {text}" for text in retrieved_texts])

        prompt = f"""
        You are a healthcare assistant providing accurate responses.
        The user asked: \"{user_query}\"
        
        Relevant information:
        {retrieved_summary}

        Please generate a helpful, clear answer.
        """

        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful healthcare assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200
        )

        return response.choices[0].message['content'].strip(), retrieved_docs

    except Exception as e:
        logging.error(f"RAG error: {e}")
        return "An error occurred.", []
```

#### --- Step 4: Voice-to-RAG Pipeline ---
``` python
def voice_to_rag_pipeline(audio_file_path, keyword=None, category=None):
    session_id = str(uuid.uuid4())
    timestamp = datetime.datetime.utcnow()
    

    user_query = transcribe_audio(audio_file_path)
    logging.info(f"Voice Query: {user_query}")

    response, retrieved_docs = generate_rag_response(user_query, keyword, category)
    logging.info(f"Response: {response}")

    history_collection.insert_one({
        "session_id": session_id,
        "timestamp": timestamp,
        "audio_file": audio_file_path,
        "transcribed_query": user_query,
        "response": response,
        "category": category,
        "retrieved_docs": retrieved_docs
    })

    return response
```

####--- Step 5: Run with Sample Audio File ---
#### Upload an audio file named 'voice_question.mp3' to your environment 
``` python
from google.colab import files
Upload an audio file named 'voice_question.mp3' to your Colab environment

response = voice_to_rag_pipeline("voice_question.mp3", category="nutrition")
print("\n🎤 User's Question (from voice):", transcribe_audio("voice_question.mp3"))
print("\n💬 Assistant's Answer:", response)
```

