import whisper

model = whisper.load_model("base")

def transcribe_audio(file_path):
    return model.transcribe(file_path)["text"]
