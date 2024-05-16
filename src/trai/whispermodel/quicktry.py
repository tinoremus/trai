import whisper

audio_file = r'/Users/tremus/Downloads/zwetschken.m4a'

model = whisper.load_model("base")
result = model.transcribe(audio_file)
print(result["text"])
