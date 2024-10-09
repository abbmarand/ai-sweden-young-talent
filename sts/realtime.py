from RealtimeSTT import AudioToTextRecorder

def process_text(text):
    print(text)
recorder = AudioToTextRecorder()

recorder.start()
recorder.stop()
print(recorder.text())