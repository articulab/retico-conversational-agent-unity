import wave


path = f"wav_files/clause_{2}.wav"
# path = f"wav_files/clause_{2}.wav"
with wave.open(path, "wb") as wav_file:
    print(path)
    wav_file.setnchannels(self.channels)  # Set the number of channels
    wav_file.setsampwidth(self.samplewidth)  # Set the sample width in bytes
    wav_file.setframerate(self.tts_framerate)  # Set the frame rate (sample rate)
    wav_file.writeframes(full_data)  # Write the audio byte data
