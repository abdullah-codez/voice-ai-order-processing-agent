import numpy as np

class SpeechSegmenter:
    def __init__(self, silence_duration_ms: int = 600, min_speech_duration_ms: int = 200):
        self.silence_threshold = silence_duration_ms / 1000.0
        # Calculate the minimum number of audio frames required to be considered valid speech
        self.min_speech_frames = int((min_speech_duration_ms / 1000.0) * 16000)
        self.buffer = []
        self.is_recording = False
        self.silence_start_time = None
        self.current_speech_frames = 0
        
    def process(self, audio_chunk: np.ndarray, is_speech: bool) -> np.ndarray | None:
        if is_speech:
            self.is_recording = True
            self.silence_start_time = None
            self.buffer.append(audio_chunk)
            self.current_speech_frames += len(audio_chunk)
            return None
            
        if self.is_recording:
            self.buffer.append(audio_chunk)
            if self.silence_start_time is None:
                self.silence_start_time = 0
            self.silence_start_time += (len(audio_chunk) / 16000.0)
            
            if self.silence_start_time >= self.silence_threshold:
                # Acoustic Debouncing Gate
                if self.current_speech_frames < self.min_speech_frames:
                    self.reset()
                    return None
                    
                utterance = np.concatenate(self.buffer)
                self.reset()
                return utterance
                
        return None

    def reset(self):
        self.buffer = []
        self.is_recording = False
        self.silence_start_time = None
        self.current_speech_frames = 0