import numpy as np


class SpeechSegmenter:
    def __init__(
        self,
        sample_rate: int = 16_000,
        frame_duration_ms: int = 16,
        silence_duration_ms: int = 900,
        min_speech_duration_ms: int = 400,
    ):
        self.sample_rate = sample_rate

        self.silence_frames_required = max(
            1,
            silence_duration_ms // frame_duration_ms,
        )

        self.min_speech_samples = int(
            sample_rate * min_speech_duration_ms / 1000
        )

        self.audio_buffer: list[np.ndarray] = []
        self.silence_frame_count = 0
        
        # NEW: Track exactly how many samples were classified as speech
        self.actual_speech_samples = 0 
        self.is_speaking = False

    def process(
        self,
        audio_chunk: np.ndarray,
        is_speech: bool,
    ) -> np.ndarray | None:

        if is_speech:
            if not self.is_speaking:
                self.is_speaking = True
                self.silence_frame_count = 0
                self.actual_speech_samples = 0

            self.audio_buffer.append(audio_chunk.copy())
            self.silence_frame_count = 0
            self.actual_speech_samples += len(audio_chunk)  # Count actual speech

            return None

        # Silence before speech has started
        if not self.is_speaking:
            return None

        # Silence after speech has started
        self.audio_buffer.append(audio_chunk.copy())
        self.silence_frame_count += 1

        # Not enough silence yet to end the utterance
        if self.silence_frame_count < self.silence_frames_required:
            return None

        return self._finish_utterance()

    def _finish_utterance(self) -> np.ndarray | None:
        audio = np.concatenate(self.audio_buffer)

        # Store this before resetting state
        total_speech_detected = self.actual_speech_samples

        self.audio_buffer.clear()
        self.silence_frame_count = 0
        self.actual_speech_samples = 0
        self.is_speaking = False

        # NEW: Check if the ACTUAL speech was shorter than our 400ms minimum
        if total_speech_detected < self.min_speech_samples:
            return None

        return audio