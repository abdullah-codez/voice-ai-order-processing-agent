import numpy as np
from ten_vad import TenVad


SAMPLE_RATE = 16000
HOP_SIZE = 256
THRESHOLD = 0.6


class VoiceActivityDetector:
    def __init__(
        self,
        hop_size: int = HOP_SIZE,
        threshold: float = THRESHOLD,
    ):
        self.vad = TenVad(hop_size, threshold)

    def process(self, audio_chunk: np.ndarray) -> tuple[float, bool]:
        audio_int16 = np.clip(audio_chunk, -1.0, 1.0)
        audio_int16 = (audio_int16 * 32767).astype(np.int16)

        probability, flag = self.vad.process(audio_int16)

        return probability, bool(flag)