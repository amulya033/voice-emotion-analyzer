"""Quick mic diagnostic — run this to check if sounddevice can hear you."""
import sounddevice as sd
import numpy as np

SAMPLE_RATE = 16_000

print("Available audio devices:")
print(sd.query_devices())
print()
print(f"Default input device: {sd.query_devices(kind='input')['name']}")
print()
print("Recording 3 seconds — speak now...")

audio = sd.rec(int(3 * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1,
               dtype="float32", blocking=True)

rms = np.sqrt(np.mean(audio ** 2))
peak = np.max(np.abs(audio))

print(f"\nDone.")
print(f"  RMS  : {rms:.6f}")
print(f"  Peak : {peak:.6f}")

if peak < 0.0001:
    print("\n  !! Almost no signal — mic may be muted, wrong device, or gain is 0.")
elif rms < 0.001:
    print(f"\n  Low signal. Set SILENCE_RMS below {rms:.4f} in emotion_recognition.py")
else:
    print(f"\n  Mic is working. Set SILENCE_RMS = {rms/4:.5f} in emotion_recognition.py")
