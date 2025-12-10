# AI-TTS-Streamer
Real time Text to speech Program using AI models to read docs/pdfs.

Convert PDFs to natural-sounding speech using a real-time producer–consumer TTS engine (Piper AI).

This project started as a simple pyttsx3 (robotic voice) reader, then evolved into a much more advanced low-latency streaming TTS system using Piper, supporting multiple high-quality models (English, Hindi, etc.).
Future versions may include Streamlit UI or even better models.

## 🚀 Features
## ✔ Real-time TTS streaming (producer–consumer architecture)

- Text is synthesized into audio chunks while streaming playback continues.

- Smooth, low-latency, natural speech.

- Handles long text without blocking.

## ✔ Supports multiple Piper voice models**

- English/ Hindi Language models

- Any Piper-compatible .onnx model

## ✔ PDF → Text → Audio**

- Reads PDF text line-by-line.

- Sends to the TTS engine in real time.

- Avoids memory spikes.

## ✔ Controls**

- Pause playback

- Resume playback

- Stop (immediately ends threads safely)

- Quit the whole application gracefully

## ✔ Thread-safe, robust design**

- Producer thread → generates audio

- Consumer thread → plays audio

* Avoids:
  - skipped lines

  - cut words

  - uneven gaps

  - deadlocks (stuck exit)

## 📁 Project Structure
`basic_tts/
│
├── src/
│   ├── piper_consumer_producer2.py     # main TTS engine
│   ├── pdf_reader.py                   # PDF text extraction (optional)
│
├── voice_models/                       # (ignored in git)
├── models/                             # (ignored in git)
│
├── output.wav
├── README.md
└── .gitignore`

## 🛠 Installation
1. Create virtual environment
   - `python -m venv venv`
   - `#Windows: venv\Scripts\activate`

3. Install dependencies
   - `   pip install -r requirements.txt`

4. Download Piper voice models

   - Place them inside:

     - `basic_tts/voice_models/`


### Example models:

- piper_voices_en_US_ljspeech_medium

- piper_voices_hi_in_priyamvada_medium

### ▶️ Usage
- Run the main TTS engine
  - `python piper_consumer_producer2.py`


- You will see runtime controls:

  - `[p]ause  [r]esume  [s]top  [q]uit:`

## 🎧 Architecture (Simplified)
`        +-------------------+
        |   PDF Reader      |
        +---------+---------+
                  |
                  v
        +---------+---------+
        |   Producer Thread |
        | (Piper synthesis) |
        +---------+---------+
                  |
        audio_queue (frames)
                  |
                  v
        +---------+---------+
        |   Consumer Thread |
        | (sounddevice out) |
        +-------------------+`
