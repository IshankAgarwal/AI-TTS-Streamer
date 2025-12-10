import threading
import queue
import time
import numpy as np
import sounddevice as sd
import soundfile as sf
from PyPDF2 import PdfReader
from piper.voice import PiperVoice
import os
from tqdm import tqdm
import logging


"""
================== SimpleTTSStreamer - Major Issues & Resolutions ==================

1. Skipping whole sentences / jumping to next line
   - Cause: Producer thread generating frames too fast; audio queue fills up (maxsize small).
   - Resolution: Increased queue size to hold more frames; ensured proper backpressure handling.

2. Cutting off the last word of a sentence
   - Cause: Incorrect frame chunking / queue timing.
   - Resolution: Checked frame slicing and queue put logic; ensured full sentence frames are enqueued before signaling END_LINE.

3. Uneven gaps between sentences
   - Cause: Miscalculation of gap times; backpressure sleep accumulated; not accounting for synthesis & playback durations.
   - Resolution: Added precise logging of perf_counter at key points; computed gaps as differences between actual playback end and next start; frame sleep adjusted correctly.

4. Exit / stop / stuck issues
   - Cause: 
       - Threads (producer / consumer) blocked on queue.get() and waiting indefinitely.
       - Stop flag `_stop` alone insufficient if queues have pending items.
       - Using STOP_SIGNAL incorrectly or not processed timely.
       - Logging can sometimes delay thread exit (writing large logs synchronously).
   - Resolution:
       - Keep `_stop` flag **and** special `STOP_SIGNAL` object in queues.
       - Ensure consumer breaks on `_stop` **or** STOP_SIGNAL.
       - Use small timeout in queue.put/get to allow periodic stop check.
       - Add `stream.stop()` and `stream.close()` in stop() and at end of consumer to release audio resources.
       - Proper thread join with small timeout after stop.

5. Pausing then quitting caused hang
   - Cause: Consumer stuck in paused loop.
   - Resolution: Always check `_stop` inside pause loop; break instead of continue when stopping.

6. Logging to file affecting shutdown
   - Cause: Buffered writes can block if daemon threads exit abruptly.
   - Resolution: Use proper stop & close logic; daemon=False for threads to allow graceful shutdown; flush logs before exit.

7. Using END_LINE and STOP_SIGNAL objects
   - Use `END_LINE = object()` to signal sentence completion.
   - Use `STOP_SIGNAL = object()` to force thread exit.
   - Always check with `is` operator, **not** equality (`==`) to avoid numpy array ambiguity.

8. Stream handling
   - Cause: Leaving `sd.OutputStream` open leads to PortAudio thread blocking Python shutdown.
   - Resolution: Explicitly call `stream.stop()` and `stream.close()` in `stop()` and at consumer thread exit.

9. Thread joining & daemon usage
   - Use daemon=False to ensure clean shutdown with join().
   - On pressing 'q', call stop() then join threads with small timeout.
   - Do not rely solely on daemon=True; may exit abruptly and leave resources hanging.

10. Optional enhancements
    - Track current line being spoken for logging or UI display (Streamlit highlighting).
    - Adjust frame size / queue size to balance latency vs smooth playback.

====================================================================================
"""



# Create logger
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# ---- Console Handler (shows INFO and above, cleaner) ----
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)       # only show important logs in console
console_format = logging.Formatter(
    "%(asctime)s | %(threadName)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)
console_handler.setFormatter(console_format)

# ---- File Handler (shows EVERYTHING) ----
file_handler = logging.FileHandler("debug.log", mode="w", encoding="utf-8")
file_handler.setLevel(logging.DEBUG)         # save ALL logs into file
file_format = logging.Formatter(
    "%(asctime)s | %(threadName)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
file_handler.setFormatter(file_format)

# ---- Add handlers to root logger ----
logger.addHandler(console_handler)
logger.addHandler(file_handler)

# logging.basicConfig(
#     level=logging.DEBUG,q

#     format="%(asctime)s | %(threadName)s | %(levelname)s | %(message)s",
#     datefmt="%H:%M:%S"
# )


is_running = True
is_paused = False
is_stopped = False
lock = threading.Lock()

# current_play_event = threading.Event()
def extract_sentences_from_pdf(pdf_path, start_page, start_line):
    reader = PdfReader(open(pdf_path, "rb"))
    all_sentences = []

    for page_index in range(start_page, len(reader.pages)):
        text = reader.pages[page_index].extract_text()
        if not text:
            continue

        lines = text.splitlines()

        if page_index == start_page:  
            lines = lines[start_line:]   # Start from selected line

        page_text = " ".join(lines)
        sentences = page_text.replace("\n", " ").split(".")
        all_sentences.extend([s + "." for s in sentences if s.strip()])

    return all_sentences



def display(msg):
    print("======================================================================================================")
    logging.debug(f"CURRENT THREAD = {threading.current_thread().name} , [DEBUG] {msg}")
    print("-------------------                   -------------------------------")
    for t in threading.enumerate():
        logging.debug(f"active Thread: {t.name}, Alive={t.is_alive()}")
    print("======================================================================================================")


def list_all_models_lang_voices(start_directory):
    file_extension = ("json", "onnx")
    models_present = []
    for root, dirs, files in os.walk(start_directory):
        # print(f"Current directory: {root}")
        # print(f"Subdirectories: {dirs}")
        # print(f"Files: {files}")
        for file in files:
            full_file_path = os.path.join(root, file)
            #print(f"  File path: {full_file_path}")
            if file.endswith(file_extension):
                models_present.append(full_file_path) 
    voice_names, languages, quality_types, full_characteristics  = [],[],[],[]
    for model in models_present:
        voice_name = model.split("\\")[-3]
        lang = model.split("\\")[-4]
        quality = model.split("\\")[-2]
        voice_names.append(voice_name)
        languages.append(lang)
        quality_types.append(quality)
        full_characteristics.append((voice_name,lang,quality))
    full_characteristics =set(full_characteristics)
    #print(voice_names, "\n", languages, "\n", quality_types,"\n")
    print(*full_characteristics, sep = "\n")

    return models_present, set(voice_names), set(languages), set(quality_types), set(full_characteristics)




def choose_model(lang, voice_name, models_present):
    chosen_model = ""
    for model in models_present :
        if (("_"+str(lang)+"_") in model) and (("_"+str(voice_name)+"_") in model) and (".onnx" in model):
            chosen_model =model
            break
    print("MODEL FINALIZED =====  ",chosen_model)
    return chosen_model



class SimpleTTSStreamer:
    def __init__(self, voice_model_path):
        #self.voice = voice  # PiperVoice object
        self.voice = PiperVoice.load(voice_model_path)

        # Queues
        self.text_queue = queue.Queue()
        self.audio_queue = queue.Queue(maxsize=100)

        # Control flags
        self._paused = False
        self._stop = False

        self.END_LINE = object()
        self.STOP_SIGNAL = object()


        # Start consumer
        self.consumer_thread = threading.Thread(target=self._audio_consumer, daemon=False)
        self.consumer_thread.start()

        # Start producer
        self.producer_thread = threading.Thread(target=self._text_producer, daemon=False)
        self.producer_thread.start()

        # Audio output
        self.stream = None
        self.sample_rate = 24000

    # ------------------ PRODUCER ---------------------
    def _text_producer(self):
        frame_size = 2048

        while True:
            if self._stop:
                break
            logging.debug(f"Check time LINE 1 : -  { time.perf_counter()}")
            text = self.text_queue.get()   # waits for next line
            if text is self.STOP_SIGNAL:
                break
            
            if text is None:
                continue
            current_line_text = text  # <<< track current line

            #logging.info(f"LINE  :  {text}")
            logging.debug(f"Check time LINE 2 : -  { time.perf_counter()}")

            start_time_before_gen = time.perf_counter()
            gen = self.voice.synthesize(text)
            synth_time = time.perf_counter() - start_time_before_gen
            #logging.info(f"[PROFILE] Synthesis time: {synth_time:.3f}s for text='{text}'")

            logging.debug(f"Check time LINE 3 : -  { time.perf_counter()}")

            for chunk in gen:
                logging.debug(f"Check time LINE 4 : -  { time.perf_counter()}")
                if self._stop:
                    break

                internal_setence_float_conversion_start = time.perf_counter()
                audio_float = chunk.audio_float_array.astype(np.float32)

                logging.debug(f"Check time LINE 5 : -  { time.perf_counter()}")

                # Re-init stream if needed
                if self.stream is None or chunk.sample_rate != self.sample_rate:
                    self._init_stream(chunk.sample_rate)

                logging.debug(f"Check time LINE 6 : -  { time.perf_counter()}")

                # Chunk audio into frames
                internal_setence_put_time_start = time.perf_counter()
                internal_setence_float_conversion_time = internal_setence_put_time_start - internal_setence_float_conversion_start
                logging.debug(f"[PROFILE] internal_setence_float_conversion_time: {internal_setence_float_conversion_time:.3f}s ")


                for start in range(0, len(audio_float), frame_size):
                    logging.debug(f"  Frames Produced  =  {abs(len(audio_float)/frame_size)}")

                    logging.debug(f"Check time LINE 7 : -  { time.perf_counter()}")
                    if self._stop:
                        break

                    frame = audio_float[start:start+frame_size]

                    logging.debug(f"Check time LINE 8 : -  { time.perf_counter()}")

                    # respect pausing
                    while self._paused and not self._stop:
                        time.sleep(0.05)

                    logging.debug(f"Check time LINE 9 : -  { time.perf_counter()}")

                    # Put into queue (backpressure)
                    while True:
                        logging.debug(f"Check time LINE 10 : -  { time.perf_counter()}")
                        try:
                            self.audio_queue.put((frame,current_line_text), timeout=0.02)
                            break
                        except queue.Full:
                            if self._stop:
                                break
                            time.sleep(0.005)
                        logging.debug(f"Check time LINE 11 : -  { time.perf_counter()}")
                internal_setence_put_time_end = time.perf_counter()
                internal_setence_put_time = internal_setence_put_time_end - internal_setence_put_time_start
                logging.debug(f"[PROFILE] internal_setence_put_time: {internal_setence_put_time:.3f}s ")

            # Producer finished text, signal to consumer
            #self.audio_queue.put("END_LINE")
            self.audio_queue.put((self.END_LINE,current_line_text))

    # ------------------ CONSUMER ---------------------
    def _audio_consumer(self):
        logging.info("OUTSIE WHILE TRUE IN audio consumer")
        last_finish_stream = time.perf_counter()
        current_line = None
        while True:

            item  = self.audio_queue.get()
            frame, line_text = item 
            if self._stop:
                break
            if frame is self.STOP_SIGNAL:
                logging.info("Consumer thread exiting now.")
                break
            

            if frame is self.END_LINE:
                time.sleep(0.25)
                logging.info(f"NO FRAME --------")
                now = time.perf_counter()
                logging.info(f"[PROFILE] Line playback finished. Gap until next audio: {now - last_finish_stream:.3f}s")
                last_finish_stream = now
                current_line = None
                # line finished
                continue

            while self._paused and not self._stop:
                time.sleep(0.05)
            #logging.info(f"[PROFILE] Synthesis time: {synth_time:.3f}s for text='{text[:30]}...'")
            before_stream_time = time.perf_counter()


            # Print ONLY when the line changes
            if line_text != current_line:
                current_line = line_text
                logging.info(f"\n### NOW SPEAKING ###\n{current_line}\n")

            if self.stream:
                #logging.info(f"[PROFILE] Synthesis time: {synth_time:.3f}s for text='{text[:30]}...'")
                self.stream.write(frame)
                after_stream_time = time.perf_counter()
                stream_time  = after_stream_time - before_stream_time
                logging.debug(f"[PROFILE] Stream _true : {stream_time:.3f}s ")
            else:
                logging.info(f"Stream =  NOOOOOO STREAM")
        # <<< CLOSE STREAM HERE
        if self.stream:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception as e:
                logging.error(f"Error closing stream inside consumer: {e}")

        logging.info("Consumer exited")



    # ------------------ STREAM INIT ---------------------
    def _init_stream(self, sr):
        if self.stream:
            self.stream.stop()
            self.stream.close()

        self.stream = sd.OutputStream(
            samplerate=sr,
            channels=1,
            dtype='float32',
            blocksize=0
        )
        self.stream.start()
        self.sample_rate = sr

    # ------------------ PUBLIC API ---------------------
    def speak(self, text):
        self.text_queue.put(text)

    def pause(self):
        self._paused = True

    def resume(self):
        self._paused = False

    def stop(self):
        self._stop = True
        # if self.stream:
        #     # flush stream immediately
        #     self.stream.stop()
        #     self.stream.start()

        # No join here, thread will exit automatically on next chunk
        self._paused = False
        logging.info("Clearing queues")
        with self.text_queue.mutex:
            self.text_queue.queue.clear()
        with self.audio_queue.mutex:
            self.audio_queue.queue.clear()

        # close audio stream if exists
        if self.stream:
            try:
                self.stream.stop()
                self.stream.close()
                logging.info("Stream closed in stop")
            except Exception as e:
                logging.error(f"Error closing stream: {e}")
 
        # self.text_queue.put(self.STOP_SIGNAL)
        # self.audio_queue.put(self.STOP_SIGNAL)
        
        
                    # Wake producer so it can exit
        

        # Wake consumer so it can exit
        
        # clear queues
        # with self.text_queue.mutex:
        #     self.text_queue.queue.clear()
        # with self.audio_queue.mutex:
        #     self.audio_queue.queue.clear()

        #self._stop = False  # allow system to continue





if __name__ == "__main__":
    #pdf_path = r"C:\Users\Ishank\Documents\Python_projects\basic_tts\The Power of Positive Thinking - Norman Vincent Peale.pdf"
    global stop_flag, pause_flag

    pdf_path = r"C:\Users\Ishank\Documents\Python_projects\basic_tts\data\The Power of Positive Thinking - Norman Vincent Peale.pdf"
    reader = PdfReader(open(pdf_path, "rb"))
    total_pages = len(reader.pages)
    print(f"\nPDF has {total_pages} pages.\n")

    page = int(input("Enter start page (1-indexed): ")) - 1
    text = reader.pages[page].extract_text()
    lines = text.splitlines()

    print("\n--- Page Preview ---")
    for i, line in enumerate(lines[:25]):
        print(f"{i+1}: {line}")

    line_num = int(input("\nStart reading from line: ")) - 1

    print("\n📖 Extracting sentences...")
    sentences = extract_sentences_from_pdf(pdf_path, page, line_num)
    print(f"Total sentences to read: {len(sentences)}")



    base_model_dir = r"C:\Users\Ishank\Documents\Python_projects\basic_tts\voice_models"
    models_present, voice_names, languages, quality_types, full_characteristics  =list_all_models_lang_voices(base_model_dir)
    lang ='en'
    voice_name = 'ljspeech'
    chosen_model = choose_model(lang, voice_name, models_present)



    #r"C:\Users\Ishank\Documents\Python_projects\basic_tts\models\en_US-amy-low.onnx"
    # player = ProfessionalPiperPlayer(chosen_model)
    # display("player_called")
    # # Start reading in a background thread
    player = SimpleTTSStreamer(chosen_model)

    # feed all lines immediately
    for i, line in enumerate(sentences):
        #logging.info(f"Line called from main : \n {i} : {line} ")
        player.speak(line)

    # Control loop
    while True:
        #display("CMd input block ")
        cmd = input("\n[p]ause [r]esume [s]top [q]uit: ").strip().lower()

        if cmd == "p":
            player.pause()
            print("Paused.")

        elif cmd == "r":
            player.resume()
            print("Resumed.")

        elif cmd == "s":
            player.stop()
            print("Stopped.")

        elif cmd == "q":
            player.stop()
            #reader_thread.join(0.5)
            logging.info("Joining threads")
            player.producer_thread.join(0.1)
            player.consumer_thread.join(0.1)
            
            break

        else:
            print("Unknown command.")
        
    logging.info("PROGRAM CLOSED SUCCESSFULLY")
