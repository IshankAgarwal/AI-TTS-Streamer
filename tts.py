import pandas as pd

print("pd version", pd.__version__)

### IMPORTANT - using pyttsx3 = 2.91 , 2.99(has issues)
import pyttsx3
import PyPDF2
import threading
import time

engine = pyttsx3.init()
engine.setProperty('rate', 170)

stop_flag = False
pause_flag = False

# def speak_sentences(sentences):
#     global stop_flag, pause_flag
#     for sentence in sentences:
#         if stop_flag:
#             break
#         while pause_flag:
#             time.sleep(0.1)

#         cleaned = sentence.strip()
#         if cleaned:
#             engine.say(cleaned)
#             engine.iterate()  # non-blocking

#     engine.runAndWait()  # process all queued audio
def speak_sentences(sentences):
    global stop_flag, pause_flag

    for sentence in sentences:
        #print("for looop 1  = ", sentence.strip())
        if stop_flag:
            break
        while pause_flag:
            time.sleep(0.1)
        if sentence.strip():
            #print(sentence.strip())
            engine.say(sentence.strip())
            #print("saying")
            engine.runAndWait()

def user_control():
    global stop_flag, pause_flag
    while True:
        cmd = input("\n[pause/resume/stop/quit] > ").strip().lower()
        if cmd == "pause":
            pause_flag = True
            print("⏸ Paused.")
        elif cmd == "resume":
            pause_flag = False
            print("▶ Resumed.")
        elif cmd == "stop":
            stop_flag = True
            print("⛔ Stopped reading.")
        elif cmd == "quit":
            stop_flag = True
            print("👋 Exiting program...")
            exit()

def extract_sentences_from_pdf(pdf_path, start_page, start_line):
    reader = PyPDF2.PdfReader(open(pdf_path, "rb"))
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


def main():
    global stop_flag, pause_flag

    pdf_path = input("Enter PDF path: ")
    pdf_path = r"C:\Users\Ishank\Documents\Python_projects\basic_tts\data\The Power of Positive Thinking - Norman Vincent Peale.pdf"
    reader = PyPDF2.PdfReader(open(pdf_path, "rb"))
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

    # Start control thread
    threading.Thread(target=user_control, daemon=True).start()

    print("\n▶ Starting reading...\n")
    speak_sentences(sentences)

    print("✔ Finished (or stopped).")


if __name__ == "__main__":
    main()
    