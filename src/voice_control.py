from vosk import Model, KaldiRecognizer
from nltk.tokenize import word_tokenize, sent_tokenize
import pyaudio
import json
import time
import os
import configparser
from doccano_client import DoccanoClient
import simpleaudio as sa

# proejct directory (for resolving relative paths)
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_FILE = os.path.join(PROJECT_DIR, 'config.ini')

config = configparser.ConfigParser()
config.read(CONFIG_FILE) # you can adjust all the settings in the 'config.ini' file

# Doccano settings
DOCCANO_URL = config.get('Doccano', 'url')
USERNAME = config.get('Doccano', 'username')
PASSWORD = config.get('Doccano', 'password')
PROJECT_ID = config.getint('Doccano', 'project_id')

# Vosk model path
MODEL_PATH = os.path.join(PROJECT_DIR, config.get('Vosk', 'model_path'))

# Audio sound path
SOUND_PATH = os.path.join(PROJECT_DIR, config.get('Audio', 'sound_path'))


# Batch settings
BATCH_INTERVAL = config.getint('BatchSettings', 'batch_interval', fallback=30)
BATCH_SIZE = config.getint('BatchSettings', 'batch_size', fallback=10)

# Retry settings
MAX_RETRIES = config.getint('RetrySettings', 'max_retries', fallback=3)
RETRY_DELAY = config.getint('RetrySettings', 'retry_delay', fallback=5)

# Display settings
TOKENS_TO_DISPLAY = config.getint('DisplaySettings', 'tokens_to_display', fallback=150)


# Speech commands mapping
COMMAND_MAP = {
    "mark": "mark",
    "mark next": "mark next",
    "mark back": "mark back",
    "next": "forward",
    "exit": "exit",
    "undo": "undo",
    "person": "PERSON",
    "occupation": "OCCUPATION",
    "political role": "POLITICAL_ROLE",
    "location": "LOCATION",
    "ideology": "IDEOLOGY",
    "notes": "NOTES",
    "institution": "INSTITUTION",
    "date": "DATE",
    "accusation": "ACCUSATION",
    "status": "STATUS",
    "show next": "show next tokens",
    "back": "backwards"
}

# --- Doccano Setup ---
if not os.path.exists(MODEL_PATH):
    print(f"Error: The Vosk model was not found at '{MODEL_PATH}'. Please download it and adjust the path.")
    exit(1)

client = DoccanoClient(DOCCANO_URL)
try:
    client.login(username=USERNAME, password=PASSWORD)
    print("Successfully connected to Doccano!")

    # Fetch available label types
    label_types = client.list_label_types(project_id=PROJECT_ID, type='span')
    if label_types:
        ALLOWED_LABELS = [label.text for label in label_types] # Loads labels from Doccano
        print(f"Loaded available labels from Doccano: {ALLOWED_LABELS}")
    else:
        print("Warning: No label types found in the project!")

    # Update COMMAND_MAP based on the loaded labels
    for label in ALLOWED_LABELS:
        COMMAND_MAP[label.lower()] = label.upper()

except Exception as e:
    print(f"Error connecting to Doccano or fetching labels: {e}")
    exit(1)

# --- Sound Setup ---
try:
    wave_obj_tag = sa.WaveObject.from_wave_file(SOUND_PATH)
except FileNotFoundError:
    print("Error: Sound files 'tag_notification.wav' not found. Sound notification will be disabled.")
    wave_obj_tag = None


# --- Speech Recognition Setup ---
model = Model(MODEL_PATH)
recognizer = KaldiRecognizer(model, 16000)
recognizer.SetGrammar(json.dumps(list(COMMAND_MAP.keys())))
audio = pyaudio.PyAudio()
stream = audio.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=8000)
stream.start_stream()

print("Speak commands (e.g., 'mark', 'next', 'person', 'location')...")

# --- Helper Functions ---
def calculate_char_positions(tokens, start_token, end_token):
    """Calculates the character start and end positions based on token indices."""
    start = 0
    for i in range(start_token):
        start += len(tokens[i]) + 1
    end = start
    for i in range(start_token, end_token + 1):
        end += len(tokens[i]) + 1
    return start, end - 1

def normalize_command(command):
    """Normalizes the recognized command by mapping it to the COMMAND_MAP."""
    return COMMAND_MAP.get(command.lower())

def play_sound(wave_object):
    """Plays a given sound object if it's not None."""
    if wave_object:
        play_it = wave_object.play()
        play_it.wait_done()


# --- Core Functions ---
def get_text_for_annotation():
    """Fetches the next unannotated or partially annotated text from Doccano."""
    try:
        project_examples = client.list_examples(project_id=PROJECT_ID)
        if not project_examples:
            print(f"No documents found in project with ID {PROJECT_ID}.")
            return None

        for example in project_examples:
            annotations = client.list_spans(project_id=PROJECT_ID, example_id=example.id)
            if not any(annotations):
                print(f"Retrieved document ID: {example.id}")
                preview_length = min(500, len(example.text))
                print(f"Retrieved text (excerpt): {example.text[:preview_length]}...")
                return example
            else:
                word_count = len(word_tokenize(example.text))
                annotation_count = sum(1 for _ in annotations)
                if annotation_count < word_count:
                    print(f"Retrieved document ID: {example.id}")
                    preview_length = min(500, len(example.text))
                    print(f"Retrieved text (excerpt): {example.text[:preview_length]}...")
                    return example
        return None
    except Exception as e:
        print(f"Error retrieving documents: {e}")
        return None

def load_existing_annotations(document_id):
    """Loads existing annotations for a given document from Doccano."""
    annotations = client.list_spans(project_id=PROJECT_ID, example_id=document_id)
    return [{"start": a.start_offset, "end": a.end_offset, "label": a.label} for a in annotations]

annotation_batch = []
last_batch_send_time = time.time()

def flush_annotation_batch():
    """Sends the current batch of annotations to Doccano with retries."""
    global annotation_batch, last_batch_send_time
    if annotation_batch:
        print(f"Attempting to send {len(annotation_batch)} annotations to Doccano...")
        retries = 0
        while retries < MAX_RETRIES:
            try:
                for tag in annotation_batch:
                    client.create_span(
                        project_id=PROJECT_ID,
                        example_id=document.id,
                        start_offset=tag["start"],
                        end_offset=tag["end"],
                        label=tag["label"]
                    )
                annotation_batch = []
                last_batch_send_time = time.time()
                print("Annotations sent successfully!")
                return  # Successfully sent, exit function
            except Exception as e:
                print(f"Error sending annotations (attempt {retries + 1}/{MAX_RETRIES}): {e}")
                retries += 1
                if retries < MAX_RETRIES:
                    wait_time = RETRY_DELAY * (2 ** retries)  # Exponential backoff
                    print(f"Waiting {wait_time} seconds before retrying...")
                    time.sleep(wait_time)
        print(f"Failed to send annotations after {MAX_RETRIES} attempts.")
        annotation_batch = [] # Discard the batch after multiple failed attempts (optional)

def mark(cursor_position):
    """Starts the marking selection at the current cursor position."""
    global selection_start, selection_end
    selection_start = cursor_position
    selection_end = cursor_position
    print(f"Marking started at token: {text[selection_start]}")

def mark_next():
    """Extends the marking selection to the next token."""
    global selection_end
    if selection_start is not None:
        selection_end = min(selection_end + 1, len(text) - 1)
        print(f"Marking extended to: {' '.join(text[selection_start:selection_end + 1])}")

def mark_back():
    """Reduces the marking selection to the previous token."""
    global selection_end
    if selection_start is not None:
        selection_end = max(selection_end - 1, selection_start)
        print(f"Marking reduced to: {' '.join(text[selection_start:selection_end + 1])}")

def show_next_tokens():
    """Shows the next tokens from the current cursor position."""
    start_index = cursor_position
    end_index = min(len(text), start_index + TOKENS_TO_DISPLAY)
    next_tokens = " ".join(text[start_index:end_index])
    print(f"\nNext tokens: ...{next_tokens}...")

def tag_selection(label):
    """Tags the currently selected text with the given label and adds it to the batch."""
    global selection_start, selection_end, tags, ALLOWED_LABELS, cursor_position, last_tag_action, annotation_batch
    if selection_start is not None and selection_end is not None:
        normalized_label = label.upper()
        if normalized_label in ALLOWED_LABELS:
            start_char, end_char = calculate_char_positions(text, selection_start, selection_end)
            tagged_text = " ".join(text[selection_start:selection_end + 1])
            new_tag = {"start": start_char, "end": end_char, "label": normalized_label, "text": tagged_text}
            annotation_batch.append(new_tag)
            last_tag_action = new_tag
            print(f"Added tag '{tagged_text}' as {normalized_label} to batch.")
            cursor_position = selection_end + 1
            selection_start = None
            selection_end = None
            play_sound(wave_obj_tag)
        else:
            print(f"Invalid label: '{label}'. Available labels: {ALLOWED_LABELS}")
    else:
        print("No text selected for tagging.")

# --- Main Application Loop ---
if __name__ == "__main__":
    document = get_text_for_annotation()
    if document:
        text = word_tokenize(document.text)
        existing_annotations = load_existing_annotations(document.id)
        tags = existing_annotations.copy()
        last_tag_action = None
    else:
        print("No unannotated documents available.")
        exit(1)

    cursor_position = 0
    selection_start = None
    selection_end = None
    last_batch_send_time = time.time()

    try:
        while True:
            data = stream.read(4000)
            if recognizer.AcceptWaveform(data):
                result = json.loads(recognizer.Result())
                raw_command = result.get("text", "").strip()
                command = normalize_command(raw_command)

                if command:
                    print(f"Executing command: {command}")
                    if command == "mark":
                        mark(cursor_position)
                    elif command == "mark next":
                        mark_next()
                    elif command == "mark back":
                        mark_back()
                    elif command == "backwards":
                        cursor_position = max(cursor_position - 1, 0)
                        print(f"Cursor moved back. Current token: {text[cursor_position]}")
                    elif command == "forward":
                        cursor_position = min(cursor_position + 1, len(text) - 1)
                        print(f"Cursor moved forward. Current token: {text[cursor_position]}")
                    elif command == "show next tokens":
                        show_next_tokens()
                    elif command == "exit":
                        print("Exiting the script...")
                        flush_annotation_batch()  # Send all remaining annotations
                        break
                    elif command == "undo":
                        if last_tag_action:
                            if last_tag_action in annotation_batch:
                                annotation_batch.remove(last_tag_action)
                                print(f"Removed last tag action: '{last_tag_action['text']}' from batch.")
                                last_tag_action = None
                            else:
                                print("Error: Could not find the last tag action in batch.")
                        else:
                            print("No tag action available to undo.")
                    elif command.upper() in ALLOWED_LABELS: # Direct label recognition
                        tag_selection(command.upper())
                else:
                    print("No command recognized.")

            # Regularly send annotations in the batch
            if time.time() - last_batch_send_time >= BATCH_INTERVAL and annotation_batch:
                flush_annotation_batch()

    except KeyboardInterrupt:
        print("\nAnnotation process ended by user.")
        flush_annotation_batch() # Ensure sending even on interruption
    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()