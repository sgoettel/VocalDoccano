import os
import configparser
import json
import time
from vosk import Model, KaldiRecognizer
import pyaudio
from nltk.tokenize import word_tokenize
from doccano_client import DoccanoClient
import simpleaudio as sa
import logging

# --- Konfiguration ---
class Config:
    def __init__(self, config_path):
        self.config = configparser.ConfigParser()
        self.config.read(config_path)

    def get(self, section, key):
        return self.config.get(section, key)

    def getint(self, section, key, fallback=None):
        return self.config.getint(section, key, fallback=fallback)

    def getint_optional(self, section, key, fallback=None):
        try:
            return self.config.getint(section, key)
        except (configparser.NoSectionError, configparser.NoOptionError):
            return fallback

# --- Hilfsfunktionen ---
def calculate_char_positions(tokens, start_token, end_token):
    start = sum(len(token) + 1 for token in tokens[:start_token])
    end = start + sum(len(token) + 1 for token in tokens[start_token:end_token + 1]) - 1
    return start, end

def play_sound(wave_object):
    if wave_object:
        play_it = wave_object.play()
        play_it.wait_done()

class VoiceAnnotationTool:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, '..', 'config', 'config.ini')
        self.config = Config(config_path)

        # Einstellungen (vor Doccano und Vosk)
        self.batch_interval = self.config.getint('BatchSettings', 'batch_interval', fallback=30)
        self.batch_size = self.config.getint('BatchSettings', 'batch_size', fallback=10)
        self.max_retries = self.config.getint('RetrySettings', 'max_retries', fallback=3)
        self.retry_delay = self.config.getint('RetrySettings', 'retry_delay', fallback=5)
        self.tokens_to_display = self.config.getint('DisplaySettings', 'tokens_to_display', fallback=150)

        # Doccano Setup (Labels werden für command_map benötigt)
        self.doccano_url = self.config.get('Doccano', 'url')
        self.username = self.config.get('Doccano', 'username')
        self.password = self.config.get('Doccano', 'password')
        self.project_id = self.config.getint('Doccano', 'project_id')
        self.client = DoccanoClient(self.doccano_url)
        self.allowed_labels = []
        self.command_map = {
            "mark": "MARK",
            "mark next": "MARK NEXT",
            "mark back": "MARK BACK",
            "next": "FORWARD",
            "exit": "EXIT",
            "undo": "UNDO",
            "show next": "SHOW NEXT TOKENS",
            "back": "BACKWARDS"
        }
        self._setup_doccano()

        # Vosk Setup (nach command_map)
        self.logger.info(f"Command map vor Vosk Setup: {self.command_map}") # Zur Überprüfung
        model_path_relative = self.config.get('Vosk', 'model_path')
        self.model_path = os.path.join(script_dir, '..', model_path_relative)
        self.model = None
        self.recognizer = None
        self._setup_vosk()

        # Audio Setup
        sound_path_relative = self.config.get('Audio', 'sound_path')
        self.sound_path = os.path.join(script_dir, '..', sound_path_relative)
        self.wave_obj_tag = self._load_sound()

        # Zustandsvariablen
        self.document = None
        self.text = None
        self.cursor_position = 0
        self.selection_start = None
        self.selection_end = None
        self.annotation_batch = []
        self.existing_annotations = []
        self.last_batch_send_time = None
        self.last_tag_action = None

        # Automatisches Laden des Dokuments aus der Konfiguration**
        example_id_to_load = self.config.getint_optional('Startup', 'document_id')
        if example_id_to_load is not None:
            if self.load_document_by_id(example_id_to_load): # Korrekter Aufruf mit example_id**
                self.logger.info(f"Successfully loaded document with ID {example_id_to_load} from config.")
            else:
                self.logger.warning(f"Could not load document with ID {example_id_to_load} from config.")
        else:
            self.logger.info("No document ID specified in config.ini, not loading any document automatically.")

    def _setup_doccano(self):
        try:
            self.client.login(username=self.username, password=self.password)
            self.logger.info("Successfully connected to Doccano!")
            label_types = self.client.list_label_types(project_id=self.project_id, type='span')
            self.allowed_labels = [label.text for label in label_types] if label_types else []
            if self.allowed_labels:
                self.logger.info(f"Loaded available labels from Doccano project {self.project_id}: {self.allowed_labels}")
                for label in self.allowed_labels:
                    self.command_map[label.lower()] = label.upper()
                self.logger.info(f"Updated command map: {self.command_map}")
            else:
                self.logger.warning(f"No label types found in project {self.project_id}!")
        except Exception as e:
            self.logger.error(f"Error connecting to Doccano or fetching labels: {e}")
            exit(1)

    def _setup_vosk(self):
        if not os.path.exists(self.model_path):
            self.logger.error(f"The Vosk model was not found at '{self.model_path}'.")
            exit(1)
        self.model = Model(self.model_path)
        self.recognizer = KaldiRecognizer(self.model, 16000)
        grammar = json.dumps(list(self.command_map.keys()))
        self.logger.info(f"Vosk grammar: {grammar}")
        self.recognizer.SetGrammar(grammar)

    def _load_sound(self):
        try:
            return sa.WaveObject.from_wave_file(self.sound_path)
        except FileNotFoundError:
            self.logger.error(f"Sound file '{self.sound_path}' not found.")
            return None

    def load_document_by_id(self, document_id):
        try:
            document = self.client.find_example_by_id(project_id=self.project_id, example_id=document_id)
            if document:
                self.logger.info(f"Document with ID {document_id} loaded successfully.")
                self.document = document
                self.text = word_tokenize(document.text)
                self.existing_annotations = self.load_existing_annotations(document_id) # Speichern der vorhandenen Annotationen
                self.annotation_batch = [] # Leeren des Batches für neue Annotationen
                self.last_batch_send_time = time.time()
                self.cursor_position = 0
                self.selection_start = None
                self.selection_end = None
                preview_length = min(self.tokens_to_display, len(self.text))
                preview_text = " ".join(self.text[:preview_length])
                self.logger.info(f"Initial text: ...{preview_text}...")
                return True
            else:
                self.logger.warning(f"Document with ID {document_id} not found.")
                self.document = None
                self.text = None
                return False
        except Exception as e:
            self.logger.error(f"Error loading document with ID {document_id}: {e}")
            self.document = None
            self.text = None
            return False

    def get_text_for_annotation(self):
        try:
            project_examples = self.client.list_examples(project_id=self.project_id)
            if not project_examples:
                self.logger.info(f"No documents found in project with ID {self.project_id}.")
                return None

            for example in project_examples:
                annotations = self.client.list_spans(project_id=self.project_id, example_id=example.id)
                if not any(annotations) or sum(1 for _ in annotations) < len(word_tokenize(example.text)):
                    self.logger.info(f"Retrieved document ID: {example.id}")
                    preview_length = min(500, len(example.text))
                    self.logger.info(f"Retrieved text (excerpt): {example.text[:preview_length]}...")
                    return example
            return None
        except Exception as e:
            self.logger.error(f"Error retrieving documents: {e}")
            return None

    def load_existing_annotations(self, document_id):
        annotations = self.client.list_spans(project_id=self.project_id, example_id=document_id)
        return [{"start": a.start_offset, "end": a.end_offset, "label": a.label} for a in annotations]

    def flush_annotation_batch(self):
        if self.annotation_batch:
            self.logger.info(f"Attempting to send {len(self.annotation_batch)} annotations to Doccano...")
            retries = 0
            while retries < self.max_retries:
                try:
                    for tag in self.annotation_batch:
                        self.client.create_span(
                            project_id=self.project_id,
                            example_id=self.document.id,
                            start_offset=tag["start"],
                            end_offset=tag["end"],
                            label=tag["label"]
                        )
                    self.annotation_batch = []
                    self.last_batch_send_time = time.time()
                    self.logger.info("Annotations sent successfully!")
                    return
                except Exception as e:
                    self.logger.error(f"Error sending annotations (attempt {retries + 1}/{self.max_retries}): {e}")
                    retries += 1
                    if retries < self.max_retries:
                        wait_time = self.retry_delay * (2 ** retries)
                        self.logger.info(f"Waiting {wait_time} seconds before retrying...")
                        time.sleep(wait_time)
            self.logger.error(f"Failed to send annotations after {self.max_retries} attempts.")
            self.annotation_batch = []

    def mark(self):
        self.selection_start = self.cursor_position
        self.selection_end = self.cursor_position
        self.logger.info(f"Marking started at token: {self.text[self.selection_start]}")

    def mark_next(self):
        if self.selection_start is not None:
            self.selection_end = min(self.selection_end + 1, len(self.text) - 1)
            self.logger.info(f"Marking extended to: {' '.join(self.text[self.selection_start:self.selection_end + 1])}")

    def mark_back(self):
        if self.selection_start is not None:
            self.selection_end = max(self.selection_end - 1, self.selection_start)
            self.logger.info(f"Marking reduced to: {' '.join(self.text[self.selection_start:self.selection_end + 1])}")

    def show_next_tokens(self):
        start_index = self.cursor_position
        end_index = min(len(self.text), start_index + self.tokens_to_display)
        next_tokens = " ".join(self.text[start_index:end_index])
        self.logger.info(f"Next tokens: ...{next_tokens}...")

    def tag_selection(self, label):
        if self.selection_start is not None and self.selection_end is not None:
            normalized_label = label.upper()
            if normalized_label in self.allowed_labels:
                start_char, end_char = calculate_char_positions(self.text, self.selection_start, self.selection_end)
                tagged_text = " ".join(self.text[self.selection_start:self.selection_end + 1])
                new_tag = {"start": start_char, "end": end_char, "label": normalized_label, "text": tagged_text}
                self.annotation_batch.append(new_tag)
                self.last_tag_action = new_tag
                self.logger.info(f"Added tag '{tagged_text}' as {normalized_label} to batch.")
                self.cursor_position = self.selection_end + 1
                self.selection_start = None
                self.selection_end = None
                play_sound(self.wave_obj_tag)
            else:
                self.logger.warning(f"Invalid label: '{label}'. Available labels: {self.allowed_labels}")
        else:
            self.logger.warning("No text selected for tagging.")

    def handle_command(self, command):
        normalized_command = self.normalize_command(command)
        if normalized_command == "MARK":
            if self.text:
                self.mark()
            else:
                self.logger.warning("No document loaded. Cannot mark.")
        elif normalized_command == "MARK NEXT":
            if self.text:
                self.mark_next()
            else:
                self.logger.warning("No document loaded. Cannot mark next.")
        elif normalized_command == "MARK BACK":
            if self.text:
                self.mark_back()
            else:
                self.logger.warning("No document loaded. Cannot adjust mark.")
        elif normalized_command == "BACKWARDS":
            if self.text:
                self.cursor_position = max(self.cursor_position - 1, 0)
                self.logger.info(f"Cursor moved back. Current token: {self.text[self.cursor_position]}")
            else:
                self.logger.warning("No document loaded. Cannot move backwards.")
        elif normalized_command == "FORWARD":
            if self.text:
                self.cursor_position = min(self.cursor_position + 1, len(self.text) - 1)
                self.logger.info(f"Cursor moved forward. Current token: {self.text[self.cursor_position]}")
            else:
                self.logger.warning("No document loaded. Cannot move forward.")
        elif normalized_command == "SHOW NEXT TOKENS":
            if self.text:
                self.show_next_tokens()
            else:
                self.logger.warning("No document loaded. Cannot show next tokens.")
        elif normalized_command == "EXIT":
            self.logger.info("Exiting the script...")
            self.flush_annotation_batch()
            return False
        elif normalized_command == "UNDO":
            if self.last_tag_action in self.annotation_batch:
                self.annotation_batch.remove(self.last_tag_action)
                self.logger.info(f"Removed last tag action: '{self.last_tag_action['text']}' from batch.")
                self.last_tag_action = None
            else:
                self.logger.warning("No tag action available to undo.")
        elif normalized_command in self.allowed_labels:
            if self.text:
                self.tag_selection(command)
            else:
                self.logger.warning("No document loaded. Cannot tag.")
        else:
            self.logger.info("No command recognized.")
        return True

    def normalize_command(self, command):
        return self.command_map.get(command.lower())

    def run(self):
        audio = pyaudio.PyAudio()
        stream = audio.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=8000)
        stream.start_stream()

        self.logger.info("Speak commands (e.g., 'mark', 'next', 'person', 'location', 'load document <ID>')...")

        try:
            while True:
                data = stream.read(4000)
                if self.recognizer.AcceptWaveform(data):
                    result = json.loads(self.recognizer.Result())
                    raw_command = result.get("text", "").strip()
                    command = raw_command # Don't normalize here, do it in handle_command

                    if command:
                        self.logger.info(f"Recognized command: {command}")
                        if not self.handle_command(command):
                            break

                if self.last_batch_send_time is not None and time.time() - self.last_batch_send_time >= self.batch_interval and self.annotation_batch:
                    self.flush_annotation_batch()

        except KeyboardInterrupt:
            self.logger.info("\nAnnotation process ended by user.")
            self.flush_annotation_batch()
        finally:
            stream.stop_stream()
            stream.close()
            audio.terminate()

if __name__ == "__main__":
    tool = VoiceAnnotationTool()
    tool.run()