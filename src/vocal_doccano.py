import os
import configparser
import json
import time
import logging
import sys

from vosk import Model, KaldiRecognizer
import pyaudio
from nltk.tokenize import RegexpTokenizer
from colorama import Fore, Style, init
from doccano_client import DoccanoClient
import simpleaudio as sa

"""Module for voice-controlled annotation of text documents in Doccano."""

# Initialize colorama for colored terminal output
init()

class Config:
    """
    Handles reading and accessing configuration settings from a .ini file.
    """

    def __init__(self, config_path):
        """
        Initializes the Config object by reading the configuration file.

        Args:
            config_path (str): The path to the configuration file.
        """
        self.config = configparser.ConfigParser()
        self.config.read(config_path)
    
    def get(self, section, key):
        """
        Retrieves a configuration value as a string.

        Args:
            section (str): The section in the configuration file.
            key (str): The key of the desired value.

        Returns:
            str: The configuration value.
        """
        return self.config.get(section, key)

    def getint(self, section, key, fallback=None):
        """
        Retrieves a configuration value as an integer.

        Args:
            section (str): The section in the configuration file.
            key (str): The key of the desired value.
            fallback (int, optional): A default value to return if the key is not found. Defaults to None.

        Returns:
            int: The configuration value as an integer.
        """
        return self.config.getint(section, key, fallback=fallback)

    def getint_optional(self, section, key, fallback=None):
        """
        Retrieves a configuration value as an integer, returning a fallback if the section or key doesn't exist.

        Args:
            section (str): The section in the configuration file.
            key (str): The key of the desired value.
            fallback (int, optional): A default value to return if the key or section is not found. Defaults to None.

        Returns:
            int: The configuration value as an integer or the fallback value.
        """
        try:
            return self.config.getint(section, key)
        except (configparser.NoSectionError, configparser.NoOptionError):
            return fallback

# --- Helper functions ---
def calculate_char_positions(text, tokens, start_token, end_token):
    """
    Calculates the start and end character positions of a selected token range
    directly in the original text.

    Args:
        text (str): The original text.
        tokens (list): A list of tokens.
        start_token (int): The index of the start token.
        end_token (int): The index of the end token.

    Returns:
        tuple: A tuple containing the start and end character positions.
    """
    if not tokens:
        return 0, 0

    start_char = -1
    end_char = -1

    # find the start position of the first token in the selected range
    try:
        start_token_text = tokens[start_token]
        start_index = 0
        for i in range(start_token):
            start_index = text.find(tokens[i], start_index) + len(tokens[i])
            if start_index == -1:
                logging.error(f"Start token '{tokens[i]}' not found in the text.")
                return -1, -1  # Fehlerbehandlung
        start_char = text.find(start_token_text, start_index)
        if start_char == -1:
            logging.error(f"Start token '{start_token_text}' not found in the text.")
            return -1, -1

        # find the start position of the first token in the selected range
        end_token_text = tokens[end_token]
        end_index_start = 0
        for i in range(end_token + 1):
            found_index = text.find(tokens[i], end_index_start)
            if found_index == -1:
                logging.error(f"End token '{tokens[i]}' not found in the text.")
                return -1, -1
            end_index_start = found_index + len(tokens[i])
        end_char = end_index_start

        logging.debug(f"  Calculating position for tokens {start_token} to {end_token}:")
        logging.debug(f"  Start-Token: '{tokens[start_token]}'")
        logging.debug(f"  End-Token: '{tokens[end_token]}'")
        logging.debug(f"  Start-Char: {start_char}")
        logging.debug(f"  End-Char: {end_char}")

        return start_char, end_char

    except IndexError:
        logging.error("Invalid token index.")
        return -1, -1
    except Exception as e:
        logging.error(f"Error calculating character positions: {e}")
        return -1, -1

def count_whitespace(text, first_token, second_token):
    """
    Counts the number of whitespace characters between the first occurrence
    of first_token and the subsequent first occurrence of second_token.
    """
    try:
        start = text.find(first_token)
        if start == -1:
            logging.warning(f"First token not found: '{first_token}'")
            return 0  # Or handle it in another appropriate way

        start_of_second = text.find(second_token, start + len(first_token))
        if start_of_second == -1:
            logging.warning(f"Second token not found after '{first_token}': '{second_token}'")
            return 0  # Or handle it in another appropriate way

        whitespace = text[start + len(first_token) : start_of_second]
        return len(whitespace)
    except Exception as e:
        logging.error(f"Error counting whitespace between '{first_token}' and '{second_token}': {e}")
        return 0  # Or handle it in another appropriate way

def play_sound(wave_object):
    """
    Plays a sound using the simpleaudio library.

    Args:
        wave_object (simpleaudio.WaveObject): The sound object to play.
    """
    if wave_object:
        play_it = wave_object.play()
        play_it.wait_done()

class VoiceAnnotationTool:
    """
    A tool for annotating text documents using voice commands.
    It integrates speech recognition (Vosk), document management (Doccano), and audio feedback.
    """

    def __init__(self):
        """
        Initializes the VoiceAnnotationTool, setting up configuration, Doccano client,
        speech recognition model, and audio output.
        """
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(
            level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s" # return to .INFO later, after successfull debugging
        )
        self.tokenizer = RegexpTokenizer(r'\w+|[^\w\s]+')
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, "..", "config", "config.ini")
        self.config = Config(config_path)

        # Settings (before Doccano and Vosk setup)
        self.batch_interval = self.config.getint(
            "BatchSettings", "batch_interval", fallback=30
        )
        self.batch_size = self.config.getint(
            "BatchSettings", "batch_size", fallback=10
        )
        self.max_retries = self.config.getint(
            "RetrySettings", "max_retries", fallback=3
        )
        self.retry_delay = self.config.getint(
            "RetrySettings", "retry_delay", fallback=5
        )
        self.tokens_to_display = self.config.getint(
            "DisplaySettings", "tokens_to_display", fallback=150
        )

        # Doccano Setup (labels are needed for command_map)
        self.doccano_url = self.config.get("Doccano", "url")
        self.username = self.config.get("Doccano", "username")
        self.password = self.config.get("Doccano", "password")
        self.project_id = self.config.getint("Doccano", "project_id")
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
            "back": "BACKWARDS",
        }
        self._setup_doccano()

        # Vosk Setup (after command_map is populated)
        model_path_relative = self.config.get("Vosk", "model_path")
        self.model_path = os.path.join(script_dir, "..", model_path_relative)
        self.model = None
        self.recognizer = None
        self._setup_vosk()

        # Audio Setup
        sound_path_relative = self.config.get("Audio", "sound_path")
        self.sound_path = os.path.join(script_dir, "..", sound_path_relative)
        self.wave_obj_tag = self._load_sound()

        # State variables
        self.document = None
        self.text = None
        self.cursor_position = 0
        self.selection_start = None
        self.selection_end = None
        self.annotation_batch = []
        self.existing_annotations = []
        self.last_batch_send_time = None
        self.last_tag_action = None

        # Automatically load a document from configuration if specified
        example_id_to_load = self.config.getint_optional("Startup", "document_id")
        if example_id_to_load is not None:
            self.load_document_by_id(example_id_to_load)

    def _setup_doccano(self):
        """
        Sets up the connection to the Doccano API and retrieves available labels.
        """
        try:
            self.client.login(username=self.username, password=self.password)
            self.logger.info("Successfully connected to Doccano!")
            label_types = self.client.list_label_types(
                project_id=self.project_id, type="span"
            )
            self.allowed_labels = [label.text for label in label_types] if label_types else []
            if self.allowed_labels:
                self.logger.info(
                    "Loaded available labels from Doccano project %s: %s",
                    self.project_id,
                    self.allowed_labels,
                )
                for label in self.allowed_labels:
                    self.command_map[label.lower()] = label.upper()
            else:
                self.logger.warning("No label types found in project %s!", self.project_id)
        except Exception as e:
            self.logger.error("Error connecting to Doccano or fetching labels: %s", e)
            sys.exit(1)

    def _setup_vosk(self):
        """
        Sets up the Vosk speech recognition model and configures the grammar.
        """
        if not os.path.exists(self.model_path):
            self.logger.error("The Vosk model was not found at '%s'.", self.model_path)
            sys.exit(1)
        self.model = Model(self.model_path)
        self.recognizer = KaldiRecognizer(self.model, 16000)
        grammar = json.dumps(list(self.command_map.keys()))
        self.logger.info("Vosk grammar: %s", grammar)
        self.recognizer.SetGrammar(grammar)

    def _load_sound(self):
        """
        Loads the sound file for audio feedback.
        """
        try:
            return sa.WaveObject.from_wave_file(self.sound_path)
        except FileNotFoundError:
            self.logger.error("Sound file '%s' not found.", self.sound_path)
            return None

    def load_document_by_id(self, document_id):
        """
        Loads a document from Doccano by its ID.

        Args:
            document_id (int): The ID of the document to load.

        Returns:
            bool: True if the document was loaded successfully, False otherwise.
        """
        try:
            document = self.client.find_example_by_id(
                project_id=self.project_id, example_id=document_id
            )
            if document:
                self.logger.info("Document with ID %s loaded successfully.", document_id)
                self.document = document
                self.text = self.tokenizer.tokenize(document.text)
                self.logger.debug(f"Tokenisierter Text: {self.text}")
                self.existing_annotations = self.load_existing_annotations(
                    document_id
                )  # Store existing annotations
                self.annotation_batch = []  # Clear the batch for new annotations
                self.last_batch_send_time = time.time()
                self.cursor_position = 0
                self.selection_start = None
                self.selection_end = None
                preview_length = min(self.tokens_to_display, len(self.text))
                preview_text = " ".join(self.text[:preview_length])
                self.logger.info("Initial text: ...%s...", preview_text)
                return True
            else:
                self.logger.warning("Document with ID %s not found.", document_id)
                self.document = None
                self.text = None
                return False
        except Exception as e:
            self.logger.error("Error loading document with ID %s: %s", document_id, e)
            self.document = None
            self.text = None
            return False

    def load_existing_annotations(self, document_id):
        """
        Loads existing annotations for a given document from Doccano.

        Args:
            document_id (int): The ID of the document.

        Returns:
            list: A list of existing annotations as dictionaries.
        """
        annotations = self.client.list_spans(
            project_id=self.project_id, example_id=document_id
        )
        return [
            {"start": a.start_offset, "end": a.end_offset, "label": a.label}
            for a in annotations
        ]

    def flush_annotation_batch(self):
        """
        Sends the current batch of annotations to Doccano.
        Implements retry logic in case of failure.
        """
        if self.annotation_batch:
            self.logger.info(
                "Attempting to send %d annotations to Doccano...",
                len(self.annotation_batch),
            )
            retries = 0
            while retries < self.max_retries:
                try:
                    for tag in self.annotation_batch:
                        self.client.create_span(
                            project_id=self.project_id,
                            example_id=self.document.id,
                            start_offset=tag["start"],
                            end_offset=tag["end"],
                            label=tag["label"],
                        )
                    self.annotation_batch = []
                    self.last_batch_send_time = time.time()
                    self.logger.info("Annotations sent successfully!")
                    return
                except Exception as e:
                    self.logger.error(
                        "Error sending annotations (attempt %d/%d): %s",
                        retries + 1,
                        self.max_retries,
                        e,
                    )
                    retries += 1
                    if retries < self.max_retries:
                        wait_time = self.retry_delay * (2**retries)
                        self.logger.info(
                            "Waiting %d seconds before retrying...", wait_time
                        )
                        time.sleep(wait_time)
            self.logger.error(
                "Failed to send annotations after %d attempts.", self.max_retries
            )
            self.annotation_batch = []

    def mark(self):
        """
        Starts the selection process by marking the token at the current cursor position.
        """
        self.selection_start = self.cursor_position
        self.selection_end = self.cursor_position
        marked_token = self.text[self.selection_start]
        colored_token = f"{Fore.BLUE}{marked_token}{Style.RESET_ALL}"
        self.logger.info("Marking started at token: %s", colored_token)

    def mark_next(self):
        """
        Extends the current selection to the next token.
        """
        if self.selection_start is not None:
            self.selection_end = min(self.selection_end + 1, len(self.text) - 1)
            marked_text = " ".join(
                self.text[self.selection_start : self.selection_end + 1]
            )
            colored_text = f"{Fore.BLUE}{marked_text}{Style.RESET_ALL}"
            self.logger.info("Marking extended to: %s", colored_text)

    def mark_back(self):
        """
        Reduces the current selection by one token from the end.
        """
        if self.selection_start is not None:
            self.selection_end = max(self.selection_end - 1, self.selection_start)
            marked_text = " ".join(
                self.text[self.selection_start : self.selection_end + 1]
            )
            colored_text = f"{Fore.BLUE}{marked_text}{Style.RESET_ALL}"
            self.logger.info("Marking reduced to: %s", colored_text)

    def show_next_tokens(self):
        """
        Displays the next few tokens from the current cursor position.
        The number of tokens displayed is configurable.
        """
        start_index = self.cursor_position
        end_index = min(len(self.text), start_index + self.tokens_to_display)
        next_tokens = " ".join(self.text[start_index:end_index])
        self.logger.info("Next tokens: ...%s...", next_tokens)

    def tag_selection(self, label):
        """
        Tags the currently selected text with the specified label and
        displays the current token after tagging.
        """
        if self.selection_start is not None and self.selection_end is not None:
            normalized_label = label.upper()
            if normalized_label in self.allowed_labels:
                start_char, end_char = calculate_char_positions(
                    self.document.text, self.text, self.selection_start, self.selection_end
                )
                tagged_text = " ".join(
                    self.text[self.selection_start : self.selection_end + 1]
                )
                colored_text = f"{Fore.GREEN}{tagged_text}{Style.RESET_ALL}"
                new_tag = {
                    "start": start_char,
                    "end": end_char,
                    "label": normalized_label,
                    "text": tagged_text,
                }
                self.annotation_batch.append(new_tag)
                self.last_tag_action = new_tag
                self.logger.info(
                    "Added tag '%s' as %s to batch.", colored_text, normalized_label
                )

                # Move the cursor to the next token
                self.cursor_position = self.selection_end + 1
                if self.cursor_position < len(self.text):
                    current_token = self.text[self.cursor_position]
                    self.logger.info("Current token: %s", current_token)
                elif self.text:
                    self.logger.info("End of document reached.")

                self.selection_start = None
                self.selection_end = None
                play_sound(self.wave_obj_tag)
            else:
                self.logger.warning(
                    "Invalid label: '%s'. Available labels: %s",
                    label,
                    self.allowed_labels,
                )
        else:
            self.logger.warning("No text selected for tagging.")

    def handle_command(self, command):
        """
        Handles recognized voice commands and performs the corresponding actions.

        Args:
            command (str): The recognized voice command.

        Returns:
            bool: True to continue processing commands, False to exit.
        """
        normalized_command = self.normalize_command(command)
        if normalized_command == "MARK":
            if self.text:
                self.logger.debug(f"Cursor position before mark: {self.cursor_position}")
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
                self.logger.info(
                    "Cursor moved back. Current token: %s",
                    self.text[self.cursor_position],
                )
            else:
                self.logger.warning("No document loaded. Cannot move backwards.")
        elif normalized_command == "FORWARD":
            if self.text:
                self.cursor_position = min(
                    self.cursor_position + 1, len(self.text) - 1
                )
                self.logger.info(
                    "Cursor moved forward. Current token: %s",
                    self.text[self.cursor_position],
                )
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
                colored_text = f"{Fore.GREEN}{self.last_tag_action['text']}{Style.RESET_ALL}"
                self.annotation_batch.remove(self.last_tag_action)
                self.logger.info("Removed last tag action: '%s' from batch.", colored_text)
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
        """
        Normalizes a command by looking it up in the command map.

        Args:
            command (str): The raw command string.

        Returns:
            str: The normalized command string or None if not found.
        """
        return self.command_map.get(command.lower())

    def run(self):
        """
        Starts the main loop of the voice annotation tool.
        This involves listening for voice commands and processing them.
        """
        audio = pyaudio.PyAudio()
        stream = audio.open(
            format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=8000
        )
        stream.start_stream()

        self.logger.info("Speak commands (e.g., 'mark', 'next')...")

        try:
            while True:
                data = stream.read(4000)
                if self.recognizer.AcceptWaveform(data):
                    result = json.loads(self.recognizer.Result())
                    raw_command = result.get("text", "").strip()
                    command = raw_command  # no normalize here

                    if command:
                        self.logger.info("Recognized command: %s", command)
                        if not self.handle_command(command):
                            break

                # Periodically send the annotation batch if the interval has passed
                if (
                    self.last_batch_send_time is not None
                    and time.time() - self.last_batch_send_time >= self.batch_interval
                    and self.annotation_batch
                ):
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