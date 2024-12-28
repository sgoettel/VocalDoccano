# VocalDoccano

VocalDoccano is a Python script for voice-controlled annotation in [Doccano](https://github.com/doccano/doccano). It enables users to perform sequence labeling tasks using voice commands, offering an alternative to manual annotation via mouse and keyboard.

By using voice commands, users can potentially annotate text with less physical effort. This script is a little experiment to see if we can make the process a bit more engaging and, dare I say, fun, by using speech recognition...anyway..

## Requirements

* **Python 3.7+**
* **pip (you'll need this to install the other stuff)**
* **Python Dependencies:**  Make sure you've got these installed. The easiest way is to run:
   ```bash
   pip install -r requirements.txt
   ```

* **Running Doccano Instance:** You need a Doccano instance up and running and accessible via its API.  It doesn't matter how you installed Doccano (Docker, pip), as long as VocalDoccano can connect. I used [Docker Compose installation](https://doccano.github.io/doccano/install_and_upgrade_doccano/#install-with-docker-compose). You'll need your Doccano URL, username, and password.
* **Vosk Speech Recognition Model:**  You'll need a speech recognition model from Vosk. Small models are usually fine and you can grab them from [https://alphacephei.com/vosk/models](https://alphacephei.com/vosk/models). The `vosk-model-small-en-us-0.15` model (which is mentioned in the config file) is a good starting point for English.

## Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/sgoettel/VocalDoccano/
   cd VocalDoccano
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

You'll find all the settings in the `config/config.ini` file.

* **`[Doccano]`:** 
    * `url`:  The web address of your Doccano (like `http://localhost/`).
    * `username`: Your Doccano login name.
    * `password`: Your Doccano login password.
    * `project_id`: The ID of the specific project in Doccano you want to work on.

* **`[Startup]`:**
    * `document_id`: The ID of the specific document to load automatically on startup.

* **`[Vosk]`:**  Settings for the speech recognition.
    * `model_path`:  Tell VocalDoccano where your downloaded Vosk model is. The path should be relative to the main folder of the VocalDoccano repository. For example: `models/vosk-model-small-en-us-0.15`.

* **`[Audio]`:** Settings for the sound you hear when you tag something.
    * `sound_path`:  The location of the sound file. This is also relative to the main repository folder (e.g., `assets/tag_notification.wav`).

* **`[BatchSettings]`:** These settings control how often your annotations are sent to Doccano.

* **`[RetrySettings]`:** If something goes wrong when sending annotations to Doccano, these settings tell the script how many times to try again and how long to wait.

* **`[DisplaySettings]`:** Controls how many words you see when you use the `show next` command, and also how much text is shown initially when you load a document.

## Usage

1. **Go to the `src` directory:**
   ```bash
   cd src
   ```

2. **Run the script:**
   ```bash
   python vocal_doccano.py
   ```
   (Running it with `python src/vocal_doccano.py` from the main directory also works)

   The script will try to connect to Doccano and load your document. Keep an eye on the terminal for messages.

## Voice Commands

* `mark`: Starts selecting text at the current word.
* `mark next`:  Expands the current selection to include the next word.
* `mark back`:  Shrinks the current selection by one word.
* `<label>`:  Say the name of a label (like "Person", "Location", etc.) to tag the currently selected text.
* `next`: Moves the cursor forward.
* `back`: Moves the cursor backward.
* `show next`: Shows you the next bit of text in the console.
* `undo`:  Removes the last tag you created in the current batch (before sending to Doccano).
* `exit`: Saves any annotations you've made but haven't sent yet and then quits the script.

**(Want to change the commands? You'll find them in the `command_map` within the `VoiceAnnotationTool` class in `src/vocal_doccano.py`.)**

**Important:** VocalDoccano uses the exact labels you've set up in your Doccano project as the voice commands for tagging. So, if you have a label called "Organization", you'd say "Organization".