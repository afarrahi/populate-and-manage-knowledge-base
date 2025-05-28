import os
import requests
import json
import logging
import re
import time
import argparse
from datetime import datetime

# --- Configuration Constants ---
# Determine the directory where the script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Define the path to the global configuration file relative to the script
GLOBAL_CONFIG_PATH = os.path.join(SCRIPT_DIR, 'config.json')

# Define the root directory for your document collection.
# By default, it's a directory named 'my_books' located next to the script.
MAIN_COLLECTION_ROOT = os.path.join(SCRIPT_DIR, 'my_books')

# Define the directory for local tracking files, relative to the script.
# This keeps tracker files separate from subject directories themselves.
TRACKER_DIR = os.path.join(SCRIPT_DIR, 'trackers')

# --- Logging Configuration ---
# Configure logging to output to console and a file
logging.basicConfig(level=logging.INFO, # Default logging level
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler(os.path.join(SCRIPT_DIR, 'openwebui_kb_manager.log')),
                        logging.StreamHandler()
                    ])
logger = logging.getLogger(__name__)

# --- Helper Functions for API Interactions ---

def load_global_config():
    """Loads the global configuration from config.json."""
    if not os.path.exists(GLOBAL_CONFIG_PATH):
        logger.error(f"Configuration file not found at: {GLOBAL_CONFIG_PATH}")
        return None
    try:
        with open(GLOBAL_CONFIG_PATH, 'r') as f:
            config = json.load(f)
        # Ensure MAIN_COLLECTION_ROOT is absolute and canonical
        if 'main_collection_root' in config:
            # If path in config.json is relative, make it absolute relative to script_dir
            if not os.path.isabs(config['main_collection_root']):
                config['main_collection_root'] = os.path.abspath(
                    os.path.join(SCRIPT_DIR, config['main_collection_root']))
            else:
                config['main_collection_root'] = os.path.abspath(
                    config['main_collection_root'])
        else:
            logger.warning(f"'main_collection_root' not found in {GLOBAL_CONFIG_PATH}. "
                           f"Using default: {MAIN_COLLECTION_ROOT}")
            config['main_collection_root'] = MAIN_COLLECTION_ROOT

        # Ensure TRACKER_DIR is absolute and canonical
        if 'tracker_directory' in config:
            if not os.path.isabs(config['tracker_directory']):
                config['tracker_directory'] = os.path.abspath(
                    os.path.join(SCRIPT_DIR, config['tracker_directory']))
            else:
                config['tracker_directory'] = os.path.abspath(
                    config['tracker_directory'])
        else:
            logger.warning(f"'tracker_directory' not found in {GLOBAL_CONFIG_PATH}. "
                           f"Using default: {TRACKER_DIR}")
            config['tracker_directory'] = TRACKER_DIR

        # Ensure the tracker directory exists
        os.makedirs(config['tracker_directory'], exist_ok=True)
        logger.debug(f"Tracker directory ensured: {config['tracker_directory']}")


        return config
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from {GLOBAL_CONFIG_PATH}: {e}")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred loading config: {e}")
        return None

# Load config and set MAIN_COLLECTION_ROOT and TRACKER_DIR globally at script start
_initial_config = load_global_config()
if _initial_config:
    MAIN_COLLECTION_ROOT = _initial_config.get('main_collection_root', MAIN_COLLECTION_ROOT)
    TRACKER_DIR = _initial_config.get('tracker_directory', TRACKER_DIR)
else:
    logger.error("Failed to load initial configuration. Script may not function correctly.")


def _get_api_headers(instance_config):
    """Constructs standard API headers."""
    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {instance_config['api_key']}",
    }
    return headers

def get_knowledge_bases(instance_config):
    """Fetches all knowledge bases from Open WebUI."""
    url = f"{instance_config['url']}/api/v1/knowledge-bases"
    headers = _get_api_headers(instance_config)
    try:
        logger.debug(f"Fetching KBs from: {url}")
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        kbs = response.json().get('items', [])
        logger.debug(f"Successfully fetched {len(kbs)} knowledge bases.")
        return kbs
    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP error fetching knowledge bases from {url}: {e} - {response.text}")
    except requests.exceptions.ConnectionError as e:
        logger.error(f"Connection error fetching knowledge bases from {url}: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
    return []

def create_knowledge_base(name, instance_config, description=""):
    """Creates a new knowledge base in Open WebUI."""
    url = f"{instance_config['url']}/api/v1/knowledge-bases"
    headers = _get_api_headers(instance_config)
    headers["Content-Type"] = "application/json"
    data = {
        "name": name,
        "description": description,
        "embeddings_model": instance_config.get('embeddings_model', 'default_model'), # Use default or config
        "splitter_chunk_size": instance_config.get('splitter_chunk_size', 1024),
        "splitter_chunk_overlap": instance_config.get('splitter_chunk_overlap', 200),
        "is_active": True # Default to active
    }
    logger.debug(f"Creating KB: {name} with data: {data}")
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        new_kb = response.json()
        logger.info(f"Knowledge base '{name}' created successfully.")
        return new_kb
    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP error creating knowledge base '{name}': {e} - {response.text}")
    except requests.exceptions.ConnectionError as e:
        logger.error(f"Connection error creating knowledge base '{name}': {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred while creating KB: {e}")
    return None

def delete_knowledge_base_on_webui(kb_id, instance_config):
    """Deletes a knowledge base by ID from Open WebUI."""
    url = f"{instance_config['url']}/api/v1/knowledge-bases/{kb_id}"
    headers = _get_api_headers(instance_config)
    logger.debug(f"Deleting KB with ID: {kb_id} from {url}")
    try:
        response = requests.delete(url, headers=headers)
        response.raise_for_status()
        logger.info(f"Knowledge base with ID '{kb_id}' deleted successfully.")
        return True
    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP error deleting knowledge base ID '{kb_id}': {e} - {response.text}")
    except requests.exceptions.ConnectionError as e:
        logger.error(f"Connection error deleting knowledge base ID '{kb_id}': {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred deleting KB: {e}")
    return False

def upload_file_to_openwebui(file_path, original_filename, instance_config):
    """Uploads a file to Open WebUI and returns its ID."""
    url = f"{instance_config['url']}/api/v1/files/upload"
    headers = {
        "Authorization": f"Bearer {instance_config['api_key']}",
        "accept": "application/json",
    }
    # Ensure a proper filename is sent in the multipart form data
    files = {
        'file': (original_filename, open(file_path, 'rb'), 'application/octet-stream')
    }
    logger.debug(f"Uploading file: {original_filename} to {url}")
    try:
        response = requests.post(url, headers=headers, files=files)
        response.raise_for_status()
        file_info = response.json()
        file_id = file_info.get("id")
        if file_id:
            logger.debug(f"File '{original_filename}' uploaded successfully with ID: {file_id}")
            return file_id
        else:
            logger.error(f"File upload response missing ID for {original_filename}: {file_info}")
            return None
    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP error uploading file '{original_filename}': {e} - {response.text}")
    except requests.exceptions.ConnectionError as e:
        logger.error(f"Connection error uploading file '{original_filename}': {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred uploading file: {e}")
    finally:
        if 'file' in files:
            files['file'][1].close() # Ensure the file handle is closed
    return None

def add_file_to_knowledge_base(knowledge_base_id, file_id, instance_config):
    """Adds an uploaded file to a specified knowledge base."""
    url = f"{instance_config['url']}/api/v1/knowledge-bases/{knowledge_base_id}/add-file"
    headers = _get_api_headers(instance_config)
    headers["Content-Type"] = "application/json"
    data = {"file_id": file_id}
    logger.debug(f"Adding file ID {file_id} to KB ID {knowledge_base_id}")
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        logger.debug(f"File ID '{file_id}' added to knowledge base ID '{knowledge_base_id}' successfully.")
        return True
    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP error adding file ID '{file_id}' to KB ID '{knowledge_base_id}': {e} - {response.text}")
    except requests.exceptions.ConnectionError as e:
        logger.error(f"Connection error adding file ID '{file_id}' to KB ID '{knowledge_base_id}': {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred adding file to KB: {e}")
    return False

# --- Helper Functions for Text Processing ---

def extract_text_from_pdf(pdf_path):
    """
    Extracts text from a PDF file using the 'pypdf' library.
    Requires 'pip install pypdf'.
    """
    try:
        import pypdf
    except ImportError:
        logger.error("pypdf library not found. Please install it using 'pip install pypdf'.")
        return ""

    text = ""
    try:
        with open(pdf_path, 'rb') as f:
            reader = pypdf.PdfReader(f)
            for page in reader.pages:
                text += page.extract_text() or "" # Ensure empty string if no text
        logger.debug(f"Extracted text from PDF: {pdf_path}")
    except pypdf.errors.PdfReadError as e:
        logger.error(f"Error reading PDF file {pdf_path}: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred while extracting text from {pdf_path}: {e}")
    return text

def chunk_text(text):
    """
    Chunks text into smaller pieces based on a simple character splitter.
    This can be replaced with more sophisticated chunking strategies (e.g.,
    Langchain's RecursiveCharacterTextSplitter) if needed.
    """
    # Get chunking parameters from the global config, fallback to defaults
    global_config = load_global_config() # Reload to get latest (though usually static)
    chunk_size = global_config.get('open_webui_instances', {}).get(
        global_config.get('default_instance', 'default_instance'), {}
    ).get('splitter_chunk_size', 1024)
    chunk_overlap = global_config.get('open_webui_instances', {}).get(
        global_config.get('default_instance', 'default_instance'), {}
    ).get('splitter_chunk_overlap', 200)

    # Basic chunking: split by lines, then combine
    chunks = []
    current_chunk = []
    current_length = 0

    # Split text into manageable units (e.g., paragraphs or lines)
    # This example splits by double newline (paragraphs) for better semantic chunks
    paragraphs = text.split('\n\n')

    for paragraph in paragraphs:
        # Strip whitespace from paragraph
        paragraph = paragraph.strip()
        if not paragraph:
            continue

        if current_length + len(paragraph) + 2 < chunk_size: # +2 for potential \n\n
            current_chunk.append(paragraph)
            current_length += len(paragraph) + 2
        else:
            # If current_chunk is not empty, add it to chunks
            if current_chunk:
                chunks.append("\n\n".join(current_chunk))
            # Start a new chunk, applying overlap if possible
            if chunk_overlap > 0 and len(current_chunk) > 0:
                # Take the end of the previous chunk as overlap
                overlap_text = "\n\n".join(current_chunk)[-chunk_overlap:]
                current_chunk = [overlap_text, paragraph]
                current_length = len(overlap_text) + len(paragraph) + 2
            else:
                current_chunk = [paragraph]
                current_length = len(paragraph) + 2

    # Add the last chunk if it's not empty
    if current_chunk:
        chunks.append("\n\n".join(current_chunk))

    return chunks

# --- Helper Functions for File Tracking ---

def get_tracker_filepath(subject_dir, recursive_processing):
    """Determines the path for the JSON tracker file."""
    # Create a unique, sanitized identifier for the subject_dir to use in the tracker filename
    # This handles cases where subject_dir might contain characters not ideal for filenames
    # And ensures tracker files are unique across different subject_dirs, even if names collide
    # e.g., /path/to/my_books/SubjectA and /another/root/SubjectA would get different trackers
    # Also ensures it's relative to TRACKER_DIR
    relative_path_identifier = os.path.relpath(subject_dir, MAIN_COLLECTION_ROOT)
    # If subject_dir is outside MAIN_COLLECTION_ROOT, use its full path for the identifier
    if relative_path_identifier.startswith('..') or os.path.isabs(relative_path_identifier):
        relative_path_identifier = subject_dir.replace(os.sep, '_').replace(':', '')
    else:
        relative_path_identifier = relative_path_identifier.replace(os.sep, '_')

    # Sanitize further for filename
    relative_path_identifier = re.sub(r'[^a-zA-Z0-9_.-]', '', relative_path_identifier)
    relative_path_identifier = re.sub(r'_{2,}', '_', relative_path_identifier) # Collapse multiple underscores
    relative_path_identifier = relative_path_identifier.strip('_')

    # Prepend 'root' if the identifier ends up empty (e.g., if subject_dir == MAIN_COLLECTION_ROOT)
    if not relative_path_identifier:
        relative_path_identifier = "root"

    # Append mode and .json extension
    tracker_filename = (f"processed_files_"
                        f"{relative_path_identifier}_"
                        f"{'recursive' if recursive_processing else 'flat'}.json")
    return os.path.join(TRACKER_DIR, tracker_filename)


def load_processed_files_tracker(subject_dir, recursive_processing):
    """Loads the tracker file for a given subject directory and mode."""
    tracker_filepath = get_tracker_filepath(subject_dir, recursive_processing)
    if os.path.exists(tracker_filepath):
        try:
            with open(tracker_filepath, 'r', encoding='utf-8') as f:
                tracker_data = json.load(f)
            logger.debug(f"Loaded tracker from: {tracker_filepath}")
            return tracker_data
        except json.JSONDecodeError as e:
            logger.error(f"Error reading tracker file {tracker_filepath}: {e}. "
                         "Starting with an empty tracker.")
            # Optionally, back up the corrupted file before returning empty
        except Exception as e:
            logger.error(f"An unexpected error occurred loading tracker {tracker_filepath}: {e}. "
                         "Starting with an empty tracker.")
    else:
        logger.debug(f"Tracker file not found for '{subject_dir}' (recursive: "
                     f"{recursive_processing}). Creating new tracker.")

    # Initialize with metadata
    return {
        "knowledge_base_name": None,
        "open_webui_instance_id": None,
        "recurse_option": recursive_processing,
        "processed_files": {}
    }

def save_processed_files_tracker(tracker_data, subject_dir, recursive_processing):
    """Saves the tracker file for a given subject directory and mode."""
    tracker_filepath = get_tracker_filepath(subject_dir, recursive_processing)
    try:
        with open(tracker_filepath, 'w', encoding='utf-8') as f:
            json.dump(tracker_data, f, indent=4)
        logger.debug(f"Saved tracker to: {tracker_filepath}")
        return True
    except Exception as e:
        logger.error(f"Error saving tracker file {tracker_filepath}: {e}")
        return False

# --- Helper Functions for process_subject_directory ---

def generate_kb_name_from_path(resolved_subject_dir, recursive_processing):
    """
    Generates a unique knowledge base name based on the resolved subject
    directory path.
    Replaces path separators, spaces, and hyphens with underscores,
    and ensures consistent casing (capitalizing each component).
    """
    # Determine the identifier part of the path
    # Check if resolved_subject_dir is indeed under MAIN_COLLECTION_ROOT
    if (os.path.commonpath([MAIN_COLLECTION_ROOT, resolved_subject_dir]) ==
            MAIN_COLLECTION_ROOT):
        # If it's under MAIN_COLLECTION_ROOT, use its relative path from there
        kb_identifier_raw = os.path.relpath(resolved_subject_dir,
                                            MAIN_COLLECTION_ROOT)
    else:
        # If it's an absolute path NOT under MAIN_COLLECTION_ROOT
        # Remove leading separators to handle cases like /mnt/data -> mnt_data
        kb_identifier_raw = resolved_subject_dir.lstrip(os.sep)
        # Handle Windows drive letters, e.g., 'C:\' -> 'C'
        if os.name == 'nt' and len(kb_identifier_raw) > 1 and \
                kb_identifier_raw[1] == ':':
            kb_identifier_raw = kb_identifier_raw.replace(':', '', 1)

    # Force to lowercase for consistent KB name generation (e.g., science/physics
    # vs Science/Physics)
    kb_identifier = kb_identifier_raw.lower()

    # Sanitize the identifier: replace path separators, spaces, and hyphens
    # with underscores
    kb_identifier = kb_identifier.replace(os.sep, '_').replace(' ', '_').\
        replace('-', '_') # Ensure all hyphens become underscores

    # Remove any non-alphanumeric or underscore characters that might remain
    # and collapse multiple underscores
    kb_identifier = re.sub(r'[^a-z0-9_]', '', kb_identifier) # Keep only
                                                             # lowercase
                                                             # letters,
                                                             # digits,
                                                             # underscores
    kb_identifier = re.sub(r'_{2,}', '_', kb_identifier) # Collapse multiple
                                                         # underscores
    kb_identifier = kb_identifier.strip('_') # Remove leading/trailing
                                             # underscores

    # Ensure it's not empty, if it is (e.g. subject_dir was just
    # MAIN_COLLECTION_ROOT)
    if not kb_identifier:
        kb_identifier = "RootCollection" # Fallback name

    kb_name_suffix = "Recursive_KB" if recursive_processing else "Flat_KB"

    # Capitalize each word in the identifier for better readability
    # This happens AFTER lowercasing the full identifier for consistency
    kb_identifier_parts = [part.capitalize() for part in kb_identifier.split('_')]
    kb_full_name = f"{'_'.join(kb_identifier_parts)}_{kb_name_suffix}"

    return kb_full_name


def _get_or_create_knowledge_base(kb_full_name, instance_config, description):
    """
    Checks for an existing knowledge base by name or creates a new one.
    Returns the knowledge base ID if successful, None otherwise.
    """
    knowledge_bases = get_knowledge_bases(instance_config)
    knowledge_base_id = None
    for kb in knowledge_bases:
        # Compare names case-insensitively when checking for existing
        if kb.get("name").lower() == kb_full_name.lower():
            knowledge_base_id = kb.get("id")
            logger.info(f"Found existing knowledge base: '{kb_full_name}' "
                        f"with ID: {knowledge_base_id}")
            break

    if not knowledge_base_id:
        logger.info(f"Knowledge base '{kb_full_name}' not found. Creating it...")
        new_kb = create_knowledge_base(
            kb_full_name, instance_config, description)
        if new_kb:
            knowledge_base_id = new_kb.get("id")
        else:
            logger.error(f"Failed to create knowledge base '{kb_full_name}'.")

    return knowledge_base_id

def _collect_files_for_processing(subject_dir, recursive_processing):
    """
    Collects files from the subject directory based on recursion option.
    Returns a list of (full_file_path, relative_file_path_for_tracker) tuples.
    Filters out unsupported file types and common hidden/system files.
    """
    files_to_process = []
    # Define allowed extensions and patterns to ignore
    ALLOWED_EXTENSIONS = ['.pdf', '.txt', '.md']
    IGNORED_PATTERNS = [
        re.compile(r'^\._.*'),       # macOS resource fork files (._filename)
        re.compile(r'^\.DS_Store$'), # macOS directory metadata
        re.compile(r'^Thumbs\.db$'), # Windows thumbnail cache
        re.compile(r'^\~\$.*'),      # Windows temporary office files
                                     # (e.g., ~$doc.docx)
        re.compile(r'.*\.tmp$'),     # General temporary files
        re.compile(r'.*\.temp$'),    # General temporary files
        re.compile(r'.*\.bak$'),     # Backup files
        re.compile(r'.*\.swp$'),     # Vim swap files
        re.compile(r'.*\.log$')      # Log files
    ]

    logger.debug(f"Collecting files from '{subject_dir}' (recursive: "
                 f"{recursive_processing}).")

    def should_include_file(filename):
        # Ignore files matching any of the IGNORED_PATTERNS
        for pattern in IGNORED_PATTERNS:
            if pattern.match(filename):
                logger.debug(f"  Ignoring system/temp file: {filename}")
                return False

        # Check if the file extension is allowed
        _, ext = os.path.splitext(filename)
        if ext.lower() not in ALLOWED_EXTENSIONS:
            logger.debug(f"  Ignoring unsupported file extension: {filename}")
            return False

        return True

    if recursive_processing:
        for root, _, files in os.walk(subject_dir):
            for filename in files:
                if should_include_file(filename):
                    full_file_path = os.path.join(root, filename)
                    relative_file_path = os.path.relpath(full_file_path,
                                                         subject_dir)
                    files_to_process.append((full_file_path,
                                             relative_file_path))
    else: # Flat processing
        for filename in os.listdir(subject_dir):
            full_file_path = os.path.join(subject_dir, filename)
            if os.path.isfile(full_file_path) and should_include_file(filename):
                files_to_process.append((full_file_path, filename))

    logger.debug(f"Found {len(files_to_process)} relevant files in scope "
                 "after filtering.")
    return files_to_process

def _process_and_upload_file_chunks(file_path, relative_file_path,
                                       knowledge_base_id, instance_config,
                                       subject_dir):
    """
    Extracts text, chunks it, uploads chunks to Open WebUI, and adds them to
    the knowledge base. Returns the count of uploaded chunks.
    """
    base_name_for_chunk, ext = os.path.splitext(os.path.basename(file_path))

    extracted_text = ""
    if ext.lower() == '.pdf':
        extracted_text = extract_text_from_pdf(file_path)
    elif ext.lower() in ['.txt', '.md']:
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                extracted_text = f.read()
            logger.debug(f"Read text from file: {relative_file_path}")
        except Exception as e:
            logger.error(f"Error reading text from {relative_file_path}: {e}")
    else:
        logger.warning(f"Skipping unsupported file type: {relative_file_path}")
        return 0

    # --- START OF REVISED CODE ---
    # Check for Empty Extracted Text
    if not extracted_text.strip(): # Use .strip() to account for whitespace-only
        logger.warning(f"No usable text extracted from '{relative_file_path}'. "
                         "Skipping chunking and upload.")
        return 0

    chunks = chunk_text(extracted_text)
    logger.info(f"Split '{relative_file_path}' into {len(chunks)} chunks.")

    # Check for Empty Chunks
    if not chunks:
        logger.warning(f"No chunks generated for '{relative_file_path}'. "
                         "Skipping upload to knowledge base.")
        return 0
    # --- END OF REVISED CODE ---

    uploaded_chunks_count = 0
    temp_chunk_dir = os.path.join(subject_dir, "temp_chunks")
    os.makedirs(temp_chunk_dir, exist_ok=True)
    logger.debug(f"Created temp chunk directory: {temp_chunk_dir}")

    for i, chunk in enumerate(chunks):
        chunk_filename_for_upload = (f"{base_name_for_chunk}_chunk_{i+1}"
                                     f"{ext}")
        temp_chunk_path = os.path.join(temp_chunk_dir,
                                       chunk_filename_for_upload)

        with open(temp_chunk_path, 'w', encoding='utf-8') as f:
            f.write(chunk)
        logger.debug(f"Saved chunk {i+1} to {temp_chunk_path}")

        logger.info(f"  Uploading chunk {i+1}/{len(chunks)}: "
                    f"{chunk_filename_for_upload}")
        file_id = upload_file_to_openwebui(temp_chunk_path,
                                           chunk_filename_for_upload,
                                           instance_config)
        if file_id:
            add_file_to_knowledge_base(knowledge_base_id, file_id,
                                       instance_config)
            uploaded_chunks_count += 1
        else:
            logger.error(f"  Failed to get file ID for chunk "
                         f"{chunk_filename_for_upload}. Skipping adding to KB.")

        os.remove(temp_chunk_path)
        logger.debug(f"Removed temporary chunk file: {temp_chunk_path}")
        time.sleep(0.1)

    # Clean up temp_chunks directory if empty
    if not os.listdir(temp_chunk_dir):
        try:
            os.rmdir(temp_chunk_dir)
            logger.debug(f"Removed empty temp_chunks directory: "
                         f"{temp_chunk_dir}")
        except OSError as e:
            logger.warning(f"Could not remove empty temp_chunks directory "
                           f"{temp_chunk_dir}: {e}")

    # --- START OF REVISED CODE ---
    # Check if any chunks were successfully uploaded
    if uploaded_chunks_count == 0:
        logger.warning(f"No chunks were successfully uploaded for '{relative_file_path}'. "
                         "This file will not contribute to the knowledge base.")
    # --- END OF REVISED CODE ---

    return uploaded_chunks_count

# --- Main Automation Script ---

def perform_dry_run(resolved_subject_dir, recursive_processing):
    """
    Performs a dry run, reporting what would be processed without actual changes.
    """
    kb_full_name = generate_kb_name_from_path(resolved_subject_dir,
                                             recursive_processing)

    # Use os.path.relpath to get a display name relative to MAIN_COLLECTION_ROOT
    # This will always work now that both are absolute.
    if (os.path.commonpath([MAIN_COLLECTION_ROOT, resolved_subject_dir]) ==
            MAIN_COLLECTION_ROOT):
        # If it's under MAIN_COLLECTION_ROOT, use its relative path from there
        display_subject_name = os.path.relpath(resolved_subject_dir,
                                                 MAIN_COLLECTION_ROOT)
    else:
        # If the subject_dir is outside MAIN_COLLECTION_ROOT, use its full path
        # for display
        display_subject_name = resolved_subject_dir


    # --- FIX: Ensure INFO messages are displayed for dry run ---
    # Store current level, set to INFO, then restore.
    # This ensures INFO is seen even if default is WARNING/ERROR,
    # but still allows DEBUG if --verbose_log is on.
    original_logger_level = logger.level
    logger.setLevel(logging.INFO) # Always ensure INFO is visible for dry run

    logger.info(f"\n--- Dry Run for Subject: '{display_subject_name}' "
                f"({'Recursive' if recursive_processing else 'Flat'}) ---")
    logger.info(f"  Expected Knowledge Base Name: '{kb_full_name}'")

    tracker_data = load_processed_files_tracker(resolved_subject_dir,
                                                recursive_processing)
    files_to_check = _collect_files_for_processing(resolved_subject_dir,
                                                   recursive_processing)

    total_files_in_scope = len(files_to_check)
    new_or_modified_files = 0
    skipped_files = 0

    logger.info(f"  Total files in scope: {total_files_in_scope}")
    if total_files_in_scope == 0:
        logger.info("  No files found in this subject directory for processing.")
        # Reset logger level before returning
        logger.setLevel(original_logger_level)
        return {
            "kb_name": kb_full_name,
            "display_name": display_subject_name,
            "mode": "Recursive" if recursive_processing else "Flat",
            "total_files": 0,
            "new_modified": 0,
            "skipped": 0
        }

    for file_path, relative_file_path in files_to_check:
        current_mtime = datetime.fromtimestamp(
            os.path.getmtime(file_path)).isoformat()

        if (relative_file_path in tracker_data["processed_files"] and
                tracker_data["processed_files"][relative_file_path]
                ["last_modified"] == current_mtime):
            skipped_files += 1
            logger.debug(f"    Would skip '{relative_file_path}': Already "
                         "processed and not modified.")
        else:
            new_or_modified_files += 1
            logger.debug(f"    Would process '{relative_file_path}': New or "
                         "modified.")

    logger.info(f"  Files that would be processed (new/modified): "
                f"{new_or_modified_files}")
    logger.info(f"  Files that would be skipped (already processed): "
                f"{skipped_files}")

    # Reset logger level before returning
    logger.setLevel(original_logger_level)

    return {
        "kb_name": kb_full_name,
        "display_name": display_subject_name,
        "mode": "Recursive" if recursive_processing else "Flat",
        "total_files": total_files_in_scope,
        "new_modified": new_or_modified_files,
        "skipped": skipped_files
    }


def process_subject_directory(resolved_subject_dir, instance_id,
                                 recursive_processing=False):
    """
    Processes documents in a subject directory, chunks them, and adds to
    Open WebUI KB. Detects new/modified files and updates tracking.

    Args:
        resolved_subject_dir (str): FULL path to the directory containing
                                    documents for a specific subject.
        instance_id (str): The ID of the Open WebUI instance from config.json
                           to target.
        recursive_processing (bool): If True, process subdirectories
                                    recursively.
    """
    global_config = load_global_config()
    if not global_config:
        return False

    instance_config = global_config['open_webui_instances'].get(instance_id)
    if not instance_config:
        logger.error(f"Open WebUI instance ID '{instance_id}' not found in "
                     f"{GLOBAL_CONFIG_PATH}.")
        return False

    kb_full_name = generate_kb_name_from_path(resolved_subject_dir,
                                             recursive_processing)

    # Use os.path.relpath to get a display name relative to MAIN_COLLECTION_ROOT
    if (os.path.commonpath([MAIN_COLLECTION_ROOT, resolved_subject_dir]) ==
            MAIN_COLLECTION_ROOT):
        display_subject_name = os.path.relpath(resolved_subject_dir,
                                                 MAIN_COLLECTION_ROOT)
    else:
        display_subject_name = resolved_subject_dir

    # --- FIX: Ensure INFO messages are displayed for processing ---
    original_logger_level = logger.level
    logger.setLevel(logging.INFO) # Always ensure INFO is visible for processing

    logger.info(f"Starting processing for subject: '{display_subject_name}' "
                f"({'Recursive' if recursive_processing else 'Flat'}) from "
                f"directory: '{resolved_subject_dir}'")
    logger.info(f"Targeting Open WebUI instance: '{instance_id}' at "
                f"{instance_config['url']}")
    logger.info(f"Knowledge Base Name: '{kb_full_name}'")

    # 1. Load file tracking data for this subject and mode
    tracker_data = load_processed_files_tracker(resolved_subject_dir,
                                                recursive_processing)
    if not tracker_data.get("knowledge_base_name"):
        tracker_data["knowledge_base_name"] = kb_full_name
    if not tracker_data.get("open_webui_instance_id"):
        tracker_data["open_webui_instance_id"] = instance_id
    if not tracker_data.get("recurse_option"):
        tracker_data["recurse_option"] = recursive_processing

    # 2. Get or Create Knowledge Base
    kb_description = (f"Knowledge base for documents from "
                      f"'{display_subject_name}' "
                      f"({'Recursive' if recursive_processing else 'Flat'}). "
                      f"Source: {resolved_subject_dir}")
    knowledge_base_id = _get_or_create_knowledge_base(
        kb_full_name, instance_config, kb_description)
    if not knowledge_base_id:
        logger.error("Could not obtain knowledge base ID. Exiting processing "
                     "for this subject.")
        logger.setLevel(original_logger_level) # Reset level
        return False

    # 3. Collect files based on recursion option
    files_to_process = _collect_files_for_processing(resolved_subject_dir,
                                                       recursive_processing)

    # 4. Process each detected file
    processed_any_new_files = False
    for file_path, relative_file_path in files_to_process:
        current_mtime = datetime.fromtimestamp(
            os.path.getmtime(file_path)).isoformat()

        # Check if file needs processing (based on its relative path)
        if (relative_file_path in tracker_data["processed_files"] and
                tracker_data["processed_files"][relative_file_path]
                ["last_modified"] == current_mtime):
            logger.info(f"Skipping '{relative_file_path}': Already processed "
                        "and not modified.")
            continue

        processed_any_new_files = True
        logger.info(f"\nProcessing file: {relative_file_path}")

        uploaded_chunks_count = _process_and_upload_file_chunks(
            file_path, relative_file_path, knowledge_base_id,
            instance_config, resolved_subject_dir)

        # 5. Update tracker data for this file (using relative path as key)
        tracker_data["processed_files"][relative_file_path] = {
            "last_modified": current_mtime,
            "processed_chunks_count": uploaded_chunks_count
        }
        # Save tracker after each file, in case of script interruption
        save_processed_files_tracker(tracker_data, resolved_subject_dir,
                                     recursive_processing)
        logger.info(f"Successfully processed and updated tracker for "
                    f"'{relative_file_path}'. Uploaded {uploaded_chunks_count} "
                    "chunks.")

    if processed_any_new_files:
        logger.info(f"\nFinished processing all documents for subject "
                    f"'{display_subject_name}' "
                    f"({'Recursive' if recursive_processing else 'Flat'}).")
    else:
        logger.info(f"\nNo new or modified documents found for subject "
                    f"'{display_subject_name}' "
                    f"({'Recursive' if recursive_processing else 'Flat'}).")

    logger.setLevel(original_logger_level) # Reset level
    return True


# --- New KB Management Functions ---

def _confirm_action(prompt, force_mode=False):
    """Helper function to ask for case-sensitive 'Yes' confirmation."""
    if force_mode:
        # Changed to print for this specific instructional output, as logger.info
        # would be suppressed by default WARN level if not verbose.
        print(f"Force mode active: Skipping confirmation for: {prompt}")
        return True # Automatically confirm if force_mode is True
    reply = input(f"{prompt} (Type 'Yes' to confirm): ")
    return reply == "Yes"

def wipe_all_knowledge_bases_for_instance(instance_id, force_mode=False):
    """Wipes all knowledge bases from a given Open WebUI instance."""
    global_config = load_global_config()
    if not global_config:
        return

    instance_config = global_config['open_webui_instances'].get(instance_id)
    if not instance_config:
        logger.error(f"Open WebUI instance ID '{instance_id}' not found in "
                     f"{GLOBAL_CONFIG_PATH}.")
        return

    knowledge_bases = get_knowledge_bases(instance_config)
    if not knowledge_bases:
        logger.info("No knowledge bases found on Open WebUI to delete.")
        return

    # --- FIX: Ensure INFO messages are displayed for wipe actions ---
    original_logger_level = logger.level
    logger.setLevel(logging.INFO)

    logger.warning(f"ATTENTION: You are about to DELETE ALL knowledge bases "
                   f"on Open WebUI instance: {instance_config['url']}\n")
    logger.warning("The following knowledge bases would be deleted:")
    for kb in knowledge_bases:
        logger.warning(f"- '{kb.get('name')}' (ID: {kb.get('id')})")
    logger.warning("\nThis action is irreversible.")

    if not _confirm_action("Do you really want to proceed?", force_mode):
        logger.info("Operation cancelled by user.")
        logger.setLevel(original_logger_level) # Reset level
        return

    deleted_count = 0
    for kb in knowledge_bases:
        kb_name = kb.get("name")
        kb_id = kb.get("id")
        logger.info(f"Attempting to delete knowledge base: '{kb_name}' "
                    f"(ID: {kb_id})...")
        if delete_knowledge_base_on_webui(kb_id, instance_config):
            deleted_count += 1
            # Note: We do NOT delete local processed_files_*.json here
            # as we don't know which subject_dirs correspond to these KBs
            # globally.
        time.sleep(0.5) # Small delay

    logger.info(f"Finished wiping. {deleted_count} knowledge bases deleted.")
    logger.warning("NOTE: Local tracking files (processed_files_*.json) for "
                   "deleted knowledge bases still exist on your file system. "
                   "You may want to manually delete them if you wish to clear "
                   "all local history for these KBs.")

    logger.setLevel(original_logger_level) # Reset level


def wipe_specific_subject_knowledge_base(resolved_subject_dir, instance_id,
                                         recursive_mode, force_mode=False):
    """
    Wipes a specific knowledge base (flat or recursive version) from Open WebUI
    and deletes its local tracking file.
    """
    global_config = load_global_config()
    if not global_config:
        return

    instance_config = global_config['open_webui_instances'].get(instance_id)
    if not instance_config:
        logger.error(f"Open WebUI instance ID '{instance_id}' not found in "
                     f"{GLOBAL_CONFIG_PATH}.")
        return

    kb_full_name = generate_kb_name_from_path(resolved_subject_dir,
                                             recursive_mode)

    # Use os.path.relpath to get a display name relative to MAIN_COLLECTION_ROOT
    if (os.path.commonpath([MAIN_COLLECTION_ROOT, resolved_subject_dir]) ==
            MAIN_COLLECTION_ROOT):
        display_subject_name = os.path.relpath(resolved_subject_dir,
                                                 MAIN_COLLECTION_ROOT)
    else:
        display_subject_name = resolved_subject_dir

    # --- FIX: Ensure INFO messages are displayed for wipe actions ---
    original_logger_level = logger.level
    logger.setLevel(logging.INFO)

    logger.warning(f"ATTENTION: You are about to DELETE knowledge base: "
                   f"'{kb_full_name}' (derived from subject: "
                   f"'{display_subject_name}') "
                   f"on Open WebUI instance: {instance_config['url']}\n")

    # Construct the local tracker file path for the warning message
    tracker_filename = (f"processed_files_"
                        f"{os.path.relpath(resolved_subject_dir, MAIN_COLLECTION_ROOT).replace(os.sep, '_').replace(':', '')}_" # Adjusted for tracker name consistency with generate_kb_name_from_path
                        f"{'recursive' if recursive_mode else 'flat'}.json")
    tracker_filepath = os.path.join(TRACKER_DIR, tracker_filename) # Ensure tracker is in TRACKER_DIR


    logger.warning(f"The following knowledge base would be deleted remotely:")
    logger.warning(f"- '{kb_full_name}'")
    logger.warning(f"\nThis will also delete the local tracking file: "
                   f"'{tracker_filepath}'\n")
    logger.warning("This action is irreversible.")


    if not _confirm_action("Do you really want to proceed?", force_mode):
        logger.info("Operation cancelled by user.")
        logger.setLevel(original_logger_level) # Reset level
        return

    knowledge_bases = get_knowledge_bases(instance_config)
    knowledge_base_id = None
    for kb in knowledge_bases:
        if kb.get("name").lower() == kb_full_name.lower():
            knowledge_base_id = kb.get("id")
            break

    if not knowledge_base_id:
        logger.info(f"Knowledge base '{kb_full_name}' not found on Open WebUI."
                    " Nothing to delete there.")
    else:
        logger.info(f"Found knowledge base '{kb_full_name}' with ID: "
                    f"{knowledge_base_id}. Attempting to delete...")
        if delete_knowledge_base_on_webui(knowledge_base_id, instance_config):
            logger.info(f"Successfully deleted knowledge base '{kb_full_name}' "
                        "from Open WebUI.")
        else:
            logger.error(f"Failed to delete knowledge base '{kb_full_name}' "
                         "from Open WebUI.")
            logger.setLevel(original_logger_level) # Reset level
            return # Exit if deletion failed on WebUI side

    # Delete local tracker file regardless of WebUI deletion success if found
    # Use the already constructed tracker_filepath
    if os.path.exists(tracker_filepath):
        try:
            os.remove(tracker_filepath)
            logger.info(f"Successfully deleted local tracker file: "
                        f"'{tracker_filepath}'.")
        except OSError as e:
            logger.error(f"Error deleting local tracker file "
                         f"'{tracker_filepath}': {e}")
    else:
        logger.info(f"Local tracker file '{tracker_filepath}' not found. "
                    "Nothing to delete.")

    logger.info(f"Finished wiping specific knowledge base '{kb_full_name}'.")
    logger.setLevel(original_logger_level) # Reset level


# --- Main Execution with Command-Line Argument Parsing ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""
This script helps manage Open WebUI Knowledge Bases by automating the
processing and uploading of documents from your local file system, and
providing tools to list and manage these knowledge bases.

Key Features:
  - Populate: Adds new or modified documents from a specified subject
    directory (or all subjects) to Open WebUI Knowledge Bases. It
    intelligently skips already processed and unchanged files. Documents
    are chunked for optimal retrieval.
  - Dry Run: Simulates the populate process, showing which files would
    be processed without making any actual changes to Open WebUI or
    local tracking files.
  - Wipe All KBs: Deletes all knowledge bases on a targeted Open WebUI
    instance. Use with extreme caution.
  - Wipe Subject KB: Deletes a specific knowledge base (and its local
    tracking file) associated with a given subject directory and its
    processing mode (flat or recursive).
  - Lookup KB Name: Helps determine the exact knowledge base name that
    would be generated for a specific subject directory and recursion
    mode.
  - List KBs: Lists all knowledge bases available on a specified Open
    WebUI instance, along with their details.

File Organization:
  Documents should be organized under a root collection directory (default:
  './my_books'). Each immediate subdirectory within this root is
  considered a 'subject directory'. For example:
  ./my_books/Mathematics/Set Theory/my_document.pdf
  ./my_books/Physics/Quantum Mechanics/another_document.txt

  Knowledge Base Naming:
  The script automatically generates knowledge base names based on the
  subject directory path and the processing mode (--flat or --recursive).
  This ensures unique and descriptive names. For example,
  './my_books/Mathematics/Set Theory' processed recursively will map to
  a KB named 'Mathematics_Set_Theory_Recursive_KB'.

Local Tracking:
  The script maintains a '.json' file within each subject directory
  (e.g., 'processed_files_flat.json' or 'processed_files_recursive.json')
  to track which files have been processed and their last modification
  times. This prevents re-uploading unchanged content.
""",
        formatter_class=argparse.RawTextHelpFormatter # Preserve formatting
                                                     # for help text
    )

    # Global options
    parser.add_argument(
        '--instance',
        type=str,
        help='Open WebUI instance ID from config.json (e.g., "dev_instance").\n'
             'Defaults to "default_instance" from config.json if not set.'
    )
    parser.add_argument(
        '--force_action', # Changed from --force to --force_action
        action='store_true',
        help='Bypass confirmation prompts for destructive operations.\n'
             'USE WITH CAUTION!'
    )
    parser.add_argument(
        '--verbose_log', # Changed from --verbose to --verbose_log
        action='store_true',
        help='Enable verbose logging (DEBUG level).'
    )

    # Mutually exclusive group for modes
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        '--populate_kb', # Changed from --populate to --populate_kb
        action='store_true',
        help='Populate knowledge base(s) with documents.'
    )
    mode_group.add_argument(
        '--dry_run', # Changed from --dry-run to --dry_run
        action='store_true',
        help='Perform a dry run for populate mode.\n'
             'Shows what would be processed without making changes.'
    )
    mode_group.add_argument(
        '--wipe_all_kbs', # Changed from --wipe-all-kbs to --wipe_all_kbs
        action='store_true',
        help='Wipe (delete) ALL knowledge bases from the specified Open WebUI\n'
             'instance.'
    )
    mode_group.add_argument(
        '--wipe_subject_kb', # Changed from --wipe-subject-kb to --wipe_subject_kb
        action='store_true',
        help='Wipe (delete) the knowledge base for a specific subject dir\n'
             'and its local tracker.'
    )
    mode_group.add_argument(
        '--lookup_kb_name', # Changed from --lookup-kb-name to --lookup_kb_name
        action='store_true',
        help='Look up the expected knowledge base name for a given subject\n'
             'directory and recursion mode.'
    )
    mode_group.add_argument(
        '--list_kbs', # Changed from --list-kbs to --list_kbs
        action='store_true',
        help='List all knowledge bases on the specified Open WebUI instance.'
    )


    # Arguments dependent on specific modes
    # For --populate_kb, --dry_run, --wipe_subject_kb, --lookup_kb_name
    parser.add_argument(
        '--subject_dir',
        type=str,
        help=f'Relative path to the subject directory within '
             f'"{os.path.basename(MAIN_COLLECTION_ROOT)}",\n'
             'e.g., "Mathematics/Set Theory". Required for single-subject\n'
             'operations (--populate_kb, --dry_run, --wipe_subject_kb,\n'
             '--lookup_kb_name).'
    )
    parser.add_argument(
        '--recursive',
        action='store_true',
        help='Process documents in the subject directory and all its\n'
             'subdirectories. Affects KB name and tracking file.\n'
             'Mutually exclusive with --flat.'
    )
    parser.add_argument(
        '--flat',
        action='store_true',
        help='Process only documents directly in the subject directory\n'
             '(no subdirectories). Affects KB name and tracking file.\n'
             'Mutually exclusive with --recursive. This is the default for\n'
             'single-subject operations if neither --recursive nor --flat\n'
             'is specified.'
    )

    # Special argument for --populate_kb if no subject_dir is given (process all)
    parser.add_argument(
        '--all_subjects', # Changed from --all-subjects to --all_subjects
        action='store_true',
        help='When used with --populate_kb or --dry_run (no --subject_dir),\n'
             'processes all subject directories under MAIN_COLLECTION_ROOT\n'
             '(both flat and recursive modes).'
    )


    args = parser.parse_args()

    # Set logging level based on --verbose_log flag
    if args.verbose_log: # Access changed to underscore
        logger.setLevel(logging.DEBUG)
        # Also adjust root logger if other modules might use it
        logging.getLogger().setLevel(logging.DEBUG)


    # --- Mode-specific argument validation and execution ---

    # Resolve subject_dir path early if provided
    resolved_subject_dir = None
    if args.subject_dir:
        # If the subject_dir is an absolute path, use it directly
        if os.path.isabs(args.subject_dir):
            resolved_subject_dir = os.path.abspath(args.subject_dir)
        else:
            # If relative, join with the already absolute MAIN_COLLECTION_ROOT
            resolved_subject_dir = os.path.join(MAIN_COLLECTION_ROOT,
                                                args.subject_dir)
        # Verify the resolved directory exists before proceeding with
        # subject-specific modes
        if not os.path.isdir(resolved_subject_dir):
            logger.error(f"Subject directory does not exist or is not a "
                         f"directory: '{resolved_subject_dir}'")
            exit(1)

    # Determine target_recursive_mode. This variable will determine the KB suffix.
    target_recursive_mode = False
    if args.recursive and args.flat:
        parser.error("Cannot specify both --recursive and --flat. Choose one.")
    elif args.recursive:
        target_recursive_mode = True
    elif args.flat:
        target_recursive_mode = False
    # If neither recursive nor flat is specified, default behavior depends on
    # mode:
    # - lookup_kb_name, dry_run, wipe_subject_kb for single subject: default to flat
    # - populate_kb for single subject: default to flat
    # - populate_kb/dry_run for all subjects: both recursive and flat are processed
    #   sequentially (handled in loop below)

    global_config = load_global_config()

    if not global_config:
        logger.error("Global configuration could not be loaded. Please check "
              "config.json.")
        exit(1)


    # If --instance is not specified, try to get default from config
    instance_id = args.instance
    if not instance_id:
        instance_id = global_config.get("default_instance")
        if not instance_id:
            logger.error("No Open WebUI instance specified via --instance, "
                         "and 'default_instance' not found in config.json. "
                         "Please specify an instance or set a default.")
            exit(1)
        # Removed this INFO line, as it was unwanted.
        # logger.info(f"Using default Open WebUI instance: '{instance_id}'")

    if instance_id not in global_config['open_webui_instances']:
        logger.error(f"Specified instance ID '{instance_id}' not found in "
                     f"'open_webui_instances' in {GLOBAL_CONFIG_PATH}.")
        exit(1)


    # --- Execute based on mode ---
    if args.populate_kb: # Access changed to underscore
        # Only set INFO level if not already DEBUG (from --verbose_log)
        if logger.level > logging.INFO:
            logger.setLevel(logging.INFO)

        if args.all_subjects: # Access changed to underscore
            if args.subject_dir:
                parser.error("Cannot specify both --all_subjects and "
                             "--subject_dir with --populate_kb.")
            logger.info(f"\nProcessing all subjects under "
                        f"'{MAIN_COLLECTION_ROOT}'.")

            processed_any_subject = False
            for subject_name in os.listdir(MAIN_COLLECTION_ROOT):
                current_subject_dir = os.path.join(MAIN_COLLECTION_ROOT,
                                                   subject_name)
                if os.path.isdir(current_subject_dir):
                    logger.info(f"\n--- Processing ALL Subject: "
                                f"'{subject_name}' (Flat Mode) ---")
                    if process_subject_directory(current_subject_dir,
                                                   instance_id,
                                                   recursive_processing=False):
                        processed_any_subject = True

                    logger.info(f"\n--- Processing ALL Subject: "
                                f"'{subject_name}' (Recursive Mode) ---")
                    if process_subject_directory(current_subject_dir,
                                                   instance_id,
                                                   recursive_processing=True):
                        processed_any_subject = True

            if not processed_any_subject:
                logger.info(f"No valid subject directories found under "
                            f"'{MAIN_COLLECTION_ROOT}' to process.")

        elif resolved_subject_dir: # Single subject_dir processing
            # If no recursive/flat specified for single subject, default to flat
            if not args.recursive and not args.flat:
                target_recursive_mode = False # Default to flat
                logger.info("Neither --recursive nor --flat specified for "
                            "single subject. Defaulting to --flat.")

            process_subject_directory(resolved_subject_dir, instance_id,
                                     target_recursive_mode)
        else:
            parser.error("To use --populate_kb, you must specify either "
                         "--subject_dir or --all_subjects.")

        # Reset logging level after operation if it was temporarily changed
        if not args.verbose_log: # Only reset if --verbose_log wasn't used
            logger.setLevel(logging.WARNING)

    elif args.dry_run: # Access changed to underscore
        # Only set INFO level if not already DEBUG (from --verbose_log)
        if logger.level > logging.INFO:
            logger.setLevel(logging.INFO)

        if args.all_subjects: # Access changed to underscore
            if args.subject_dir:
                parser.error("Cannot specify both --all_subjects and "
                             "--subject_dir with --dry_run.")
            logger.info(f"\nPerforming dry run for all subjects under "
                        f"'{MAIN_COLLECTION_ROOT}'.")

            for subject_name in os.listdir(MAIN_COLLECTION_ROOT):
                current_subject_dir = os.path.join(MAIN_COLLECTION_ROOT,
                                                   subject_name)
                if os.path.isdir(current_subject_dir):
                    # Dry run for flat mode
                    perform_dry_run(current_subject_dir,
                                    recursive_processing=False)
                    # Dry run for recursive mode
                    perform_dry_run(current_subject_dir,
                                    recursive_processing=True)
        elif resolved_subject_dir: # Single subject_dir dry run
            # If no recursive/flat specified for single subject, default to flat
            if not args.recursive and not args.flat:
                target_recursive_mode = False # Default to flat
                logger.info("Neither --recursive nor --flat specified for "
                            "single subject dry run. Defaulting to --flat.")

            perform_dry_run(resolved_subject_dir, target_recursive_mode)
        else:
            parser.error("To use --dry_run, you must specify either "
                         "--subject_dir or --all_subjects.")

        # Reset logging level after operation if it was temporarily changed
        if not args.verbose_log: # Only reset if --verbose_log wasn't used
            logger.setLevel(logging.WARNING)

    elif args.wipe_all_kbs: # Access changed to underscore
        # Only set INFO level if not already DEBUG (from --verbose_log)
        if logger.level > logging.INFO:
            logger.setLevel(logging.INFO)

        wipe_all_knowledge_bases_for_instance(instance_id, args.force_action) # Access changed to underscore
        # Reset logging level after operation if it was temporarily changed
        if not args.verbose_log: # Only reset if --verbose_log wasn't used
            logger.setLevel(logging.WARNING)

    elif args.wipe_subject_kb: # Access changed to underscore
        if not resolved_subject_dir:
            parser.error("To use --wipe_subject_kb, you must specify "
                         "--subject_dir.")

        # If no recursive/flat specified for single subject, default to flat
        if not args.recursive and not args.flat:
            target_recursive_mode = False # Default to flat
            logger.info("Neither --recursive nor --flat specified for wipe. "
                        "Defaulting to --flat mode KB.")

        # Only set INFO level if not already DEBUG (from --verbose_log)
        if logger.level > logging.INFO:
            logger.setLevel(logging.INFO)

        wipe_specific_subject_knowledge_base(resolved_subject_dir, instance_id,
                                             target_recursive_mode, args.force_action) # Access changed to underscore
        # Reset logging level after operation if it was temporarily changed
        if not args.verbose_log: # Only reset if --verbose_log wasn't used
            logger.setLevel(logging.WARNING)

    elif args.lookup_kb_name: # Access changed to underscore
        if not resolved_subject_dir:
            parser.error("To use --lookup_kb_name, you must specify "
                         "--subject_dir.")

        # If no recursive/flat specified for single subject, default to flat
        if not args.recursive and not args.flat:
            target_recursive_mode = False # Default to flat
            logger.info("Neither --recursive nor --flat specified for KB name "
                        "lookup. Defaulting to --flat mode KB.")

        # Only set INFO level if not already DEBUG (from --verbose_log)
        if logger.level > logging.INFO:
            logger.setLevel(logging.INFO)

        def lookup_kb_name_cli(subject_dir_path, recursive_mode_val):
            kb_name = generate_kb_name_from_path(subject_dir_path, recursive_mode_val)
            logger.info(f"The generated knowledge base name for '{subject_dir_path}' "
                        f"with recursion={recursive_mode_val} is: '{kb_name}'")

        lookup_kb_name_cli(resolved_subject_dir, target_recursive_mode)
        # Reset logging level after operation if it was temporarily changed
        if not args.verbose_log: # Only reset if --verbose_log wasn't used
            logger.setLevel(logging.WARNING)

    elif args.list_kbs: # Access changed to underscore
        # Only set INFO level if not already DEBUG (from --verbose_log)
        if logger.level > logging.INFO:
            logger.setLevel(logging.INFO)

        def list_knowledge_bases_cli(instance_id_val):
            instance_config = global_config['open_webui_instances'].get(instance_id_val)
            if not instance_config:
                logger.error(f"Open WebUI instance ID '{instance_id_val}' not found.")
                return

            logger.info(f"\n--- Listing Knowledge Bases on {instance_config['url']} ---")
            knowledge_bases = get_knowledge_bases(instance_config)
            if knowledge_bases:
                for kb in knowledge_bases:
                    logger.info(f"  Name: {kb.get('name')}")
                    logger.info(f"  ID:   {kb.get('id')}")
                    logger.info(f"  Description: {kb.get('description', 'N/A')}")
                    logger.info("-" * 40)
            else:
                logger.info("No knowledge bases found.")
            logger.info("--- End of List ---")

        list_knowledge_bases_cli(instance_id)
        # Reset logging level after operation if it was temporarily changed
        if not args.verbose_log: # Only reset if --verbose_log wasn't used
            logger.setLevel(logging.WARNING)
