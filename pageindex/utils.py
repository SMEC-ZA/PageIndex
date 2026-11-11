import litellm
import logging
import os
import textwrap
from datetime import datetime
import time
import json
import PyPDF2
import copy
import asyncio
import pymupdf
from io import BytesIO
from dotenv import load_dotenv
load_dotenv()
import logging
import yaml
from pathlib import Path
from types import SimpleNamespace as config
import re

# Backward compatibility: support CHATGPT_API_KEY as alias for OPENAI_API_KEY
if not os.getenv("OPENAI_API_KEY") and os.getenv("CHATGPT_API_KEY"):
    os.environ["OPENAI_API_KEY"] = os.getenv("CHATGPT_API_KEY")

litellm.drop_params = True

def count_tokens(text, model=None):
    if not text:
        return 0
    return litellm.token_counter(model=model, text=text)


def llm_completion(model, prompt, chat_history=None, return_finish_reason=False):
    if model:
        model = model.removeprefix("litellm/")
        if "/" not in model:
            model = "openai/" + model
    api_base = os.getenv("OPENAI_API_BASE") or os.getenv("OPENAI_BASE_URL") or os.getenv("LLM_BASE_URL")
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY") or os.getenv("LITELLM_MASTER_KEY")
    max_retries = 10
    messages = list(chat_history) + [{"role": "user", "content": prompt}] if chat_history else [{"role": "user", "content": prompt}]
    for i in range(max_retries):
        try:
            kwargs = {
                "model": model,
                "messages": messages,
                "temperature": 0
            }
            if api_base:
                kwargs["api_base"] = api_base
            if api_key:
                kwargs["api_key"] = api_key
            response = litellm.completion(**kwargs)
            content = response.choices[0].message.content
            if return_finish_reason:
                finish_reason = "max_output_reached" if response.choices[0].finish_reason == "length" else "finished"
                return content, finish_reason
            return content
        except Exception as e:
            print('************* Retrying *************')
            logging.error(f"Error: {e}")
            if i < max_retries - 1:
                time.sleep(1)
            else:
                logging.error('Max retries reached for prompt: ' + prompt)
                if return_finish_reason:
                    return "", "error"
                return ""



async def llm_acompletion(model, prompt):
    if model:
        model = model.removeprefix("litellm/")
        if "/" not in model:
            model = "openai/" + model
    api_base = os.getenv("OPENAI_API_BASE") or os.getenv("OPENAI_BASE_URL") or os.getenv("LLM_BASE_URL")
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY") or os.getenv("LITELLM_MASTER_KEY")
    max_retries = 10
    messages = [{"role": "user", "content": prompt}]
    for i in range(max_retries):
        try:
            kwargs = {
                "model": model,
                "messages": messages,
                "temperature": 0
            }
            if api_base:
                kwargs["api_base"] = api_base
            if api_key:
                kwargs["api_key"] = api_key
            response = await litellm.acompletion(**kwargs)
            return response.choices[0].message.content
        except Exception as e:
            print('************* Retrying *************')
            logging.error(f"Error: {e}")
            if i < max_retries - 1:
                await asyncio.sleep(1)
            else:
                logging.error('Max retries reached for prompt: ' + prompt)
                return ""
            
            
def get_json_content(response):
    start_idx = response.find("```json")
    if start_idx != -1:
        start_idx += 7
        response = response[start_idx:]
        
    end_idx = response.rfind("```")
    if end_idx != -1:
        response = response[:end_idx]
    
    json_content = response.strip()
    return json_content
         

def extract_json(content):
    try:
        # First, try to extract JSON enclosed within ```json and ```
        start_idx = content.find("```json")
        if start_idx != -1:
            start_idx += 7  # Adjust index to start after the delimiter
            end_idx = content.rfind("```")
            json_content = content[start_idx:end_idx].strip()
        else:
            # If no delimiters, assume entire content could be JSON
            json_content = content.strip()

        # Clean up common issues that might cause parsing errors
        json_content = json_content.replace('None', 'null')  # Replace Python None with JSON null
        json_content = json_content.replace('\n', ' ').replace('\r', ' ')  # Remove newlines
        json_content = ' '.join(json_content.split())  # Normalize whitespace

        # Attempt to parse and return the JSON object
        return json.loads(json_content)
    except json.JSONDecodeError as e:
        logging.error(f"Failed to extract JSON: {e}")
        # Try to clean up the content further if initial parsing fails
        try:
            # Remove any trailing commas before closing brackets/braces
            json_content = json_content.replace(',]', ']').replace(',}', '}')
            return json.loads(json_content)
        except:
            logging.error("Failed to parse JSON even after cleanup")
            return {}
    except Exception as e:
        logging.error(f"Unexpected error while extracting JSON: {e}")
        return {}

def write_node_id(data, node_id=0):
    if isinstance(data, dict):
        data['node_id'] = str(node_id).zfill(4)
        node_id += 1
        for key in list(data.keys()):
            if 'nodes' in key:
                node_id = write_node_id(data[key], node_id)
    elif isinstance(data, list):
        for index in range(len(data)):
            node_id = write_node_id(data[index], node_id)
    return node_id

def get_nodes(structure):
    if isinstance(structure, dict):
        structure_node = copy.deepcopy(structure)
        structure_node.pop('nodes', None)
        nodes = [structure_node]
        for key in list(structure.keys()):
            if 'nodes' in key:
                nodes.extend(get_nodes(structure[key]))
        return nodes
    elif isinstance(structure, list):
        nodes = []
        for item in structure:
            nodes.extend(get_nodes(item))
        return nodes
    
def structure_to_list(structure):
    if isinstance(structure, dict):
        nodes = []
        nodes.append(structure)
        if 'nodes' in structure:
            nodes.extend(structure_to_list(structure['nodes']))
        return nodes
    elif isinstance(structure, list):
        nodes = []
        for item in structure:
            nodes.extend(structure_to_list(item))
        return nodes

    
def get_leaf_nodes(structure):
    if isinstance(structure, dict):
        if not structure['nodes']:
            structure_node = copy.deepcopy(structure)
            structure_node.pop('nodes', None)
            return [structure_node]
        else:
            leaf_nodes = []
            for key in list(structure.keys()):
                if 'nodes' in key:
                    leaf_nodes.extend(get_leaf_nodes(structure[key]))
            return leaf_nodes
    elif isinstance(structure, list):
        leaf_nodes = []
        for item in structure:
            leaf_nodes.extend(get_leaf_nodes(item))
        return leaf_nodes

def is_leaf_node(data, node_id):
    # Helper function to find the node by its node_id
    def find_node(data, node_id):
        if isinstance(data, dict):
            if data.get('node_id') == node_id:
                return data
            for key in data.keys():
                if 'nodes' in key:
                    result = find_node(data[key], node_id)
                    if result:
                        return result
        elif isinstance(data, list):
            for item in data:
                result = find_node(item, node_id)
                if result:
                    return result
        return None

    # Find the node with the given node_id
    node = find_node(data, node_id)

    # Check if the node is a leaf node
    if node and not node.get('nodes'):
        return True
    return False

def get_last_node(structure):
    return structure[-1]


def extract_text_from_pdf(pdf_path):
    pdf_reader = PyPDF2.PdfReader(pdf_path)
    ###return text not list 
    text=""
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        text+=page.extract_text()
    return text

def get_pdf_title(pdf_path):
    pdf_reader = PyPDF2.PdfReader(pdf_path)
    meta = pdf_reader.metadata
    title = meta.title if meta and meta.title else 'Untitled'
    return title

def get_text_of_pages(pdf_path, start_page, end_page, tag=True):
    pdf_reader = PyPDF2.PdfReader(pdf_path)
    text = ""
    for page_num in range(start_page-1, end_page):
        page = pdf_reader.pages[page_num]
        page_text = page.extract_text()
        if tag:
            text += f"<start_index_{page_num+1}>\n{page_text}\n<end_index_{page_num+1}>\n"
        else:
            text += page_text
    return text

def get_first_start_page_from_text(text):
    start_page = -1
    start_page_match = re.search(r'<start_index_(\d+)>', text)
    if start_page_match:
        start_page = int(start_page_match.group(1))
    return start_page

def get_last_start_page_from_text(text):
    start_page = -1
    # Find all matches of start_index tags
    start_page_matches = re.finditer(r'<start_index_(\d+)>', text)
    # Convert iterator to list and get the last match if any exist
    matches_list = list(start_page_matches)
    if matches_list:
        start_page = int(matches_list[-1].group(1))
    return start_page


def sanitize_filename(filename, replacement='-'):
    # In Linux, only '/' and '\0' (null) are invalid in filenames.
    # Null can't be represented in strings, so we only handle '/'.
    return filename.replace('/', replacement)

def get_pdf_name(pdf_path):
    # Extract PDF name
    if isinstance(pdf_path, str):
        pdf_name = os.path.basename(pdf_path)
    elif isinstance(pdf_path, BytesIO):
        pdf_reader = PyPDF2.PdfReader(pdf_path)
        meta = pdf_reader.metadata
        pdf_name = meta.title if meta and meta.title else 'Untitled'
        pdf_name = sanitize_filename(pdf_name)
    return pdf_name


class JsonLogger:
    def __init__(self, file_path):
        # Extract PDF name for logger name
        pdf_name = get_pdf_name(file_path)
            
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filename = f"{pdf_name}_{current_time}.json"
        os.makedirs("./logs", exist_ok=True)
        # Initialize empty list to store all messages
        self.log_data = []

    def log(self, level, message, **kwargs):
        if isinstance(message, dict):
            self.log_data.append(message)
        else:
            self.log_data.append({'message': message})
        # Add new message to the log data
        
        # Write entire log data to file
        with open(self._filepath(), "w") as f:
            json.dump(self.log_data, f, indent=2)

    def info(self, message, **kwargs):
        self.log("INFO", message, **kwargs)

    def error(self, message, **kwargs):
        self.log("ERROR", message, **kwargs)

    def debug(self, message, **kwargs):
        self.log("DEBUG", message, **kwargs)

    def exception(self, message, **kwargs):
        kwargs["exception"] = True
        self.log("ERROR", message, **kwargs)

    def _filepath(self):
        return os.path.join("logs", self.filename)
    



def list_to_tree(data):
    def get_parent_structure(structure):
        """Helper function to get the parent structure code"""
        if not structure:
            return None
        parts = str(structure).split('.')
        return '.'.join(parts[:-1]) if len(parts) > 1 else None
    
    # First pass: Create nodes and track parent-child relationships
    nodes = {}
    root_nodes = []
    
    for item in data:
        structure = item.get('structure')
        node = {
            'title': item.get('title'),
            'start_index': item.get('start_index'),
            'end_index': item.get('end_index'),
            'nodes': []
        }
        
        nodes[structure] = node
        
        # Find parent
        parent_structure = get_parent_structure(structure)
        
        if parent_structure:
            # Add as child to parent if parent exists
            if parent_structure in nodes:
                nodes[parent_structure]['nodes'].append(node)
            else:
                root_nodes.append(node)
        else:
            # No parent, this is a root node
            root_nodes.append(node)
    
    # Helper function to clean empty children arrays
    def clean_node(node):
        if not node['nodes']:
            del node['nodes']
        else:
            for child in node['nodes']:
                clean_node(child)
        return node
    
    # Clean and return the tree
    return [clean_node(node) for node in root_nodes]

def add_preface_if_needed(data):
    if not isinstance(data, list) or not data:
        return data

    if data[0]['physical_index'] is not None and data[0]['physical_index'] > 1:
        preface_node = {
            "structure": "0",
            "title": "Preface",
            "physical_index": 1,
        }
        data.insert(0, preface_node)
    return data



def get_page_tokens(pdf_path, model=None, pdf_parser="PyPDF2"):
    if pdf_parser == "PyPDF2":
        pdf_reader = PyPDF2.PdfReader(pdf_path)
        page_list = []
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            page_text = page.extract_text()
            token_length = litellm.token_counter(model=model, text=page_text)
            page_list.append((page_text, token_length))
        return page_list
    elif pdf_parser == "PyMuPDF":
        if isinstance(pdf_path, BytesIO):
            pdf_stream = pdf_path
            doc = pymupdf.open(stream=pdf_stream, filetype="pdf")
        elif isinstance(pdf_path, str) and os.path.isfile(pdf_path) and pdf_path.lower().endswith(".pdf"):
            doc = pymupdf.open(pdf_path)
        page_list = []
        for page in doc:
            page_text = page.get_text()
            token_length = litellm.token_counter(model=model, text=page_text)
            page_list.append((page_text, token_length))
        return page_list
    else:
        raise ValueError(f"Unsupported PDF parser: {pdf_parser}")

        

def get_text_of_pdf_pages(pdf_pages, start_page, end_page):
    text = ""
    for page_num in range(start_page-1, end_page):
        text += pdf_pages[page_num][0]
    return text

def get_text_of_pdf_pages_with_labels(pdf_pages, start_page, end_page):
    text = ""
    for page_num in range(start_page-1, end_page):
        text += f"<physical_index_{page_num+1}>\n{pdf_pages[page_num][0]}\n<physical_index_{page_num+1}>\n"
    return text

def get_number_of_pages(pdf_path):
    pdf_reader = PyPDF2.PdfReader(pdf_path)
    num = len(pdf_reader.pages)
    return num



def post_processing(structure, end_physical_index):
    # First convert page_number to start_index in flat list
    for i, item in enumerate(structure):
        item['start_index'] = item.get('physical_index')
        if i < len(structure) - 1:
            if structure[i + 1].get('appear_start') == 'yes':
                item['end_index'] = structure[i + 1]['physical_index']-1
            else:
                item['end_index'] = structure[i + 1]['physical_index']
        else:
            item['end_index'] = end_physical_index
    tree = list_to_tree(structure)
    if len(tree)!=0:
        return tree
    else:
        ### remove appear_start 
        for node in structure:
            node.pop('appear_start', None)
            node.pop('physical_index', None)
        return structure

def clean_structure_post(data):
    if isinstance(data, dict):
        data.pop('page_number', None)
        data.pop('start_index', None)
        data.pop('end_index', None)
        if 'nodes' in data:
            clean_structure_post(data['nodes'])
    elif isinstance(data, list):
        for section in data:
            clean_structure_post(section)
    return data

def remove_fields(data, fields=['text']):
    if isinstance(data, dict):
        return {k: remove_fields(v, fields)
            for k, v in data.items() if k not in fields}
    elif isinstance(data, list):
        return [remove_fields(item, fields) for item in data]
    return data

def print_toc(tree, indent=0):
    for node in tree:
        print('  ' * indent + node['title'])
        if node.get('nodes'):
            print_toc(node['nodes'], indent + 1)

def print_json(data, max_len=40, indent=2):
    def simplify_data(obj):
        if isinstance(obj, dict):
            return {k: simplify_data(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [simplify_data(item) for item in obj]
        elif isinstance(obj, str) and len(obj) > max_len:
            return obj[:max_len] + '...'
        else:
            return obj
    
    simplified = simplify_data(data)
    print(json.dumps(simplified, indent=indent, ensure_ascii=False))


def remove_structure_text(data):
    if isinstance(data, dict):
        data.pop('text', None)
        if 'nodes' in data:
            remove_structure_text(data['nodes'])
    elif isinstance(data, list):
        for item in data:
            remove_structure_text(item)
    return data


def check_token_limit(structure, limit=110000):
    list = structure_to_list(structure)
    for node in list:
        num_tokens = count_tokens(node['text'], model=None)
        if num_tokens > limit:
            print(f"Node ID: {node['node_id']} has {num_tokens} tokens")
            print("Start Index:", node['start_index'])
            print("End Index:", node['end_index'])
            print("Title:", node['title'])
            print("\n")


def convert_physical_index_to_int(data):
    if isinstance(data, list):
        for i in range(len(data)):
            # Check if item is a dictionary and has 'physical_index' key
            if isinstance(data[i], dict) and 'physical_index' in data[i]:
                if isinstance(data[i]['physical_index'], str):
                    if data[i]['physical_index'].startswith('<physical_index_'):
                        data[i]['physical_index'] = int(data[i]['physical_index'].split('_')[-1].rstrip('>').strip())
                    elif data[i]['physical_index'].startswith('physical_index_'):
                        data[i]['physical_index'] = int(data[i]['physical_index'].split('_')[-1].strip())
    elif isinstance(data, str):
        if data.startswith('<physical_index_'):
            data = int(data.split('_')[-1].rstrip('>').strip())
        elif data.startswith('physical_index_'):
            data = int(data.split('_')[-1].strip())
        # Check data is int
        if isinstance(data, int):
            return data
        else:
            return None
    return data


def convert_page_to_int(data):
    for item in data:
        if 'page' in item and isinstance(item['page'], str):
            try:
                item['page'] = int(item['page'])
            except ValueError:
                # Keep original value if conversion fails
                pass
    return data


def add_node_text(node, pdf_pages):
    if isinstance(node, dict):
        start_page = node.get('start_index')
        end_page = node.get('end_index')
        node['text'] = get_text_of_pdf_pages(pdf_pages, start_page, end_page)
        if 'nodes' in node:
            add_node_text(node['nodes'], pdf_pages)
    elif isinstance(node, list):
        for index in range(len(node)):
            add_node_text(node[index], pdf_pages)
    return


def add_node_text_with_labels(node, pdf_pages):
    if isinstance(node, dict):
        start_page = node.get('start_index')
        end_page = node.get('end_index')
        node['text'] = get_text_of_pdf_pages_with_labels(pdf_pages, start_page, end_page)
        if 'nodes' in node:
            add_node_text_with_labels(node['nodes'], pdf_pages)
    elif isinstance(node, list):
        for index in range(len(node)):
            add_node_text_with_labels(node[index], pdf_pages)
    return


async def generate_node_summary(node, model=None):
    prompt = f"""You are given a part of a document, your task is to generate a description of the partial document about what are main points covered in the partial document.

    Partial Document Text: {node['text']}
    
    Directly return the description, do not include any other text.
    """
    response = await llm_acompletion(model, prompt)
    return response


async def generate_summaries_for_structure(structure, model=None):
    nodes = structure_to_list(structure)
    tasks = [generate_node_summary(node, model=model) for node in nodes]
    summaries = await asyncio.gather(*tasks)
    
    for node, summary in zip(nodes, summaries):
        node['summary'] = summary
    return structure


# When MarkItDown emits page-sliced markdown, headings are often
# "Section / Page N" — titles alone cannot support a useful doc description.
_LOW_SIGNAL_TITLE_RE = re.compile(
    r"(?i)^\s*(?:#{1,6}\s*)?(?:section\s*/\s*)?page\s*\d+\s*$"
    r"|^\s*(?:#{1,6}\s*)?section\s*/\s*page\s*\d+"
)
DOC_DESC_EXCERPT_MAX_CHARS = 400


def is_low_signal_title(title: str) -> bool:
    """True when a heading carries little topical signal (e.g. Section / Page 12)."""
    t = (title or "").strip()
    if not t:
        return True
    return bool(_LOW_SIGNAL_TITLE_RE.search(t))


def text_excerpt_for_outline(text: str, max_chars: int = DOC_DESC_EXCERPT_MAX_CHARS) -> str:
    """Strip images/markup noise and return a short prose excerpt for the outline."""
    if not text:
        return ""
    cleaned = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", text)
    cleaned = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", cleaned)
    cleaned = re.sub(r"[`*_>#]+", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    # Drop a leading page-marker echo if present.
    cleaned = re.sub(
        r"(?i)^(?:section\s*/\s*)?page\s*\d+\s*",
        "",
        cleaned,
    ).strip()
    if len(cleaned) <= max_chars:
        return cleaned
    truncated = cleaned[:max_chars]
    if " " in truncated:
        truncated = truncated.rsplit(" ", 1)[0]
    return truncated.strip()


def create_clean_structure_for_description(structure):
    """
    Create a clean structure for document description generation.

    Keeps titles/summaries. For low-signal titles (page dumps), also keeps a
    short text excerpt so the LLM has real topical content — not just
    "Section / Page N" labels.
    """
    if isinstance(structure, dict):
        clean_node = {}
        for key in ['title', 'node_id', 'summary', 'prefix_summary']:
            if key in structure:
                clean_node[key] = structure[key]

        title = (clean_node.get('title') or '').strip()
        text = structure.get('text') or ''
        # Prefer explicit summaries; otherwise borrow body when the title is weak.
        if text and not (clean_node.get('summary') or clean_node.get('prefix_summary')):
            if is_low_signal_title(title) or not title:
                excerpt = text_excerpt_for_outline(text)
                if excerpt:
                    clean_node['text_excerpt'] = excerpt

        if 'nodes' in structure and structure['nodes']:
            clean_node['nodes'] = create_clean_structure_for_description(structure['nodes'])

        return clean_node
    elif isinstance(structure, list):
        return [create_clean_structure_for_description(item) for item in structure]
    else:
        return structure


# Token budgets for doc description on ~32k-context models (e.g. Qwen via LiteLLM).
DOC_DESC_SINGLE_MAX_TOKENS = 12000
DOC_DESC_BATCH_MAX_TOKENS = 8000
DOC_DESC_MAX_REDUCE_DEPTH = 8

_DOC_DESC_LOGGER = logging.getLogger(__name__)


def flatten_structure_to_outline(structure, depth=0):
    """Flatten a cleaned tree into ordered outline lines: '{indent}{node_id}: {title}'."""
    lines = []
    if isinstance(structure, list):
        for item in structure:
            lines.extend(flatten_structure_to_outline(item, depth))
        return lines
    if not isinstance(structure, dict):
        return lines

    title = (structure.get('title') or '').strip()
    excerpt = (structure.get('text_excerpt') or '').strip()
    if title or excerpt:
        node_id = (structure.get('node_id') or '').strip()
        indent = '  ' * depth
        prefix = f"{node_id}: " if node_id else ""
        label = title or "(untitled)"
        line = f"{indent}{prefix}{label}"
        extras = []
        for key in ('summary', 'prefix_summary', 'text_excerpt'):
            value = (structure.get(key) or '').strip()
            if value:
                extras.append(value)
        if extras:
            line = f"{line} — {'; '.join(extras)}"
        lines.append(line)

    children = structure.get('nodes') or []
    if children:
        lines.extend(flatten_structure_to_outline(children, depth + 1))
    return lines


def _pack_outline_batches(lines, model=None, max_tokens=DOC_DESC_BATCH_MAX_TOKENS):
    """Pack consecutive outline lines into batches under max_tokens."""
    if not lines:
        return []

    batches = []
    current = []
    current_tokens = 0

    for line in lines:
        line_tokens = count_tokens(line, model=model)
        # Single oversize line still gets its own batch so progress can continue.
        if current and current_tokens + line_tokens + 1 > max_tokens:
            batches.append(current)
            current = []
            current_tokens = 0
        current.append(line)
        current_tokens += line_tokens + (1 if len(current) > 1 else 0)

    if current:
        batches.append(current)
    return batches


def _doc_description_from_outline_text(outline_text, model=None, kind='full'):
    if kind == 'partial':
        prompt = f"""You are an expert in summarizing documents.
You are given part of a document's outline (section headings and any short text excerpts). Write one summary paragraph covering only this part.
Capture the main subjects and key entities. Ground the summary only in the provided text; do not invent unrelated domains.
Directly return the summary, do not include any other text.

Partial Document Outline:
{outline_text}
"""
    elif kind == 'reduce':
        prompt = f"""You are an expert in summarizing documents.
You are given partial summaries of different sections of the same document. Combine them into one detailed summary paragraph (1 comprehensive paragraph) of the whole document.
Provide enough detail to capture the core context, primary subjects, and key entities, which makes it easy to distinguish this document from others and understand its overall purpose.
Directly return the summary, do not include any other text.

Partial Summaries:
{outline_text}
"""
    else:
        prompt = f"""You are an expert in summarizing documents.
You are given the structure of a document (section titles, and where present short text excerpts). Your task is to generate a detailed summary paragraph (1 comprehensive paragraph) of the document.
Provide enough detail to capture the core context, primary subjects, and key entities, which makes it easy to distinguish this document from others and understand its overall purpose.
Ground the summary only in the provided outline/excerpts. Do not invent unrelated domains or systems that are not evidenced there.

Document Structure:
{outline_text}

Directly return the summary, do not include any other text.
"""
    return llm_completion(model, prompt)


def _map_reduce_outline_lines(lines, model=None, depth=0):
    """Map-reduce outline lines into one document description."""
    if not lines:
        return ""

    if depth > DOC_DESC_MAX_REDUCE_DEPTH:
        _DOC_DESC_LOGGER.error(
            "doc_description exceeded max reduce depth=%d lines=%d; aborting",
            DOC_DESC_MAX_REDUCE_DEPTH, len(lines),
        )
        return ""

    outline_text = "\n".join(lines)
    total_tokens = count_tokens(outline_text, model=model)

    if total_tokens <= DOC_DESC_SINGLE_MAX_TOKENS:
        kind = 'reduce' if depth > 0 else 'full'
        _DOC_DESC_LOGGER.info(
            "doc_description path=%s depth=%d lines=%d tokens=%d",
            kind, depth, len(lines), total_tokens,
        )
        return _doc_description_from_outline_text(outline_text, model=model, kind=kind)

    batches = _pack_outline_batches(lines, model=model, max_tokens=DOC_DESC_BATCH_MAX_TOKENS)

    # One unsplittable oversize chunk: attempt a single call rather than looping forever.
    if len(batches) <= 1:
        kind = 'reduce' if depth > 0 else 'partial'
        _DOC_DESC_LOGGER.warning(
            "doc_description unsplittable oversize payload depth=%d tokens=%d; attempting single %s call",
            depth, total_tokens, kind,
        )
        return _doc_description_from_outline_text(outline_text, model=model, kind=kind)

    _DOC_DESC_LOGGER.info(
        "doc_description path=map_reduce depth=%d lines=%d tokens=%d batches=%d",
        depth, len(lines), total_tokens, len(batches),
    )

    partials = []
    for i, batch in enumerate(batches):
        batch_text = "\n".join(batch)
        batch_tokens = count_tokens(batch_text, model=model)
        _DOC_DESC_LOGGER.info(
            "doc_description map batch=%d/%d lines=%d tokens=%d",
            i + 1, len(batches), len(batch), batch_tokens,
        )
        partial = _doc_description_from_outline_text(batch_text, model=model, kind='partial')
        if not partial:
            _DOC_DESC_LOGGER.error(
                "doc_description map batch=%d/%d returned empty (tokens=%d); aborting",
                i + 1, len(batches), batch_tokens,
            )
            return ""
        partials.append(partial.strip())

    return _map_reduce_outline_lines(partials, model=model, depth=depth + 1)


def generate_doc_description(structure, model=None):
    """
    Generate a document-level summary from a cleaned structure.

    Small outlines use a single LLM call. Large outlines use token-capped
    map-reduce so the prompt fits ~32k-context models.
    """
    lines = flatten_structure_to_outline(structure)
    if not lines:
        # Fallback: previous behavior for unexpected empty clean trees
        outline_text = str(structure)
        _DOC_DESC_LOGGER.info("doc_description path=fallback_raw empty_outline=True")
        return _doc_description_from_outline_text(outline_text, model=model, kind='full')

    return _map_reduce_outline_lines(lines, model=model, depth=0)


def reorder_dict(data, key_order):
    if not key_order:
        return data
    return {key: data[key] for key in key_order if key in data}


def format_structure(structure, order=None):
    if not order:
        return structure
    if isinstance(structure, dict):
        if 'nodes' in structure:
            structure['nodes'] = format_structure(structure['nodes'], order)
        if not structure.get('nodes'):
            structure.pop('nodes', None)
        structure = reorder_dict(structure, order)
    elif isinstance(structure, list):
        structure = [format_structure(item, order) for item in structure]
    return structure


class ConfigLoader:
    def __init__(self, default_path: str = None):
        if default_path is None:
            default_path = Path(__file__).parent / "config.yaml"
        self._default_dict = self._load_yaml(default_path)

    @staticmethod
    def _load_yaml(path):
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def _validate_keys(self, user_dict):
        unknown_keys = set(user_dict) - set(self._default_dict)
        if unknown_keys:
            raise ValueError(f"Unknown config keys: {unknown_keys}")

    def load(self, user_opt=None) -> config:
        """
        Load the configuration, merging user options with default values.
        """
        if user_opt is None:
            user_dict = {}
        elif isinstance(user_opt, config):
            user_dict = vars(user_opt)
        elif isinstance(user_opt, dict):
            user_dict = user_opt
        else:
            raise TypeError("user_opt must be dict, config(SimpleNamespace) or None")

        self._validate_keys(user_dict)
        merged = {**self._default_dict, **user_dict}
        return config(**merged)

def create_node_mapping(tree):
    """Create a flat dict mapping node_id to node for quick lookup."""
    mapping = {}
    def _traverse(nodes):
        for node in nodes:
            if node.get('node_id'):
                mapping[node['node_id']] = node
            if node.get('nodes'):
                _traverse(node['nodes'])
    _traverse(tree)
    return mapping

def print_tree(tree, indent=0):
    for node in tree:
        summary = node.get('summary') or node.get('prefix_summary', '')
        summary_str = f"  —  {summary[:60]}..." if summary else ""
        print('  ' * indent + f"[{node.get('node_id', '?')}] {node.get('title', '')}{summary_str}")
        if node.get('nodes'):
            print_tree(node['nodes'], indent + 1)

def print_wrapped(text, width=100):
    for line in text.splitlines():
        print(textwrap.fill(line, width=width))

