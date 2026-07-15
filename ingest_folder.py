import argparse
import os
import json
import asyncio
import psycopg2
from psycopg2.extras import Json
from pathlib import Path
from dotenv import load_dotenv

# Import pageindex logic
from pageindex.page_index_md import md_to_tree
from pageindex.utils import ConfigLoader

def get_toc_string(tree, indent=0):
    """
    Format a tree structure into an indented Table of Contents text string.
    """
    lines = []
    def recurse(nodes, level):
        for node in nodes:
            lines.append('  ' * level + node['title'])
            if node.get('nodes'):
                recurse(node['nodes'], level + 1)
    recurse(tree, indent)
    return '\n'.join(lines)

def get_db_connection(args):
    """
    Establish a connection to the PostgreSQL database.
    """
    return psycopg2.connect(
        host=args.db_host,
        port=args.db_port,
        user=args.db_user,
        password=args.db_password,
        database=args.db_name,
        connect_timeout=args.db_timeout
    )

def setup_database_table(conn, schema_name, table_name):
    """
    Creates the table for storing PageIndex structure if it does not exist.
    """
    with conn.cursor() as cur:
        # Create schema if not exists and set search path
        cur.execute(f'CREATE SCHEMA IF NOT EXISTS "{schema_name}";')
        cur.execute(f'SET search_path TO "{schema_name}", public;')
        
        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            id SERIAL PRIMARY KEY,
            doc_name VARCHAR(255) NOT NULL,
            file_path TEXT UNIQUE NOT NULL,
            toc TEXT,
            json JSONB NOT NULL,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );
        """
        cur.execute(create_table_sql)
    conn.commit()

def upsert_document_index(conn, schema_name, table_name, doc_name, file_path, toc, structure):
    """
    Upserts document indexing results into the PostgreSQL database.
    """
    upsert_sql = f"""
    INSERT INTO {table_name} (doc_name, file_path, toc, json, updated_at)
    VALUES (%s, %s, %s, %s, CURRENT_TIMESTAMP)
    ON CONFLICT (file_path)
    DO UPDATE SET
        doc_name = EXCLUDED.doc_name,
        toc = EXCLUDED.toc,
        json = EXCLUDED.json,
        updated_at = CURRENT_TIMESTAMP;
    """
    with conn.cursor() as cur:
        cur.execute(f'SET search_path TO "{schema_name}", public;')
        cur.execute(upsert_sql, (doc_name, file_path, toc, Json(structure)))
    conn.commit()

async def convert_file_via_api(file_path: Path, args) -> str:
    """
    Uploads a file to the MarkItDown API for conversion to Markdown.
    """
    import httpx
    print(f"Uploading '{file_path.name}' to conversion service at {args.markitdown_url}...")
    
    timeout = httpx.Timeout(300.0, connect=10.0)
    
    with open(file_path, "rb") as f:
        files = {"file": (file_path.name, f, "application/octet-stream")}
        data = {
            "enable_llm": "true" if args.enable_llm else "false",
        }
        if args.enable_llm:
            if args.llm_model:
                data["llm_model"] = args.llm_model
            if args.llm_api_base:
                data["llm_api_base"] = args.llm_api_base
                
        headers = {}
        if args.api_key:
            headers["Authorization"] = f"Bearer {args.api_key}"
            
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(args.markitdown_url, files=files, data=data, headers=headers)
            
    response.raise_for_status()
    
    if response.headers.get("X-Success") == "false":
        raise ValueError(f"MarkItDown conversion error: {response.text[:500]}")
        
    return response.text

def convert_file_locally(file_path: Path, args) -> str:
    """
    Converts a file to Markdown locally using the markitdown Python package.
    """
    try:
        from markitdown import MarkItDown
        from openai import OpenAI
    except ImportError:
        raise ImportError("Local conversion requires 'pip install markitdown openai'. Please install them or run with remote API conversion.")

    kwargs = {}
    if args.enable_llm and args.llm_model and args.llm_api_base:
        client = OpenAI(
            base_url=args.llm_api_base,
            api_key=args.api_key or "none"
        )
        kwargs["llm_client"] = client
        kwargs["llm_model"] = args.llm_model

    md = MarkItDown(**kwargs)
    result = md.convert(str(file_path))
    return result.text_content

async def process_file(file_path, args, opt):
    """
    Process a single markdown file using PageIndex md_to_tree.
    """
    print(f"Running PageIndex on: {file_path}")
    try:
        result = await md_to_tree(
            md_path=str(file_path),
            if_thinning=args.if_thinning.lower() == 'yes',
            min_token_threshold=args.thinning_threshold,
            if_add_node_summary=opt.if_add_node_summary,
            summary_token_threshold=args.summary_token_threshold,
            model=opt.model,
            if_add_doc_description=opt.if_add_doc_description,
            if_add_node_text=opt.if_add_node_text,
            if_add_node_id=opt.if_add_node_id
        )
        return result
    except Exception as e:
        print(f"Error parsing file {file_path}: {e}")
        return None

async def main():
    # Load dotenv from workspace
    load_dotenv(override=True)

    # Argument parsing
    parser = argparse.ArgumentParser(description='Parse a directory of raw documents, convert to Markdown, and upload PageIndex tree to PostgreSQL')
    
    # Ingestion folders
    parser.add_argument('--folder', type=str, required=True, help='Path to the folder containing source documents')
    parser.add_argument('--recursive', action='store_true', help='Recursively scan folder for files')
    parser.add_argument('--md-output-dir', type=str, default='./converted_md', help='Output directory for converted markdown files')
    parser.add_argument('--results-dir', type=str, default='./results', help='Output directory for JSON structures')
    
    # DB connection arguments
    parser.add_argument('--db-host', type=str, default=os.getenv('DB_HOST', '10.12.5.23'), help='PostgreSQL host')
    parser.add_argument('--db-port', type=str, default=os.getenv('DB_PORT', '5432'), help='PostgreSQL port')
    parser.add_argument('--db-name', type=str, default=os.getenv('DB_NAME', 'global_knowledgebase'), help='PostgreSQL database name')
    parser.add_argument('--db-user', type=str, default=os.getenv('DB_USER', 'postgres'), help='PostgreSQL username')
    parser.add_argument('--db-password', type=str, default=os.getenv('DB_PASSWORD', 'postgres'), help='PostgreSQL password')
    parser.add_argument('--db-schema', type=str, default=os.getenv('DB_SCHEMA', 'common'), help='PostgreSQL schema name')
    parser.add_argument('--db-timeout', type=int, default=5, help='PostgreSQL connection timeout in seconds')
    parser.add_argument('--table-name', type=str, default=os.getenv('DB_TABLE_NAME', 'document_toc'), help='Target database table')

    # Conversion options
    parser.add_argument('--local', action='store_true', help='Perform conversion locally using Python package instead of HTTP API')
    parser.add_argument('--markitdown-url', type=str, default=os.getenv('MARKITDOWN_URL', 'http://10.12.5.23:8112/api/upload-and-process'), help='MarkItDown API URL')
    parser.add_argument('--enable-llm', type=str, default='yes', help='Enable LLM vision / OCR description during conversion (yes/no)')
    parser.add_argument('--llm-model', type=str, default=os.getenv('MARKITDOWN_MODEL', 'qwen3-vl-7b'), help='VLM Model for MarkItDown')
    parser.add_argument('--llm-api-base', type=str, default=os.getenv('LLM_BASE_URL', 'http://10.12.5.23:4000/v1'), help='LiteLLM URL')
    parser.add_argument('--api-key', type=str, default=os.getenv('LITELLM_MASTER_KEY', 'sk-horizonai-litellm-2026'), help='API key for LLM')

    # PageIndex options
    parser.add_argument('--model', type=str, default=os.getenv('LLM_MODEL', 'openai/deepseek-v4-pro'), help='Model to use for PageIndex (overrides config.yaml)')
    parser.add_argument('--if-add-node-id', type=str, default=None, help='Whether to add node id to the node (yes/no)')
    parser.add_argument('--if-add-node-summary', type=str, default=None, help='Whether to add summary to the node (yes/no)')
    parser.add_argument('--if-add-doc-description', type=str, default=None, help='Whether to add doc description to the doc (yes/no)')
    parser.add_argument('--if-add-node-text', type=str, default=None, help='Whether to add text to the node (yes/no)')
    parser.add_argument('--if-thinning', type=str, default='no', help='Whether to apply tree thinning for markdown (yes/no)')
    parser.add_argument('--thinning-threshold', type=int, default=5000, help='Minimum token threshold for thinning')
    parser.add_argument('--summary-token-threshold', type=int, default=200, help='Token threshold for generating summaries')
    
    args = parser.parse_args()
    args.enable_llm = args.enable_llm.lower() == 'yes'

    # Load configuration
    user_opt = {
        'model': args.model,
        'if_add_node_summary': args.if_add_node_summary,
        'if_add_doc_description': args.if_add_doc_description,
        'if_add_node_text': args.if_add_node_text,
        'if_add_node_id': args.if_add_node_id
    }
    opt = ConfigLoader().load({k: v for k, v in user_opt.items() if v is not None})

    # Find source files
    folder_path = Path(args.folder)
    if not folder_path.exists() or not folder_path.is_dir():
        print(f"Error: Folder path '{args.folder}' does not exist or is not a directory.")
        return

    # Supported formats
    CONVERTIBLE_EXTENSIONS = {".pdf", ".docx", ".doc", ".xlsx", ".xls", ".pptx", ".ppt", ".msg", ".eml", ".txt"}
    MARKDOWN_EXTENSIONS = {".md", ".markdown"}
    
    source_files = []
    pattern = "**/*" if args.recursive else "*"
    for p in folder_path.glob(pattern):
        if p.is_file() and p.suffix.lower() in (CONVERTIBLE_EXTENSIONS | MARKDOWN_EXTENSIONS):
            source_files.append(p)
            
    source_files = list(set(source_files))
    if not source_files:
        print(f"No supported files found in '{args.folder}' (recursive={args.recursive}).")
        return

    print(f"Found {len(source_files)} files to process.")

    # Initialize Postgres Connection
    try:
        print(f"Connecting to database '{args.db_name}' schema '{args.db_schema}' on {args.db_host}:{args.db_port}...")
        conn = get_db_connection(args)
        setup_database_table(conn, args.db_schema, args.table_name)
        print(f"Database table '{args.table_name}' verified/created in schema '{args.db_schema}'.")
    except Exception as e:
        print(f"Database connection error: {e}")
        return

    # Create directories
    md_out_path = Path(args.md_output_dir)
    md_out_path.mkdir(parents=True, exist_ok=True)
    
    # Corrected name resolution
    results_dir_name = getattr(args, 'results_dir', './results')
    results_path = Path(results_dir_name)
    results_path.mkdir(parents=True, exist_ok=True)

    success_count = 0
    for file_path in source_files:
        original_abs_path = file_path.resolve()
        suffix = file_path.suffix.lower()
        
        # 1. Convert to Markdown if needed
        markdown_file_path = None
        if suffix in MARKDOWN_EXTENSIONS:
            markdown_file_path = original_abs_path
            print(f"File '{file_path.name}' is already Markdown. Processing directly...")
        else:
            print(f"File '{file_path.name}' needs conversion to Markdown...")
            converted_md_path = md_out_path / f"{file_path.stem}.md"
            try:
                if args.local:
                    md_text = convert_file_locally(original_abs_path, args)
                else:
                    md_text = await convert_file_via_api(original_abs_path, args)
                    
                with open(converted_md_path, "w", encoding="utf-8") as md_file:
                    md_file.write(md_text)
                    
                markdown_file_path = converted_md_path.resolve()
                print(f"Successfully converted '{file_path.name}' to '{markdown_file_path}'")
            except Exception as e:
                print(f"Failed to convert '{file_path.name}': {e}")
                continue

        # 2. Run PageIndex on the Markdown file
        if markdown_file_path:
            result = await process_file(markdown_file_path, args, opt)
            if result:
                doc_name = result.get('doc_name', file_path.stem)
                structure = result.get('structure', [])
                
                # Format table of contents string
                toc_str = get_toc_string(structure)
                
                # Save JSON structure
                output_file = results_path / f"{file_path.stem}_structure.json"
                try:
                    with open(output_file, 'w', encoding='utf-8') as sf:
                        json.dump(result, sf, indent=2, ensure_ascii=False)
                    print(f"Tree structure saved to: {output_file}")
                except Exception as e:
                    print(f"Failed to save JSON structure for {doc_name}: {e}")

                # Upsert into PostgreSQL DB (keyed by original file path)
                try:
                    upsert_document_index(
                        conn=conn,
                        schema_name=args.db_schema,
                        table_name=args.table_name,
                        doc_name=doc_name,
                        file_path=str(original_abs_path),
                        toc=toc_str,
                        structure=result
                    )
                    print(f"Successfully upserted database index for: {doc_name}")
                    success_count += 1
                except Exception as e:
                    print(f"Database upsert failed for {doc_name}: {e}")
                    conn.rollback()
                    
    conn.close()
    print(f"\nIngestion completed. Successfully processed and stored {success_count}/{len(source_files)} documents.")

if __name__ == "__main__":
    asyncio.run(main())
