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

async def process_file(file_path, args, opt):
    """
    Process a single markdown file using PageIndex md_to_tree.
    """
    print(f"Processing file: {file_path}")
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
    parser = argparse.ArgumentParser(description='Parse a directory of markdown files and upload PageIndex tree to PostgreSQL')
    
    # Ingestion arguments
    parser.add_argument('--folder', type=str, required=True, help='Path to the folder containing markdown files')
    parser.add_argument('--recursive', action='store_true', help='Recursively scan folder for markdown files')
    
    # DB connection arguments
    parser.add_argument('--db-host', type=str, default=os.getenv('DB_HOST', '10.12.5.23'), help='PostgreSQL host')
    parser.add_argument('--db-port', type=str, default=os.getenv('DB_PORT', '5432'), help='PostgreSQL port')
    parser.add_argument('--db-name', type=str, default=os.getenv('DB_NAME', 'global_knowledgebase'), help='PostgreSQL database name')
    parser.add_argument('--db-user', type=str, default=os.getenv('DB_USER', 'postgres'), help='PostgreSQL username')
    parser.add_argument('--db-password', type=str, default=os.getenv('DB_PASSWORD', 'postgres'), help='PostgreSQL password')
    parser.add_argument('--db-schema', type=str, default=os.getenv('DB_SCHEMA', 'common'), help='PostgreSQL schema name')
    parser.add_argument('--db-timeout', type=int, default=5, help='PostgreSQL connection timeout in seconds')
    parser.add_argument('--table-name', type=str, default=os.getenv('DB_TABLE_NAME', 'document_toc'), help='Target database table')

    # PageIndex options
    parser.add_argument('--model', type=str, default=os.getenv('LLM_MODEL', 'openai/deepseek-v4-pro'), help='Model to use (overrides config.yaml)')
    parser.add_argument('--if-add-node-id', type=str, default=None, help='Whether to add node id to the node (yes/no)')
    parser.add_argument('--if-add-node-summary', type=str, default=None, help='Whether to add summary to the node (yes/no)')
    parser.add_argument('--if-add-doc-description', type=str, default=None, help='Whether to add doc description to the doc (yes/no)')
    parser.add_argument('--if-add-node-text', type=str, default=None, help='Whether to add text to the node (yes/no)')
    parser.add_argument('--if-thinning', type=str, default='no', help='Whether to apply tree thinning for markdown (yes/no)')
    parser.add_argument('--thinning-threshold', type=int, default=5000, help='Minimum token threshold for thinning')
    parser.add_argument('--summary-token-threshold', type=int, default=200, help='Token threshold for generating summaries')
    
    args = parser.parse_args()

    # Load configuration
    user_opt = {
        'model': args.model,
        'if_add_node_summary': args.if_add_node_summary,
        'if_add_doc_description': args.if_add_doc_description,
        'if_add_node_text': args.if_add_node_text,
        'if_add_node_id': args.if_add_node_id
    }
    opt = ConfigLoader().load({k: v for k, v in user_opt.items() if v is not None})

    # Find markdown files
    folder_path = Path(args.folder)
    if not folder_path.exists() or not folder_path.is_dir():
        print(f"Error: Folder path '{args.folder}' does not exist or is not a directory.")
        return

    pattern = "**/*.md" if args.recursive else "*.md"
    markdown_files = list(folder_path.glob(pattern)) + list(folder_path.glob("**/*.markdown" if args.recursive else "*.markdown"))
    
    # De-duplicate files (glob might find duplicates depending on pattern)
    markdown_files = list(set(markdown_files))
    
    if not markdown_files:
        print(f"No markdown files found in '{args.folder}' (recursive={args.recursive}).")
        return

    print(f"Found {len(markdown_files)} markdown files to process.")

    # Initialize Postgres Connection
    try:
        print(f"Connecting to database '{args.db_name}' schema '{args.db_schema}' on {args.db_host}:{args.db_port}...")
        conn = get_db_connection(args)
        setup_database_table(conn, args.db_schema, args.table_name)
        print(f"Database table '{args.table_name}' verified/created in schema '{args.db_schema}'.")
    except Exception as e:
        print(f"Database connection error: {e}")
        return

    # Process and upload files
    success_count = 0
    for file_path in markdown_files:
        abs_path = file_path.resolve()
        result = await process_file(abs_path, args, opt)
        if result:
            doc_name = result['doc_name']
            structure = result['structure']
            
            # Format table of contents string
            toc_str = get_toc_string(structure)
            
            try:
                upsert_document_index(
                    conn=conn,
                    schema_name=args.db_schema,
                    table_name=args.table_name,
                    doc_name=doc_name,
                    file_path=str(abs_path),
                    toc=toc_str,
                    structure=result
                )
                print(f"Successfully upserted: {doc_name}")
                success_count += 1
            except Exception as e:
                print(f"Database upsert failed for {doc_name}: {e}")
                conn.rollback()
                
    conn.close()
    print(f"\nIngestion completed. Successfully processed and stored {success_count}/{len(markdown_files)} documents.")

if __name__ == "__main__":
    asyncio.run(main())
