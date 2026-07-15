import sys
import os
import json
import asyncio
from pathlib import Path
from dotenv import load_dotenv
from fastmcp import FastMCP

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables
load_dotenv(dotenv_path=project_root / ".env", override=True)

from pageindex.page_index_md import md_to_tree
from pageindex.page_index import page_index_main
from pageindex.utils import ConfigLoader

mcp = FastMCP(
    name="pageindex",
    instructions="Tools for document structure chunking, table-of-contents extraction, and PostgreSQL database ingestion."
)

class ArgsNamespace:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

@mcp.tool()
async def index_markdown_document(
    file_path: str,
    model: str = "openai/deepseek-v4-pro",
    if_thinning: bool = False,
    thinning_threshold: int = 5000,
    summary_token_threshold: int = 200,
    add_node_summary: bool = True,
    add_doc_description: bool = True,
    add_node_text: bool = True,
    add_node_id: bool = True
) -> dict:
    """
    Parse a Markdown document into a structured hierarchical Table of Contents (TOC) JSON tree.

    Args:
        file_path: Absolute path to the markdown file on disk.
        model: LLM model to route queries through.
        if_thinning: Apply token thinning/pruning for large files.
        thinning_threshold: Minimum tokens threshold to trigger thinning.
        summary_token_threshold: Minimum tokens threshold to trigger section summaries.
        add_node_summary: Generate a brief summary of each heading section.
        add_doc_description: Generate a high-level summary of the entire document.
        add_node_text: Include the raw markdown text inside each node structure.
        add_node_id: Generate unique UUIDs for each document block node.

    Returns:
        A dictionary containing the parsed PageIndex tree.
    """
    path = Path(file_path).resolve()
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Markdown file not found: {file_path}")

    # Use ConfigLoader to parse options
    config_loader = ConfigLoader()
    user_opt = {
        'model': model,
        'if_add_node_summary': 'yes' if add_node_summary else 'no',
        'if_add_doc_description': 'yes' if add_doc_description else 'no',
        'if_add_node_text': 'yes' if add_node_text else 'no',
        'if_add_node_id': 'yes' if add_node_id else 'no'
    }
    opt = config_loader.load(user_opt)

    toc_tree = await md_to_tree(
        md_path=str(path),
        if_thinning=if_thinning,
        min_token_threshold=thinning_threshold,
        if_add_node_summary=opt.if_add_node_summary,
        summary_token_threshold=summary_token_threshold,
        model=opt.model,
        if_add_doc_description=opt.if_add_doc_description,
        if_add_node_text=opt.if_add_node_text,
        if_add_node_id=opt.if_add_node_id
    )

    return toc_tree

@mcp.tool()
def index_pdf_document(
    file_path: str,
    model: str = "openai/deepseek-v4-pro",
    toc_check_pages: int = 10,
    max_pages_per_node: int = 5,
    max_tokens_per_node: int = 4000,
    add_node_summary: bool = True,
    add_doc_description: bool = True,
    add_node_text: bool = True,
    add_node_id: bool = True
) -> dict:
    """
    Parse a PDF document into a structured hierarchical Table of Contents (TOC) JSON tree.

    Args:
        file_path: Absolute path to the PDF file on disk.
        model: LLM model to route queries through.
        toc_check_pages: Number of initial pages to inspect for the printed TOC index.
        max_pages_per_node: Maximum number of pages group into a single block node.
        max_tokens_per_node: Maximum token threshold per block node.
        add_node_summary: Generate a brief summary of each chunk node.
        add_doc_description: Generate a high-level summary of the entire document.
        add_node_text: Include the raw extracted text inside each node structure.
        add_node_id: Generate unique UUIDs for each document block node.

    Returns:
        A dictionary containing the parsed PageIndex tree.
    """
    path = Path(file_path).resolve()
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"PDF file not found: {file_path}")

    # Use ConfigLoader to parse options
    config_loader = ConfigLoader()
    user_opt = {
        'model': model,
        'toc_check_page_num': toc_check_pages,
        'max_page_num_each_node': max_pages_per_node,
        'max_token_num_each_node': max_tokens_per_node,
        'if_add_node_summary': 'yes' if add_node_summary else 'no',
        'if_add_doc_description': 'yes' if add_doc_description else 'no',
        'if_add_node_text': 'yes' if add_node_text else 'no',
        'if_add_node_id': 'yes' if add_node_id else 'no'
    }
    opt = config_loader.load(user_opt)

    # Call main indexing routine from pageindex library
    toc_tree = page_index_main(str(path), opt)
    return toc_tree

@mcp.tool()
async def ingest_markdown_directory(
    folder_path: str,
    recursive: bool = True,
    model: str = "openai/deepseek-v4-pro",
    db_host: str = "10.12.5.23",
    db_port: str = "5432",
    db_name: str = "global_knowledgebase",
    db_schema: str = "common",
    table_name: str = "document_toc"
) -> dict:
    """
    Recursively parse a directory of markdown files and upload their structural trees directly to a PostgreSQL database table.

    Args:
        folder_path: Path to the target folder containing markdown files.
        recursive: Recursively search through subfolders.
        model: LLM model to route queries through.
        db_host: Host IP/domain of the PostgreSQL server.
        db_port: Connection port of the PostgreSQL server.
        db_name: Name of the target database.
        db_schema: Target schema name in PostgreSQL.
        table_name: Target table name inside the schema.

    Returns:
        A success report detailing folder path, total files found, processed successfully, and failed files list.
    """
    from ingest_md_folder import get_db_connection, setup_database_table, process_file, upsert_document_index, get_toc_string

    folder = Path(folder_path).resolve()
    if not folder.exists() or not folder.is_dir():
        raise NotADirectoryError(f"Target folder not found: {folder_path}")

    # Build DB/Ingest args mock namespaces
    args = ArgsNamespace(
        db_host=db_host,
        db_port=db_port,
        db_name=db_name,
        db_user=os.getenv("DB_USER", "postgres"),
        db_password=os.getenv("DB_PASSWORD", "postgres"),
        db_timeout=5,
        if_thinning="no",
        thinning_threshold=5000,
        summary_token_threshold=200
    )

    opt = ConfigLoader().load({
        'model': model,
        'if_add_node_summary': 'yes',
        'if_add_doc_description': 'yes',
        'if_add_node_text': 'yes',
        'if_add_node_id': 'yes'
    })

    pattern = "**/*.md" if recursive else "*.md"
    markdown_files = list(folder.glob(pattern)) + list(folder.glob("**/*.markdown" if recursive else "*.markdown"))
    markdown_files = list(set(markdown_files))

    if not markdown_files:
        return {
            "status": "warning",
            "message": f"No markdown files found in {folder_path}",
            "processed": 0,
            "failed": []
        }

    # Initialize connection
    try:
        conn = get_db_connection(args)
        setup_database_table(conn, db_schema, table_name)
    except Exception as e:
        return {
            "status": "error",
            "message": f"Database initialization failed: {str(e)}",
            "processed": 0,
            "failed": [str(f) for f in markdown_files]
        }

    success_count = 0
    failed_files = []

    for file_path in markdown_files:
        abs_path = file_path.resolve()
        try:
            result = await process_file(abs_path, args, opt)
            if result:
                doc_name = abs_path.stem
                toc_str = get_toc_string(result)
                upsert_document_index(
                    conn=conn,
                    schema_name=db_schema,
                    table_name=table_name,
                    doc_name=doc_name,
                    file_path=str(abs_path),
                    toc=toc_str,
                    structure=result
                )
                success_count += 1
            else:
                failed_files.append(str(abs_path))
        except Exception as e:
            failed_files.append(f"{abs_path}: {str(e)}")

    conn.close()

    return {
        "status": "success" if not failed_files else "partial_success",
        "folder": str(folder),
        "total_files": len(markdown_files),
        "processed_successfully": success_count,
        "failed_files": failed_files
    }

if __name__ == "__main__":
    mcp.run(transport="stdio")
