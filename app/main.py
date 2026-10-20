import sys

import os
import shutil
import uuid
import logging
import asyncio
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse

from pageindex import page_index_main, md_to_tree
from pageindex.utils import ConfigLoader
from pageindex.context import request_api_key, request_api_base

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="PageIndex Service", version="1.0.0")

UPLOAD_DIR = Path("/app/data/uploads")

# Ensure upload directory exists
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/index/")
async def index_document(
    file: UploadFile = File(...),
    api_key: Optional[str] = Form(None),
    model: Optional[str] = Form(None),
    base_url: Optional[str] = Form(None),
    if_add_doc_description: Optional[str] = Form(None),
    if_add_node_summary: Optional[str] = Form(None)
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in [".pdf", ".md", ".markdown", ".txt"]:
        raise HTTPException(status_code=422, detail=f"Unsupported file extension: {suffix}")

    # Store api key and base url in thread-safe context variables
    if api_key:
        request_api_key.set(api_key)
    if base_url:
        request_api_base.set(base_url)

    # Save file to disk
    file_id = str(uuid.uuid4())
    temp_file_path = UPLOAD_DIR / f"{file_id}_{file.filename}"
    
    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logger.info(f"Saved uploaded file to {temp_file_path}")

        # Load configuration for both PDF and MD
        config_path = "/app/pageindex_config.yaml" if os.path.exists("/app/pageindex_config.yaml") else None
        config_loader = ConfigLoader(config_path)
        
        user_opt = {}
        if model:
            user_opt["model"] = model
            
        opt = config_loader.load(user_opt=user_opt)
        
        # Override with explicit form fields if provided
        final_doc_desc = if_add_doc_description if if_add_doc_description else getattr(opt, 'if_add_doc_description', 'yes')
        final_node_sum = if_add_node_summary if if_add_node_summary else getattr(opt, 'if_add_node_summary', 'no')
        opt.if_add_doc_description = final_doc_desc
        opt.if_add_node_summary = final_node_sum

        tree = None
        if suffix == ".pdf":
            # Run extraction in thread to prevent blocking event loop
            logger.info(f"Starting page_index_main for {temp_file_path} with model {opt.model}")
            tree = await asyncio.to_thread(page_index_main, str(temp_file_path), opt)
            
        else:
            # Markdown/txt processing
            # We must await md_to_tree because it's async in pageindex
            logger.info(f"Starting md_to_tree for {temp_file_path} with model {opt.model}")
            tree = await md_to_tree(
                md_path=str(temp_file_path),
                if_thinning=getattr(opt, 'if_thinning', False),
                min_token_threshold=getattr(opt, 'min_node_token', 5000),
                if_add_node_summary=final_node_sum,
                summary_token_threshold=getattr(opt, 'summary_token_threshold', 200),
                model=opt.model,
                if_add_doc_description=final_doc_desc,
                if_add_node_text=getattr(opt, 'if_add_node_text', 'yes'),
                if_add_node_id=getattr(opt, 'if_add_node_id', 'yes')
            )
            
        return JSONResponse(content={"status": "success", "tree": tree})
        
    except Exception as e:
        logger.error(f"Error processing document: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )
        
    finally:
        # Cleanup file
        if temp_file_path.exists():
            os.remove(temp_file_path)
            logger.info(f"Deleted temporary file {temp_file_path}")
