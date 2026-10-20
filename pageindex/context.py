from contextvars import ContextVar
from typing import Optional

request_api_key: ContextVar[Optional[str]] = ContextVar("request_api_key", default=None)
request_api_base: ContextVar[Optional[str]] = ContextVar("request_api_base", default=None)
