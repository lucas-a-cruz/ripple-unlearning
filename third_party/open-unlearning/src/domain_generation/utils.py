"""Utility functions for logging and LLM initialization."""

import logging
import pprint
import sys
from datetime import datetime
from typing import Dict

from langchain.chat_models import init_chat_model
from loguru import logger as loguru_logger
from retrying import retry

from src.config import config

# -----------------------------------------------------------------------------
# Logging setup (Loguru + stdlib interoperability)
# -----------------------------------------------------------------------------

LOG_FORMAT = (
    "<blue>{time:YYYY-MM-DD HH:mm:ss}</blue> "
    "| <level>{level: <7}</level> "
    "| <cyan>{extra[component]: <18}</cyan> "
    "| {message}"
)

loguru_logger.remove()
loguru_logger.add(sys.stderr, format=LOG_FORMAT, colorize=True, enqueue=False)


class InterceptHandler(logging.Handler):
    """Redirect standard logging records through Loguru."""

    def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover - passthrough
        try:
            level = loguru_logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Add component field if missing (for third-party libraries)
        frame, depth = logging.currentframe(), 2
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        # Bind a default component for third-party logs
        loguru_logger.bind(component=record.name or "external").opt(
            depth=depth, exception=record.exc_info
        ).log(level, record.getMessage())


logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)

# Reduce noise from third-party libraries
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

logger = loguru_logger.bind(component="domain_generation")

pp = pprint.PrettyPrinter(indent=2)

STAGE_COLOR_HINTS: Dict[str, str] = {
    "start": "green",
    "waiting": "yellow",
    "wait": "yellow",
    "complete": "blue",
    "chapters_complete": "blue",
    "sections_complete": "blue",
    "end": "cyan",
    "truncate": "magenta",
    "error": "red",
    "warning": "yellow",
    "fail": "red",
}

STEP_COLOR_PALETTE = [
    "magenta",
    "cyan",
    "yellow",
    "blue",
    "green",
    "white",
]


def _color_for_stage(stage: str) -> str:
    stage_lower = stage.lower()
    for key, color in STAGE_COLOR_HINTS.items():
        if key in stage_lower:
            return color
    return "white"


def _color_for_step(step: str) -> str:
    index = abs(hash(step)) % len(STEP_COLOR_PALETTE)
    return STEP_COLOR_PALETTE[index]


def pretty_log(step: str, stage: str, payload, full_dump: bool = False):
    """Pretty-print a log entry for a graph step with color-coded context."""

    try:
        serialized = payload.model_dump() if hasattr(payload, "model_dump") else payload
    except Exception:
        serialized = payload

    if full_dump:
        formatted = pp.pformat(serialized)
    elif isinstance(serialized, (dict, list, tuple, set)):
        formatted = pp.pformat(serialized)
    elif serialized is None:
        formatted = ""
    else:
        formatted = str(serialized)
    stage_color = _color_for_stage(stage)
    step_color = _color_for_step(step)

    colored_stage = f"<{stage_color}>{stage}</{stage_color}>"
    colored_step = f"<{step_color}>{step}</{step_color}>"

    if formatted and formatted != "None":
        message = f"{colored_step} | {colored_stage}\n<dim>{formatted}</dim>"
    else:
        message = f"{colored_step} | {colored_stage}"

    logger.opt(colors=True).info(message)


# -----------------------------------------------------------------------------
# LLM Retry Wrapper
# -----------------------------------------------------------------------------


def _should_retry_on_exception(exception: Exception) -> bool:
    """Determine if we should retry based on the exception type.

    Args:
        exception: The exception that was raised

    Returns:
        True if we should retry, False otherwise
    """
    import openai

    # Retry on rate limit errors
    if isinstance(exception, openai.RateLimitError):
        logger.opt(colors=True).warning(
            "<yellow>[RETRY]</yellow> Rate limit error, will retry with exponential backoff"
        )
        return True

    # Retry on API connection errors
    if isinstance(exception, (openai.APIConnectionError, openai.APITimeoutError)):
        logger.opt(colors=True).warning(
            "<yellow>[RETRY]</yellow> API connection/timeout error, will retry"
        )
        return True

    # Don't retry on other errors
    return False


class RetryableLLM:
    """Wrapper that adds retry logic to LLM invoke calls."""

    def __init__(self, llm):
        """Initialize with an LLM instance.

        Args:
            llm: The underlying LLM instance to wrap
        """
        self._llm = llm

    def with_structured_output(self, schema):
        """Return a new RetryableLLM with structured output.

        Args:
            schema: Pydantic model or schema for structured output

        Returns:
            New RetryableLLM instance with structured output configured
        """
        structured_llm = self._llm.with_structured_output(schema)
        return RetryableLLM(structured_llm)

    @retry(
        retry_on_exception=_should_retry_on_exception,
        wait_exponential_multiplier=1000,  # Start at 1 second
        wait_exponential_max=60000,  # Max 60 seconds between retries
        stop_max_attempt_number=5,  # Max 5 attempts (matches config.max_retries)
    )
    def invoke(self, *args, **kwargs):
        """Invoke the LLM with retry logic.

        Retries with exponential backoff on rate limit and connection errors.
        Wait times: 1s, 2s, 4s, 8s, 16s (capped at 60s)

        Args:
            *args: Positional arguments to pass to LLM invoke
            **kwargs: Keyword arguments to pass to LLM invoke

        Returns:
            LLM response
        """
        return self._llm.invoke(*args, **kwargs)

    def __getattr__(self, name):
        """Delegate all other attributes to the underlying LLM.

        Args:
            name: Attribute name

        Returns:
            Attribute value from underlying LLM
        """
        return getattr(self._llm, name)


# -----------------------------------------------------------------------------
# LLM Initialization
# -----------------------------------------------------------------------------


def get_llm():
    """Get configured LLM instance with fallback support and retry logic.

    Returns:
        RetryableLLM instance wrapping the configured ChatOpenAI instance.
    """
    model_candidates = [config.model_name]
    if config.fallback_model_name:
        if config.fallback_model_name not in model_candidates:
            model_candidates.append(config.fallback_model_name)

    errors: list[str] = []
    for model_name in model_candidates:
        try:
            logger.opt(colors=True).info(
                "<green>[LLM]</green> Initializing model '<yellow>{}</yellow>' "
                "(retries={}, max_tokens={})",
                model_name,
                config.max_retries,
                config.max_tokens,
            )
            base_llm = init_chat_model(
                model_name,
                temperature=config.temperature,
                max_retries=config.max_retries,
                max_tokens=config.max_tokens,
            )
            # Wrap with retry logic for rate limit handling
            return RetryableLLM(base_llm)
        except Exception as exc:  # pragma: no cover - defensive
            logger.opt(colors=True).warning(
                "<yellow>[LLM]</yellow> Failed to init '<red>{}</red>': {}",
                model_name,
                exc,
            )
            errors.append(f"{model_name}: {exc}")

    error_message = " | ".join(errors) if errors else "Unknown error"
    raise RuntimeError(
        f"Unable to initialize any configured chat model. Attempts: {error_message}"
    )


def get_current_date() -> str:
    """Get current date formatted for prompts.

    Returns:
        Current date in YYYY-MM-DD format
    """
    return datetime.now().strftime("%Y-%m-%d")
