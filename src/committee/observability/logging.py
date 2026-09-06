"""structlog configuration.

Every log line emitted anywhere in orchestration carries `run_id` as a
correlation id, bound once per debate via `get_run_logger` rather than passed
explicitly through every function call.
"""

from __future__ import annotations

import logging

import structlog


def configure_logging(log_level: str = "INFO", log_format: str = "json") -> None:
    logging.basicConfig(level=log_level.upper(), format="%(message)s")

    renderer = (
        structlog.processors.JSONRenderer()
        if log_format == "json"
        else structlog.dev.ConsoleRenderer()
    )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelName(log_level.upper())
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_run_logger(run_id: str) -> structlog.BoundLogger:
    return structlog.get_logger().bind(run_id=run_id)
