import logging
import os
from functools import lru_cache
from logging import Logger


def _get_log_level() -> int:
    return getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO)


@lru_cache(maxsize=1)
def _configure_logging_once() -> None:
    logging.basicConfig(
        level=_get_log_level(),
        format="%(levelname)s:\t%(asctime)s %(pathname)s:%(lineno)d: %(message)s",
    )


def get_logger(name: str) -> Logger:
    _configure_logging_once()
    return logging.getLogger(name)


# どこからでも import して使える共通logger
logger: Logger = get_logger("app")

__all__ = ["get_logger", "logger"]
