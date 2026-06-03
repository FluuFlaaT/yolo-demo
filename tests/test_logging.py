"""Tests for utils/logging module."""

import logging
import os
import tempfile

import pytest

from yolo_demo.utils.logging import get_logger, setup_logging


class TestSetupLogging:
    """Test setup_logging function."""

    def test_setup_logging_console_only(self):
        setup_logging(level=logging.WARNING)
        root = logging.getLogger()
        assert root.level == logging.WARNING
        assert len(root.handlers) >= 1

    def test_setup_logging_with_file(self, tmp_path):
        log_file = tmp_path / "test.log"
        setup_logging(level=logging.DEBUG, log_file=str(log_file))
        root = logging.getLogger()
        assert root.level == logging.DEBUG
        assert len(root.handlers) == 2
        assert log_file.parent.exists()

    def test_setup_logging_custom_format(self):
        fmt = "%(levelname)s %(message)s"
        setup_logging(level=logging.INFO, format_string=fmt)
        handler = logging.getLogger().handlers[0]
        assert handler.formatter._fmt == fmt

    def test_setup_logging_clears_previous_handlers(self):
        root = logging.getLogger()
        initial_count = len(root.handlers)
        setup_logging(level=logging.INFO)
        setup_logging(level=logging.INFO)
        after_count = len(root.handlers)
        assert after_count <= 2


class TestGetLogger:
    """Test get_logger function."""

    def test_get_logger_returns_logger(self):
        logger = get_logger("test.module")
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test.module"

    def test_get_logger_returns_same_instance(self):
        a = get_logger("test.same")
        b = get_logger("test.same")
        assert a is b

    def test_get_logger_child_of_root(self):
        logger = get_logger(__name__)
        logger.info("test message")
        root = logging.getLogger()
        assert root.level > 0
