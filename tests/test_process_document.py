"""Tests for document processing functionality."""

import pytest


class TestDocumentProcessor:
    """Test cases for document processing."""

    def test_process_document_module_exists(self):
        """Test that process_document module exists."""
        from llmkglitrev import process_document

        assert process_document is not None

    def test_docling_import(self):
        """Test that docling can be imported."""
        try:
            from docling.document_converter import DocumentConverter

            converter = DocumentConverter()
            assert converter is not None
        except ImportError:
            pytest.fail("docling not properly installed")

    def test_basic_file_operations(self, temp_dir):
        """Test basic file operations."""
        test_file = temp_dir / "test.txt"
        test_file.write_text("Hello, World!")

        assert test_file.exists()
        assert test_file.read_text() == "Hello, World!"
