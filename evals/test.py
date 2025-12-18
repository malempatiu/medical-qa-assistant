"""
Test Module for Medical QA Assistant Evaluations

This module provides functionality to load and manage test questions for evaluating
the performance of the Medical QA Assistant's RAG (Retrieval-Augmented Generation) system.
It defines data models for test questions and utilities to load them from a JSONL file.

The test questions include:
- Direct fact questions
- Questions requiring information spanning multiple documents
- Temporal questions involving time-based reasoning
- Questions with expected keywords for context retrieval validation
"""

import json
from pathlib import Path
from pydantic import BaseModel, Field

TEST_FILE = str(Path(__file__).parent / "tests.jsonl")


class TestQuestion(BaseModel):
    """
    A test question with expected keywords and reference answer.

    This Pydantic model represents a single test case for evaluating the QA system.
    Each test question has a question text, expected keywords that should appear in
    the retrieved context, a reference answer for comparison.
    """

    question: str = Field(description="The question to ask the RAG system")
    reference_answer: str = Field(description="The reference answer for this question")
    keywords: list[str] = Field(description="Keywords that must appear in retrieved context")


def load_tests() -> list[TestQuestion]:
    """
    Load test questions from JSONL file.

    This function reads the tests.jsonl file line by line, where each line is a
    JSON object representing a test question. It parses each line into a TestQuestion
    object using Pydantic's validation and returns a list of all test questions.

    Returns:
        list[TestQuestion]: A list of TestQuestion objects loaded from the file.

    Raises:
        FileNotFoundError: If tests.jsonl does not exist
        json.JSONDecodeError: If any line contains invalid JSON
        ValidationError: If any test question data doesn't match the expected schema
    """
    tests = []
    # Open the JSONL file for reading with UTF-8 encoding to handle special characters
    with open(TEST_FILE, "r", encoding="utf-8") as f:
        # Process each line in the file
        for line in f:
            # Parse the JSON line into a dictionary
            data = json.loads(line.strip())
            # Create a TestQuestion object from the data, which validates the structure
            tests.append(TestQuestion(**data))
    return tests