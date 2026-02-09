#!/usr/bin/env python3
"""
Test script to demonstrate enhanced error logging capabilities.

This script shows different error logging patterns and their output.
"""

import sys
from pathlib import Path

# Import from python_magnetrun.hybrid module
from python_magnetrun.hybrid.utils import log_exception, format_exception_location


def test_full_traceback():
    """Test full traceback logging"""
    print("=" * 70)
    print("Test 1: Full Traceback with Print")
    print("=" * 70)

    def inner_function():
        raise ValueError("This is a test error in inner function")

    def outer_function():
        inner_function()

    try:
        outer_function()
    except Exception as e:
        log_exception(
            "Failed to execute outer function",
            e,
            use_print=True,
            include_traceback=True,
        )
    print()


def test_concise_location():
    """Test concise location reporting"""
    print("=" * 70)
    print("Test 2: Concise Error Location")
    print("=" * 70)

    def process_data(value):
        if value < 0:
            raise ValueError("Value must be non-negative")
        return value * 2

    try:
        result = process_data(-5)
    except Exception as e:
        log_exception(
            "Warning: Could not process data",
            e,
            use_print=True,
            include_traceback=False,
        )
    print()


def test_format_exception_location():
    """Test format_exception_location utility"""
    print("=" * 70)
    print("Test 3: Using format_exception_location() in error message")
    print("=" * 70)

    def load_file(path):
        if not Path(path).exists():
            raise FileNotFoundError(f"File not found: {path}")

    try:
        load_file("/nonexistent/file.txt")
    except FileNotFoundError as e:
        print(f"Error at {format_exception_location()}: {e}")
    print()


def test_nested_exceptions():
    """Test error reporting with nested function calls"""
    print("=" * 70)
    print("Test 4: Nested Function Calls")
    print("=" * 70)

    def level_3():
        raise RuntimeError("Error at deepest level")

    def level_2():
        level_3()

    def level_1():
        level_2()

    try:
        level_1()
    except Exception as e:
        log_exception(
            "Error in nested calls", e, use_print=True, include_traceback=True
        )
    print()


def test_type_error():
    """Test with different exception types"""
    print("=" * 70)
    print("Test 5: Different Exception Types")
    print("=" * 70)

    try:
        # TypeError
        result = "string" + 123
    except Exception as e:
        print(f"TypeError at {format_exception_location()}: {e}")

    try:
        # KeyError
        d = {"a": 1}
        value = d["b"]
    except Exception as e:
        print(f"KeyError at {format_exception_location()}: {e}")

    try:
        # IndexError
        lst = [1, 2, 3]
        value = lst[10]
    except Exception as e:
        log_exception("IndexError occurred", e, use_print=True, include_traceback=False)
    print()


def test_with_context():
    """Test error logging with contextual information"""
    print("=" * 70)
    print("Test 6: Error Logging with Context")
    print("=" * 70)

    def process_file(filename):
        raise IOError(f"Cannot read {filename}")

    filename = "data.txt"
    try:
        process_file(filename)
    except Exception as e:
        log_exception(
            f"Failed to process file '{filename}'",
            e,
            use_print=True,
            include_traceback=False,
        )
        print(f"  Location: {format_exception_location()}")
    print()


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Enhanced Error Logging - Test Suite")
    print("=" * 70)
    print()

    test_full_traceback()
    test_concise_location()
    test_format_exception_location()
    test_nested_exceptions()
    test_type_error()
    test_with_context()

    print("=" * 70)
    print("All tests completed!")
    print("=" * 70)
