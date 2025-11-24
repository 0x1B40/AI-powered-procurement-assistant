#!/usr/bin/env python3
"""
Minimal debug script to test compound query detection logic without full app dependencies.
"""

import json
import re
from typing import List

def _strip_code_fences(text: str) -> str:
    """Remove markdown code fences and trim whitespace."""
    if not text:
        return ""
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()
    return cleaned

def simulate_compound_detection(question: str) -> List[str]:
    """
    Simulate the compound query detection logic without LLM calls.
    This will help us understand if the issue is in the logic or the LLM response.
    """

    # Based on the prompt in classification.py, let's manually check what the expected behavior should be
    print(f"Analyzing question: '{question}'")

    # The prompt asks to split queries that contain multiple distinct requests
    # For the user's query: "give me The total number of orders created during the first quarter of 2014, and the last quarter of 2013 show them seperately,"

    # This clearly contains:
    # 1. "total number of orders created during the first quarter of 2014"
    # 2. "last quarter of 2013"

    # According to the example in the prompt:
    # Input: "give me The total number of orders created during the first quarter of 2014, and the last quarter of 2013 show them seperately, then add them up"
    # Output: ["How many orders were created during the first quarter of 2014?", "How many orders were created during the last quarter of 2013?"]

    expected_result = [
        "How many orders were created during the first quarter of 2014?",
        "How many orders were created during the last quarter of 2013?"
    ]

    print(f"Expected result: {expected_result}")
    return expected_result

def test_manual_logic():
    """Test the compound query detection logic manually."""

    # Test the exact query from the user
    test_query = "give me The total number of orders created during the first quarter of 2014, and the last quarter of 2013 show them seperately,"

    print("TESTING COMPOUND QUERY DETECTION LOGIC")
    print("=" * 80)
    print(f"Input query: '{test_query}'")
    print()

    # Manual analysis
    print("MANUAL ANALYSIS:")
    print("- Contains 'and' connecting two time periods")
    print("- Asks to 'show them seperately' (note: user has typo 'seperately' instead of 'separately')")
    print("- Should be split into two independent queries")
    print()

    expected = simulate_compound_detection(test_query)

    print(f"Expected detection result: {len(expected)} sub-queries")
    for i, sub_query in enumerate(expected, 1):
        print(f"  {i}. '{sub_query}'")

    print()
    print("CONCLUSION:")
    if len(expected) > 1:
        print("[SUCCESS] This SHOULD be detected as a compound query")
        print("[SUCCESS] Both sub-queries should execute")
        print("[SUCCESS] Response should show both results")
    else:
        print("[ISSUE] This should NOT be detected as compound (but it should be!)")

    print()
    print("If the application is only showing one result, the issue is likely:")
    print("1. LLM compound detection not working correctly")
    print("2. One of the sub-queries failing during pipeline generation")
    print("3. One of the sub-queries failing during MongoDB execution")
    print("4. Response formatting not handling compound results properly")

if __name__ == "__main__":
    test_manual_logic()
