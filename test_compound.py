#!/usr/bin/env python3

from src.llm_and_classification.classification import detect_compound_queries

# Test the compound query detection
test_query = "give me The total number of orders created during the first quarter of 2014, and the last quarter of 2013 show them seperately,"

print(f"Testing query: {test_query}")
result = detect_compound_queries(test_query)
print(f"Compound query detection result: {result}")
print(f"Number of sub-queries: {len(result)}")

if len(result) > 1:
    print("SUCCESS: Compound query detected!")
    for i, sub_query in enumerate(result):
        print(f"  Sub-query {i+1}: {sub_query}")
else:
    print("ISSUE: Not detected as compound query")

