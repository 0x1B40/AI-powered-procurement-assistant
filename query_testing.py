from __future__ import annotations

import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple
import re
from datetime import datetime

# Import your data loading functions
from src.data_loader import snake_case, sanitize_currency_series, normalize_chunk

def load_csv_to_dataframe(csv_path: Path, sample_size: int = None) -> pd.DataFrame:
    """Load CSV into pandas DataFrame using the same logic as data_loader.py"""
    print(f"Loading data from {csv_path}...")

    # Read the CSV (optionally sample for testing)
    if sample_size:
        df = pd.read_csv(csv_path, nrows=sample_size)
        print(f"Loaded sample of {len(df)} rows for testing")
    else:
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} rows")

    # Apply the same normalization as data_loader.py
    # normalize_chunk returns a list of dicts, so we convert back to DataFrame
    records = list(normalize_chunk(df))
    df_normalized = pd.DataFrame(records)

    return df_normalized

def create_test_queries() -> List[Dict[str, Any]]:
    """Create a comprehensive list of test queries, from simple to complex"""
    
    queries = [
        # SIMPLE QUERIES
        {
            "id": "simple_count",
            "natural_language": "How many purchase orders are there?",
            "pandas_query": "len(df)",
            "expected_result_type": "scalar",
            "category": "count"
        },
        {
            "id": "simple_departments",
            "natural_language": "What are the unique department names?",
            "pandas_query": "df['department_name'].unique().tolist()",
            "expected_result_type": "list",
            "category": "unique_values"
        },
        {
            "id": "simple_fiscal_years",
            "natural_language": "What fiscal years are in the data?",
            "pandas_query": "sorted(df['fiscal_year'].unique())",
            "expected_result_type": "list",
            "category": "unique_values"
        },
        
        # FILTERING QUERIES
        {
            "id": "filter_fiscal_year",
            "natural_language": "How many records are from fiscal year 2014-2015?",
            "pandas_query": "len(df[df['fiscal_year'] == '2014-2015'])",
            "expected_result_type": "scalar",
            "category": "filter_count"
        },
        {
            "id": "filter_department",
            "natural_language": "How many records are from the Department of Consumer Affairs?",
            "pandas_query": "len(df[df['department_name'] == 'Consumer Affairs, Department of'])",
            "expected_result_type": "scalar",
            "category": "filter_count"
        },
        {
            "id": "filter_acquisition_type",
            "natural_language": "How many IT Goods purchases are there?",
            "pandas_query": "len(df[df['acquisition_type'] == 'IT Goods'])",
            "expected_result_type": "scalar",
            "category": "filter_count"
        },
        
        # AGGREGATION QUERIES
        {
            "id": "top_departments_spend",
            "natural_language": "What are the top 5 departments by total spend?",
            "pandas_query": "df.groupby('department_name')['total_price'].sum().sort_values(ascending=False).head(5).to_dict()",
            "expected_result_type": "dict",
            "category": "aggregation"
        },
        {
            "id": "total_spend_fiscal_year",
            "natural_language": "What is the total spend by fiscal year?",
            "pandas_query": "df.groupby('fiscal_year')['total_price'].sum().sort_values(ascending=False).to_dict()",
            "expected_result_type": "dict",
            "category": "aggregation"
        },
        {
            "id": "avg_order_value",
            "natural_language": "What is the average order value?",
            "pandas_query": "df['total_price'].mean()",
            "expected_result_type": "scalar",
            "category": "aggregation"
        },
        
        # COMPLEX FILTERING
        {
            "id": "spend_by_dept_fiscal",
            "natural_language": "What is the total spend by department for fiscal year 2014-2015?",
            "pandas_query": "df[df['fiscal_year'] == '2014-2015'].groupby('department_name')['total_price'].sum().sort_values(ascending=False).head(10).to_dict()",
            "expected_result_type": "dict",
            "category": "complex_filter"
        },
        {
            "id": "it_goods_2014_spend",
            "natural_language": "What is the total spend on IT Goods in fiscal year 2014-2015?",
            "pandas_query": "df[(df['acquisition_type'] == 'IT Goods') & (df['fiscal_year'] == '2014-2015')]['total_price'].sum()",
            "expected_result_type": "scalar",
            "category": "complex_filter"
        },
        {
            "id": "supplier_count_by_type",
            "natural_language": "How many unique suppliers are there for each acquisition type?",
            "pandas_query": "df.groupby('acquisition_type')['supplier_name'].nunique().sort_values(ascending=False).to_dict()",
            "expected_result_type": "dict",
            "category": "complex_aggregation"
        },
        
        # DATE-BASED QUERIES
        {
            "id": "creation_date_range",
            "natural_language": "How many orders were created in 2014?",
            "pandas_query": "len(df[df['creation_date'].str.contains('2014')])",
            "expected_result_type": "scalar",
            "category": "date_filter"
        },
        
        # TOP N QUERIES
        {
            "id": "top_suppliers",
            "natural_language": "Who are the top 5 suppliers by total spend?",
            "pandas_query": "df.groupby('supplier_name')['total_price'].sum().sort_values(ascending=False).head(5).to_dict()",
            "expected_result_type": "dict",
            "category": "top_n"
        },
        {
            "id": "largest_orders",
            "natural_language": "What are the 5 largest orders by total price?",
            "pandas_query": "df.nlargest(5, 'total_price')[['purchase_order_number', 'supplier_name', 'total_price', 'department_name']].to_dict('records')",
            "expected_result_type": "list_of_dicts",
            "category": "top_n"
        },
        
        # PERCENTAGE AND RATIO QUERIES
        {
            "id": "calcard_percentage",
            "natural_language": "What percentage of orders use CalCard?",
            "pandas_query": "(df['calcard'] == 'YES').mean() * 100",
            "expected_result_type": "scalar",
            "category": "percentage"
        },
        {
            "id": "acquisition_type_distribution",
            "natural_language": "What is the distribution of acquisition types?",
            "pandas_query": "df['acquisition_type'].value_counts().to_dict()",
            "expected_result_type": "dict",
            "category": "distribution"
        }
    ]
    
    return queries

def execute_pandas_query(df: pd.DataFrame, query: str) -> Any:
    """Safely execute a pandas query and return the result"""
    try:
        # Create a safe local environment with common built-ins
        safe_builtins = {
            'len': len,
            'sorted': sorted,
            'sum': sum,
            'min': min,
            'max': max,
            'abs': abs,
            'round': round,
            'int': int,
            'float': float,
            'str': str,
            'bool': bool,
        }
        local_vars = {'df': df, 'pd': pd}
        result = eval(query, {"__builtins__": safe_builtins}, local_vars)
        return result
    except Exception as e:
        return f"Error executing query: {str(e)}"

def format_result(result: Any) -> str:
    """Format result for display"""
    if isinstance(result, (int, float)):
        if isinstance(result, float):
            return ".2f"
        return str(result)
    elif isinstance(result, dict):
        return json.dumps(result, indent=2)
    elif isinstance(result, list):
        return json.dumps(result, indent=2)
    else:
        return str(result)

def run_query_test(df: pd.DataFrame, query_info: Dict[str, Any]) -> Dict[str, Any]:
    """Run a single query test and return results"""
    print(f"\n{'='*60}")
    print(f"TEST: {query_info['id']}")
    print(f"CATEGORY: {query_info['category']}")
    print(f"{'='*60}")
    
    print(f"Natural Language: {query_info['natural_language']}")
    print(f"Pandas Query: {query_info['pandas_query']}")
    
    result = execute_pandas_query(df, query_info['pandas_query'])
    formatted_result = format_result(result)
    
    print(f"Expected Result Type: {query_info['expected_result_type']}")
    print(f"Actual Result:\n{formatted_result}")
    
    return {
        "query_info": query_info,
        "result": result,
        "formatted_result": formatted_result,
        "success": not isinstance(result, str) or not result.startswith("Error")
    }

def main():
    """Main function to run the query testing suite"""
    print("California Procurement Data Query Testing Suite")
    print("=" * 60)
    
    # Load data
    csv_path = Path("data/PURCHASE ORDER DATA EXTRACT.csv")
    
    # For testing, use a sample first
    sample_size = 10000  # Adjust this based on your needs
    df = load_csv_to_dataframe(csv_path, sample_size=sample_size)
    
    print(f"\nDataFrame shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Get test queries
    queries = create_test_queries()
    
    print(f"\nLoaded {len(queries)} test queries")
    
    # Run tests
    results = []
    for query_info in queries:
        result = run_query_test(df, query_info)
        results.append(result)
    
    # Summary
    successful = sum(1 for r in results if r['success'])
    total = len(results)
    
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Total queries: {total}")
    print(f"Successful: {successful}")
    print(f"Failed: {total - successful}")
    print(".1f")
    
    # Save results to file
    output_file = "query_test_results.json"
    with open(output_file, 'w') as f:
        json.dump({
            "metadata": {
                "total_rows": len(df),
                "sample_size": sample_size,
                "total_queries": total,
                "successful_queries": successful
            },
            "results": [
                {
                    "id": r["query_info"]["id"],
                    "natural_language": r["query_info"]["natural_language"],
                    "pandas_query": r["query_info"]["pandas_query"],
                    "expected_type": r["query_info"]["expected_result_type"],
                    "success": r["success"],
                    "result_preview": str(r["result"])[:200] + "..." if len(str(r["result"])) > 200 else str(r["result"])
                }
                for r in results
            ]
        }, f, indent=2)
    
    print(f"\nDetailed results saved to {output_file}")
    
    print("\n" + "="*60)
    print("USAGE INSTRUCTIONS")
    print("="*60)
    print("1. Run your Streamlit UI: streamlit run src/ui.py")
    print("2. Ask your agent the natural language questions above")
    print("3. Compare the agent's MongoDB results with the pandas results here")
    print("4. Debug any discrepancies in your agent's query generation")
    print("\nFor full dataset testing, set sample_size=None in main()")

if __name__ == "__main__":
    main()
