#!/usr/bin/env python3
"""
Helper script to find and analyze fuzz testing candidate files in test/distributed.

Usage:
    python find_fuzz_candidates.py [options]

Options:
    --list              List all fuzz testing candidate files
    --stats             Show statistics about fuzz testing candidates
    --parametrized      Show only files with parametrized tests
    --random-data       Show only files with random data generation
    --both              Show only files with both criteria
    --export-json       Export results to JSON file
"""

import argparse
import json
import os
import re
from pathlib import Path
from collections import defaultdict


def parse_marker(content):
    """Parse the FUZZ_TESTING_CANDIDATE marker to extract reasons."""
    match = re.search(r'FUZZ_TESTING_CANDIDATE:\s*This test uses (.+)', content)
    if not match:
        return None
    
    reasons_text = match.group(1)
    reasons = []
    
    if 'parametrized' in reasons_text:
        reasons.append('parametrized')
    if 'random data' in reasons_text:
        reasons.append('random_data')
    if 'hypothesis' in reasons_text:
        reasons.append('hypothesis')
    
    return reasons


def find_fuzz_candidates(test_dir):
    """Find all fuzz testing candidate files."""
    results = []
    
    for root, dirs, files in os.walk(test_dir):
        for file in files:
            if file.startswith('test_') and file.endswith('.py'):
                filepath = Path(root) / file
                
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                except Exception:
                    continue
                
                if 'FUZZ_TESTING_CANDIDATE' in content:
                    reasons = parse_marker(content)
                    rel_path = filepath.relative_to(test_dir)
                    
                    results.append({
                        'path': str(rel_path),
                        'absolute_path': str(filepath),
                        'reasons': reasons or [],
                    })
    
    return sorted(results, key=lambda x: x['path'])


def print_stats(results):
    """Print statistics about fuzz testing candidates."""
    total = len(results)
    
    parametrized_count = sum(1 for r in results if 'parametrized' in r['reasons'])
    random_data_count = sum(1 for r in results if 'random_data' in r['reasons'])
    both_count = sum(1 for r in results if 'parametrized' in r['reasons'] and 'random_data' in r['reasons'])
    hypothesis_count = sum(1 for r in results if 'hypothesis' in r['reasons'])
    
    print("=" * 80)
    print("FUZZ TESTING CANDIDATES STATISTICS")
    print("=" * 80)
    print(f"Total candidates: {total}")
    print(f"  - With parametrized tests: {parametrized_count}")
    print(f"  - With random data generation: {random_data_count}")
    print(f"  - With both parametrized and random data: {both_count}")
    print(f"  - With hypothesis: {hypothesis_count}")
    print()
    
    # Category breakdown
    categories = defaultdict(int)
    for result in results:
        category = Path(result['path']).parts[0] if '/' in result['path'] else 'root'
        categories[category] += 1
    
    print("Breakdown by category:")
    for category, count in sorted(categories.items()):
        print(f"  - {category}: {count}")
    print("=" * 80)


def print_list(results, filter_func=None):
    """Print list of fuzz testing candidates."""
    if filter_func:
        results = [r for r in results if filter_func(r)]
    
    print(f"\nFound {len(results)} files:\n")
    for result in results:
        reasons = ", ".join(result['reasons'])
        print(f"  📝 {result['path']}")
        print(f"     Reasons: {reasons}")


def export_json(results, output_file):
    """Export results to JSON file."""
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results exported to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Find and analyze fuzz testing candidate files'
    )
    parser.add_argument('--list', action='store_true',
                       help='List all fuzz testing candidate files')
    parser.add_argument('--stats', action='store_true',
                       help='Show statistics')
    parser.add_argument('--parametrized', action='store_true',
                       help='Show only files with parametrized tests')
    parser.add_argument('--random-data', action='store_true',
                       help='Show only files with random data generation')
    parser.add_argument('--both', action='store_true',
                       help='Show only files with both criteria')
    parser.add_argument('--export-json', type=str,
                       help='Export results to JSON file')
    
    args = parser.parse_args()
    
    # Default to showing stats if no option is specified
    if not any([args.list, args.stats, args.parametrized, 
                args.random_data, args.both, args.export_json]):
        args.stats = True
    
    # Find test directory
    script_dir = Path(__file__).parent
    test_dir = script_dir
    
    # Find all candidates
    results = find_fuzz_candidates(test_dir)
    
    if args.stats:
        print_stats(results)
    
    if args.list:
        print_list(results)
    
    if args.parametrized:
        print("\nFiles with PARAMETRIZED TESTS:")
        print_list(results, lambda r: 'parametrized' in r['reasons'])
    
    if args.random_data:
        print("\nFiles with RANDOM DATA GENERATION:")
        print_list(results, lambda r: 'random_data' in r['reasons'])
    
    if args.both:
        print("\nFiles with BOTH parametrized tests AND random data:")
        print_list(results, lambda r: 'parametrized' in r['reasons'] and 'random_data' in r['reasons'])
    
    if args.export_json:
        export_json(results, args.export_json)


if __name__ == '__main__':
    main()
