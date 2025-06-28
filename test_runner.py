"""
Test runner script to validate test structure and run basic tests.
"""

import sys
import subprocess
from pathlib import Path

def check_test_structure():
    """Check if test directory structure is correct."""
    test_dir = Path("tests")
    
    required_dirs = [
        "tests/unit/venice",
        "tests/unit/mcp", 
        "tests/unit/memory",
        "tests/unit/agent",
        "tests/unit/mcp_servers",
        "tests/integration",
        "tests/e2e"
    ]
    
    print("Checking test directory structure...")
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            print(f"✓ {dir_path} exists")
        else:
            print(f"✗ {dir_path} missing")
    
    test_files = list(test_dir.rglob("test_*.py"))
    print(f"\nFound {len(test_files)} test files:")
    for test_file in test_files:
        print(f"  - {test_file}")
    
    return len(test_files) > 0

def run_basic_validation():
    """Run basic pytest validation."""
    print("\nRunning pytest collection test...")
    try:
        result = subprocess.run([
            sys.executable, "-m", "pytest", 
            "--collect-only", "-q"
        ], capture_output=True, text=True, timeout=30)
        
        print(f"Return code: {result.returncode}")
        if result.stdout:
            print(f"STDOUT:\n{result.stdout}")
        if result.stderr:
            print(f"STDERR:\n{result.stderr}")
            
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print("Test collection timed out")
        return False
    except Exception as e:
        print(f"Error running pytest: {e}")
        return False

if __name__ == "__main__":
    print("PDX Hackerspace AI Agent - Test Validation")
    print("=" * 50)
    
    structure_ok = check_test_structure()
    if not structure_ok:
        print("Test structure validation failed")
        sys.exit(1)
    
    validation_ok = run_basic_validation()
    if validation_ok:
        print("\n✓ Test framework validation successful")
        sys.exit(0)
    else:
        print("\n✗ Test framework validation failed")
        sys.exit(1)
