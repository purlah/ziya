import sys
import os

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import unittest
import json
import tempfile
import shutil
import difflib
from app.utils.code_util import use_git_to_apply_code_diff, PatchApplicationError

class DiffRegressionTest(unittest.TestCase):
    """Regression tests for diff application using real-world examples"""
    
    maxDiff = None 

    # Directory containing test cases
    TEST_CASES_DIR = os.path.join(os.path.dirname(__file__), 'diff_test_cases')
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        # Set required environment variable
        os.environ['ZIYA_USER_CODEBASE_DIR'] = self.temp_dir
        
    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def get_test_files(self, case_dir: str, metadata: dict) -> tuple[str, str]:
        """
        Get the paths for original and expected files.
        First tries to find existing files, then falls back to extension from metadata.
        """
        # Common test file extensions
        extensions = ['.py', '.tsx', '.ts', '.js', '.jsx']
        
        # First try to find existing files with common extensions
        for ext in extensions:
            original = os.path.join(case_dir, f'original{ext}')
            expected = os.path.join(case_dir, f'expected{ext}')
            if os.path.exists(original) and os.path.exists(expected):
                return original, expected
        
        # If no files found, use extension from target file in metadata
        target_ext = os.path.splitext(metadata['target_file'])[1]
        if target_ext:
            original = os.path.join(case_dir, f'original{target_ext}')
            expected = os.path.join(case_dir, f'expected{target_ext}')
            return original, expected
            
        # Fall back to .py if nothing else works
        original = os.path.join(case_dir, 'original.py')
        expected = os.path.join(case_dir, 'expected.py')
        return original, expected

        
    def load_test_case(self, case_name):
        """Load a test case from the test cases directory"""
        case_dir = os.path.join(self.TEST_CASES_DIR, case_name)
        
        # Load metadata
        with open(os.path.join(case_dir, 'metadata.json'), 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        # Get correct file paths based on target file extension
        original_file, expected_file = self.get_test_files(case_dir, metadata)
        
        if not os.path.exists(original_file):
            raise FileNotFoundError(
                f"Could not find original file for test case '{case_name}'. "
                f"Tried: {original_file}"
            )
            
        # Load original file
        with open(original_file, 'r', encoding='utf-8') as f:
            original = f.read()
            
        # Load diff
        with open(os.path.join(case_dir, 'changes.diff'), 'r', encoding='utf-8') as f:
            diff = f.read()
            
        # Load expected result
        with open(expected_file, 'r', encoding='utf-8') as f:
            expected = f.read()
            
        return metadata, original, diff, expected
        
    def run_diff_test(self, case_name):
        """Run a single diff test case"""
        metadata, original, diff, expected = self.load_test_case(case_name)
        
        # Set up the test file in the temp directory
        test_file_path = os.path.join(self.temp_dir, metadata['target_file'])
        os.makedirs(os.path.dirname(test_file_path), exist_ok=True)
        
        with open(test_file_path, 'w', encoding='utf-8') as f:
            f.write(original)
            
        # Apply the diff
        use_git_to_apply_code_diff(diff,test_file_path)
        
        # Read the result
        with open(test_file_path, 'r', encoding='utf-8') as f:
            result = f.read()
            
        # Compare with expected
        if result != expected:
            # Generate a readable diff
            diff = difflib.unified_diff(
                expected.splitlines(True),
                result.splitlines(True),
                fromfile='Expected',
                tofile='Got'
            )

            # Create detailed error message
            error_msg = [
                "\nTest case '{}' failed:".format(case_name),
                "=" * 60,
                "Differences between expected and actual output:",
                "".join(diff),
                "-" * 60,
                "Expected file length: {} lines".format(len(expected.splitlines())),
                "Got file length: {} lines".format(len(result.splitlines())),
                "=" * 60
            ]

            self.fail("\n".join(error_msg))

    def test_all_cases(self):
        """Run all test cases found in the test cases directory"""
        for case_name in os.listdir(self.TEST_CASES_DIR):
            if os.path.isdir(os.path.join(self.TEST_CASES_DIR, case_name)):
                with self.subTest(case=case_name):
                    self.run_diff_test(case_name)

    def test_embedded_diff_markers(self):
        """Test handling content that contains diff-like markers"""
        self.run_diff_test('embedded_diff_markers')

    def test_indentation_changes(self):
        """Test preserving different indentation levels"""
        self.run_diff_test('indentation_changes')

    def test_multi_chunk_changes(self):
        """Test handling multiple chunks with embedded diff markers"""
        self.run_diff_test('multi_chunk_changes')

    def test_new_file_creation(self):
        """Test creating a new file from a diff"""
        self.run_diff_test('new_file_creation')

    def test_simple_nested(self):
        """Test adding a simple nested function"""
        self.run_diff_test('simple_nested')

    def test_two_functions(self):
        """Test adding nested function to second of two functions"""
        self.run_diff_test('two_functions')

    def test_multi_hunk_same_function(self):
        """Test multiple hunks modifying the same function"""
        self.run_diff_test('multi_hunk_same_function')

    def test_json_escape_sequence(self):
        """Test handling of JSON string content with escape sequences and comments"""
        self.run_diff_test('json_escape_sequence')

    def test_indentation_change(self):
        """Test changes that modify indentation levels"""
        self.run_diff_test('indentation_change')

    def test_function_collision(self):
        """Test handling of multiple functions with the same name"""
        self.run_diff_test('function_collision')

    def test_nested_function(self):
        """Test adding a nested function within an existing function"""
        self.run_diff_test('nested_function')


    def test_single_line_replace(self):
        """Test replacing a single line with multiple lines"""
        self.run_diff_test('single_line_replace')

    def test_new_file_new_dir(self):
        """Test creating a new file in a new directory"""
        self.run_diff_test('new_file_new_dir')

    def test_new_file_existing_dir(self):
        """Test creating a new file in an existing directory"""
        self.run_diff_test('new_file_existing_dir')

    def test_markdown_renderer_language_cache(self):
        """Test optimization of language loading in MarkdownRenderer"""
        self.run_diff_test('markdown_renderer_language_cache')
        
    def test_escape_sequence_content(self):
        """Test handling of escape sequences and content after them"""
        self.run_diff_test('escape_sequence_content')

    def test_import_line_order(self):
        """Test inserting an import line between existing imports"""
        self.run_diff_test('import_line_order')

class PrettyTestResult(unittest.TestResult):
    def __init__(self):
        super(PrettyTestResult, self).__init__()
        self.test_results = []
        self.current_test = None
    def startTest(self, test):
        self.current_test = test
    def addSuccess(self, test):
        self.test_results.append((test, 'PASS', None))
    def addError(self, test, err):
        self.test_results.append((test, 'ERROR', err))
    def addFailure(self, test, err):
        self.test_results.append((test, 'FAIL', err))
    def printSummary(self):
        print("\n" + "=" * 80)
        print("Test Results Summary")
        print("=" * 80)
        # Group results by test case
        results_by_case = {}
        for test, status, error in self.test_results:
            case_name = test._testMethodName
            if '(case=' in str(test):
                # Extract case name for parameterized tests
                case_name = f"{test._testMethodName} ({str(test).split('case=')[1].rstrip(')')}"
            results_by_case[case_name] = (status, error)
        # Print results in a table format
        print(f"{'Test Case':<50} {'Status':<10}")
        print("-" * 80)

        for case_name, (status, error) in sorted(results_by_case.items()):
            status_color = '\033[92m' if status == 'PASS' else '\033[91m'  # Green for pass, red for fail/error
            print(f"{case_name:<50} {status_color}{status}\033[0m")
            if error and status == 'ERROR':
                import traceback
                print(f"  └─ {''.join(traceback.format_exception(*error))}")
            elif error and status == 'FAIL':
                print(f"  └─ {str(error[1])}")        # Print overall summary
        print("\n" + "=" * 80)
        print(f"Total: {len(self.test_results)} tests")
        print(f"Passed: {len([r for r in self.test_results if r[1] == 'PASS'])} tests")
        print(f"Failed: {len([r for r in self.test_results if r[1] != 'PASS'])} tests")
        print("=" * 80 + "\n")

if __name__ == '__main__':
    import argparse

    def print_test_case_details(case_name=None):
        """Print details of test cases without running them"""
        test = DiffRegressionTest()
        cases = []

        # Get list of cases to show
        if case_name:
            if os.path.isdir(os.path.join(test.TEST_CASES_DIR, case_name)):
                cases = [case_name]
            else:
                print(f"Test case '{case_name}' not found")
                return
        else:
            cases = [d for d in os.listdir(test.TEST_CASES_DIR)
                    if os.path.isdir(os.path.join(test.TEST_CASES_DIR, d))]

        # Print details for each case
        for case in sorted(cases):
            try:
                metadata, original, diff, expected = test.load_test_case(case)
                print("\n" + "=" * 80)
                print(f"Test Case: {case}")
                print(f"Description: {metadata.get('description', 'No description')}")
                print("-" * 80)
                print("Original File:")
                print("-" * 80)
                print(original)
                print("-" * 80)
                print("Changes (diff):")
                print("-" * 80)
                print(diff)
                print("-" * 80)
                print("Expected Result:")
                print("-" * 80)
                print(expected)
            except Exception as e:
                print(f"Error loading test case '{case}': {str(e)}")
    parser = argparse.ArgumentParser()
    parser.add_argument('--show-cases', action='store_true',
                      help='Show test case details without running tests')
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('-k', '--test-filter', help='Only run tests matching this pattern')
    parser.add_argument('-l', '--log-level', default='INFO',
                      choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                      help='Set the log level')
    args = parser.parse_args()
 
    os.environ['ZIYA_LOG_LEVEL'] = args.log_level
 
    # If --show-cases is specified, print test case details and exit
    if args.show_cases:
        print_test_case_details(args.test_filter)
        sys.exit(0)
 
    # Otherwise run the tests normally
    suite = unittest.TestLoader().loadTestsFromTestCase(DiffRegressionTest)
    if args.test_filter:
        suite = unittest.TestLoader().loadTestsFromName(args.test_filter, DiffRegressionTest)
    result = PrettyTestResult()
    suite.run(result)
    result.printSummary()
    # Exit with appropriate status code
    sys.exit(len([r for r in result.test_results if r[1] != 'PASS']))
