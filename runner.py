"""
Advanced Experiment Runner - Version 3 (Enhanced)
More flexible handling of different algorithm interfaces and return types
Now exports detailed solutions and comprehensive analysis
"""

import os
import sys
import time
import importlib.util
import glob
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import pandas as pd
from datetime import datetime
import traceback
import json

# Configuration
NUM_RUNS = 10
TIMEOUT_SECONDS = 3000
TEST_FILES_DIR = "./data"
OUTPUT_DIR = "./results"
ALGORITHMS_BASE_DIR = "./src"
VERBOSE = True  # Set to False for less output


class DetailedExperimentRunner:
    def __init__(self, num_runs: int = 10, timeout: int = 300, verbose: bool = True):
        self.num_runs = num_runs
        self.timeout = timeout
        self.verbose = verbose
        self.results = {}  # test_name -> algo_name -> stats
        self.detailed_runs = {}  # For storing individual run data with solutions
        self.best_solutions = {}  # For storing best solution per algorithm per test file
        self.create_output_dir()
    
    def create_output_dir(self):
        os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    def log(self, message: str):
        if self.verbose:
            print(message)
    
    def get_test_files(self) -> List[str]:
        """Find all test files"""
        if not os.path.exists(TEST_FILES_DIR):
            self.log(f"⚠ Test files directory '{TEST_FILES_DIR}' not found")
            return []
        
        test_files = sorted(glob.glob(f"{TEST_FILES_DIR}/tourism_*.txt"))
        return test_files
    
    def discover_algorithms(self) -> Dict[str, str]:
        """Discover all algorithm files with smart categorization"""
        algorithms = {}
        
        if not os.path.exists(ALGORITHMS_BASE_DIR):
            self.log(f"⚠ Algorithms directory '{ALGORITHMS_BASE_DIR}' not found")
            return algorithms
        
        # Walk through the structure
        for root, dirs, files in os.walk(ALGORITHMS_BASE_DIR):
            for file in files:
                if not file.endswith('.py') or file.startswith('__'):
                    continue
                
                full_path = os.path.join(root, file)
                relative_path = os.path.relpath(root, ALGORITHMS_BASE_DIR)
                
                # Extract category from directory structure
                if relative_path == '.':
                    category = 'root'
                else:
                    parts = relative_path.split(os.sep)
                    category = parts[0]
                
                algo_name = file[:-3]  # Remove .py
                key = f"{category}/{algo_name}" if category != 'root' else algo_name
                
                algorithms[key] = full_path
        
        return algorithms
    
    def import_algorithm(self, module_path: str) -> Optional[Any]:
        """Safely import an algorithm module"""
        try:
            spec = importlib.util.spec_from_file_location(
                os.path.basename(module_path).replace('.py', ''),
                module_path
            )
            if spec is None or spec.loader is None:
                return None
            
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
        except Exception as e:
            self.log(f"  ✗ Import error: {str(e)[:80]}")
            return None
    
    def find_executable_function(self, module: Any) -> Tuple[Optional[callable], str]:
        """
        Find the main executable function in a module
        Tries common names: run, optimize, execute, solve
        """
        function_names = ['run', 'optimize', 'execute', 'solve', 'main']
        
        for func_name in function_names:
            if hasattr(module, func_name):
                func = getattr(module, func_name)
                if callable(func):
                    return func, func_name
        
        return None, None
    
    def parse_result(self, result: Any) -> Dict[str, Any]:
        """
        Parse algorithm result from various possible formats
        """
        parsed = {
            'fitness': None,
            'solution': None,
            'iterations': None,
            'time': None,
            'valid': False,
            'raw_type': type(result).__name__
        }
        
        try:
            if result is None:
                return parsed
            
            # Handle tuple returns (fitness, solution, iterations, time, ...)
            elif isinstance(result, tuple) and len(result) > 0:
                if len(result) >= 1 and isinstance(result[0], (int, float)):
                    parsed['fitness'] = float(result[0])
                    parsed['valid'] = True
                
                if len(result) >= 2:
                    parsed['solution'] = result[1]
                if len(result) >= 3:
                    parsed['iterations'] = result[2]
                if len(result) >= 4:
                    parsed['time'] = result[3]
            
            # Handle dictionary returns
            elif isinstance(result, dict):
                # Try common key names
                fitness_keys = ['fitness', 'best_fitness', 'objective', 'cost', 'score']
                for key in fitness_keys:
                    if key in result:
                        val = result[key]
                        if isinstance(val, (int, float)):
                            parsed['fitness'] = float(val)
                            parsed['valid'] = True
                            break
                
                solution_keys = ['solution', 'best_solution', 'result']
                for key in solution_keys:
                    if key in result:
                        parsed['solution'] = result[key]
                        break
                
                parsed['iterations'] = result.get('iterations')
                parsed['time'] = result.get('time', result.get('elapsed_time'))
            
            # Handle single numeric return
            elif isinstance(result, (int, float)):
                parsed['fitness'] = float(result)
                parsed['valid'] = True
            
            else:
                # Try to convert to float as last resort
                try:
                    parsed['fitness'] = float(result)
                    parsed['valid'] = True
                except (ValueError, TypeError):
                    pass
        
        except Exception as e:
            self.log(f"  Parse error: {str(e)[:60]}")
        
        return parsed
    
    def solution_to_string(self, solution: Any) -> str:
        """Convert solution to string for Excel export"""
        if solution is None:
            return ""
        try:
            if hasattr(solution, '__iter__') and not isinstance(solution, str):
                return " ".join(str(int(x)) for x in solution)
            return str(solution)
        except:
            return str(solution)
    
    def run_algorithm_once(self, module: Any, test_file: str, 
                           func: callable, func_name: str) -> Dict[str, Any]:
        """
        Run a single algorithm instance once
        """
        result_data = {
            'status': 'error',
            'fitness': None,
            'solution': None,
            'solution_str': '',
            'solution_size': 0,
            'time': 0,
            'iterations': None,
            'error': None
        }
        
        try:
            start_time = time.time()
            
            # Call the algorithm - support both file path and other input styles
            try:
                # Try passing the test file path
                raw_result = func(test_file)
            except TypeError:
                # If that fails, try calling without arguments
                raw_result = func()
            
            elapsed = time.time() - start_time
            
            # Parse the result
            parsed = self.parse_result(raw_result)
            
            if parsed['valid'] and parsed['fitness'] is not None:
                result_data['status'] = 'success'
                result_data['fitness'] = parsed['fitness']
                result_data['solution'] = parsed['solution']
                result_data['solution_str'] = self.solution_to_string(parsed['solution'])
                result_data['time'] = elapsed
                result_data['iterations'] = parsed['iterations']
                
                # Calculate solution size
                if parsed['solution'] is not None:
                    try:
                        if hasattr(parsed['solution'], '__len__'):
                            result_data['solution_size'] = len(parsed['solution'])
                    except:
                        pass
            else:
                result_data['status'] = 'invalid_result'
                result_data['error'] = f"Could not extract valid fitness from {parsed['raw_type']}"
        
        except Exception as e:
            result_data['status'] = 'exception'
            result_data['error'] = str(e)
            result_data['time'] = time.time() - start_time
        
        return result_data
    
    def run_experiments(self):
        """Execute all experiments"""
        test_files = self.get_test_files()
        algorithms = self.discover_algorithms()
        
        if not test_files:
            print("❌ No test files found in './data/tourism_*.txt'")
            return
        
        if not algorithms:
            print("❌ No algorithms found in './src'")
            return
        
        print("\n" + "="*70)
        print("EXPERIMENT RUNNER - DETAILED EXECUTION (v3 Enhanced)")
        print("="*70)
        print(f"✓ Test files: {len(test_files)}")
        for tf in test_files:
            print(f"  - {os.path.basename(tf)}")
        
        print(f"\n✓ Algorithm variations: {len(algorithms)}")
        for algo in sorted(algorithms.keys())[:5]:
            print(f"  - {algo}")
        if len(algorithms) > 5:
            print(f"  ... and {len(algorithms) - 5} more")
        
        print(f"\n✓ Configuration:")
        print(f"  - Runs per algorithm: {self.num_runs}")
        print(f"  - Output directory: {OUTPUT_DIR}")
        print("="*70 + "\n")
        
        # Main experiment loop
        total_experiments = len(test_files) * len(algorithms)
        current_exp = 0
        
        for test_file in test_files:
            test_name = os.path.basename(test_file).replace('.txt', '')
            self.results[test_name] = {}
            self.detailed_runs[test_name] = {}
            self.best_solutions[test_name] = {}
            
            print(f"\n{'─'*70}")
            print(f"Test File: {test_name}")
            print(f"{'─'*70}")
            
            for algo_name, algo_path in sorted(algorithms.items()):
                current_exp += 1
                
                # Import the algorithm
                module = self.import_algorithm(algo_path)
                if module is None:
                    self.log(f"[{current_exp}/{total_experiments}] {algo_name:<50} ✗ Import failed")
                    continue
                
                # Find the main function
                func, func_name = self.find_executable_function(module)
                if func is None:
                    self.log(f"[{current_exp}/{total_experiments}] {algo_name:<50} ✗ No executable function")
                    continue
                
                # Run the algorithm multiple times
                runs_data = []
                success_count = 0
                best_fitness_overall = -float('inf')
                best_run_idx = -1
                
                for run_num in range(self.num_runs):
                    result = self.run_algorithm_once(module, test_file, func, func_name)
                    runs_data.append(result)
                    
                    if result['status'] == 'success':
                        success_count += 1
                        if result['fitness'] > best_fitness_overall:
                            best_fitness_overall = result['fitness']
                            best_run_idx = run_num
                
                # Calculate statistics
                self.detailed_runs[test_name][algo_name] = runs_data
                
                # Store best solution for this algorithm on this test file
                if best_run_idx >= 0:
                    best_run = runs_data[best_run_idx]
                    self.best_solutions[test_name][algo_name] = {
                        'fitness': best_run['fitness'],
                        'solution': best_run['solution_str'],
                        'solution_size': best_run['solution_size'],
                        'run_number': best_run_idx + 1,
                        'time': best_run['time']
                    }
                
                if success_count > 0:
                    successful_runs = [r for r in runs_data if r['status'] == 'success']
                    fitnesses = [r['fitness'] for r in successful_runs]
                    times = [r['time'] for r in successful_runs]
                    
                    stats = {
                        'best': max(fitnesses),
                        'worst': min(fitnesses),
                        'avg': sum(fitnesses) / len(fitnesses),
                        'std': self._std_dev(fitnesses),
                        'avg_time': sum(times) / len(times),
                        'success_rate': success_count / self.num_runs,
                        'successful_runs': success_count
                    }
                    
                    self.results[test_name][algo_name] = stats
                    
                    # Print progress
                    status_str = "✓" if success_count == self.num_runs else f"⚠ {success_count}/{self.num_runs}"
                    avg_str = f"{stats['avg']:.2f}"
                    best_str = f"{stats['best']:.2f}"
                    time_str = f"{stats['avg_time']:.2f}s"
                    
                    print(f"[{current_exp}/{total_experiments}] {algo_name:<50} {status_str} avg:{avg_str} best:{best_str} time:{time_str}")
                else:
                    print(f"[{current_exp}/{total_experiments}] {algo_name:<50} ✗ All runs failed")
    
    def _std_dev(self, values: List[float]) -> float:
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance ** 0.5
    
    def export_to_excel(self) -> str:
        """Export detailed results to Excel"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = os.path.join(OUTPUT_DIR, f"experiment_results_{timestamp}.xlsx")
        
        print(f"\n\n{'='*70}")
        print("EXPORTING RESULTS TO EXCEL")
        print(f"{'='*70}")
        
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            
            # Summary sheet
            self._create_summary_sheet(writer)
            
            # Best solutions sheet
            self._create_best_solutions_sheet(writer)
            
            # Per-test-file sheets
            for test_name in sorted(self.results.keys()):
                self._create_test_sheet(writer, test_name)
            
            # Detailed runs sheet
            self._create_detailed_sheet(writer)
        
        print(f"✓ Results saved to: {output_file}")
        return output_file
    
    def _create_summary_sheet(self, writer):
        """Create overview summary"""
        summary_data = []
        
        for test_name in sorted(self.results.keys()):
            algo_results = self.results[test_name]
            if not algo_results:
                continue
            
            best_algo_name = max(algo_results, key=lambda x: algo_results[x]['avg'])
            best_stats = algo_results[best_algo_name]
            
            summary_data.append({
                'Test File': test_name,
                'Best Algorithm': best_algo_name,
                'Avg Fitness': f"{best_stats['avg']:.4f}",
                'Best Fitness': f"{best_stats['best']:.4f}",
                'Std Dev': f"{best_stats['std']:.4f}",
                'Avg Time (s)': f"{best_stats['avg_time']:.2f}",
                'Success Rate': f"{best_stats['success_rate']*100:.0f}%"
            })
        
        if summary_data:
            df = pd.DataFrame(summary_data)
            df.to_excel(writer, sheet_name='Summary', index=False)
            self._format_worksheet(writer.sheets['Summary'])
    
    def _create_best_solutions_sheet(self, writer):
        """Create sheet with best solution for each algorithm per test file"""
        all_best_solutions = []
        
        for test_name in sorted(self.best_solutions.keys()):
            for algo_name in sorted(self.best_solutions[test_name].keys()):
                best_sol = self.best_solutions[test_name][algo_name]
                all_best_solutions.append({
                    'Test File': test_name,
                    'Algorithm': algo_name,
                    'Best Fitness': f"{best_sol['fitness']:.4f}",
                    'Best Run #': best_sol['run_number'],
                    'Solution Size': best_sol['solution_size'],
                    'Best Solution': best_sol['solution'],
                    'Time (s)': f"{best_sol['time']:.2f}"
                })
        
        if all_best_solutions:
            df = pd.DataFrame(all_best_solutions)
            df.to_excel(writer, sheet_name='Best_Solutions', index=False)
            self._format_worksheet(writer.sheets['Best_Solutions'])
    
    def _create_test_sheet(self, writer, test_name: str):
        """Create detailed sheet for each test file"""
        algo_results = self.results.get(test_name, {})
        
        data = []
        for algo_name, stats in algo_results.items():
            data.append({
                'Algorithm': algo_name,
                'Best': f"{stats['best']:.4f}",
                'Worst': f"{stats['worst']:.4f}",
                'Average': f"{stats['avg']:.4f}",
                'Std Dev': f"{stats['std']:.4f}",
                'Avg Time (s)': f"{stats['avg_time']:.2f}",
                'Success Rate': f"{stats['success_rate']*100:.0f}%",
                'Successful Runs': stats['successful_runs']
            })
        
        # Sort by average (best first)
        df = pd.DataFrame(data)
        if not df.empty:
            df = df.sort_values('Average', key=lambda x: x.str.replace('.', '', 1).astype(float), ascending=False)
        
        sheet_name = test_name[:31]  # Excel limit
        df.to_excel(writer, sheet_name=sheet_name, index=False)
        self._format_worksheet(writer.sheets[sheet_name])
    
    def _create_detailed_sheet(self, writer):
        """Create detailed per-run data with solutions"""
        all_runs_data = []
        
        for test_name in sorted(self.detailed_runs.keys()):
            for algo_name in sorted(self.detailed_runs[test_name].keys()):
                runs = self.detailed_runs[test_name][algo_name]
                
                for run_num, run_data in enumerate(runs, 1):
                    all_runs_data.append({
                        'Test File': test_name,
                        'Algorithm': algo_name,
                        'Run #': run_num,
                        'Status': run_data['status'],
                        'Fitness': f"{run_data['fitness']:.4f}" if run_data['fitness'] else "N/A",
                        'Solution Size': run_data['solution_size'] if run_data['solution_size'] > 0 else "N/A",
                        'Solution': run_data['solution_str'],
                        'Iterations': run_data['iterations'] if run_data['iterations'] else "N/A",
                        'Time (s)': f"{run_data['time']:.2f}" if run_data['time'] else "N/A",
                        'Error': run_data['error'] if run_data['error'] else ""
                    })
        
        if all_runs_data:
            df = pd.DataFrame(all_runs_data)
            df.to_excel(writer, sheet_name='All_Runs', index=False)
            self._format_worksheet(writer.sheets['All_Runs'])
    
    def _format_worksheet(self, worksheet):
        """Apply formatting to a worksheet"""
        for column in worksheet.columns:
            max_length = max(
                (len(str(cell.value)) for cell in column if cell.value),
                default=10
            )
            adjusted_width = min(max_length + 2, 50)
            worksheet.column_dimensions[column[0].column_letter].width = adjusted_width


def main():
    runner = DetailedExperimentRunner(num_runs=NUM_RUNS, timeout=TIMEOUT_SECONDS, verbose=VERBOSE)
    runner.run_experiments()
    runner.export_to_excel()
    print("\n✓ Experiment complete!")


if __name__ == "__main__":
    main()