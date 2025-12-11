#!/usr/bin/env python3
"""
Tourism Route Optimization CLI
Dynamically loads and runs algorithms from src/ directory
"""

import sys
from pathlib import Path
import importlib.util


def discover_algorithms():
    """Discover all algorithm files in src/ directory."""
    src_dir = Path("src")
    algorithms = {}
    
    if not src_dir.exists():
        print("Error: src/ directory not found")
        return algorithms
    
    # Simulated Annealing algorithms
    sa_dir = src_dir / "sim_annealing"
    if sa_dir.exists():
        print("Found Simulated Annealing algorithms:")
        for file in sorted(sa_dir.glob("*.py")):
            if not file.name.startswith("_"):
                name = file.stem
                algorithms[f"SA - {name}"] = str(file)
                print(f"  - {name}")
    
    # Evolutionary algorithms
    ea_dir = src_dir / "evolutionary_algs"
    if ea_dir.exists():
        print("\nFound Evolutionary algorithms:")
        for file in sorted(ea_dir.glob("*.py")):
            if not file.name.startswith("_"):
                name = file.stem
                algorithms[f"EA - {name}"] = str(file)
                print(f"  - {name}")
    
    # Hybrid algorithms
    hybrid_dir = src_dir / "hybrid"
    if hybrid_dir.exists():
        print("\nFound Hybrid algorithms:")
        for file in sorted(hybrid_dir.glob("*.py")):
            if not file.name.startswith("_"):
                name = file.stem
                algorithms[f"Hybrid - {name}"] = str(file)
                print(f"  - {name}")
    
    return algorithms


def load_algorithm_module(filepath):
    """Dynamically load a Python module from filepath."""
    spec = importlib.util.spec_from_file_location("algorithm", filepath)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def select_problem():
    """Let user select a problem file."""
    data_dir = Path("data")
    if not data_dir.exists():
        print("Error: data/ directory not found")
        return None
    
    files = sorted([f for f in data_dir.glob("tourism_*.txt")])
    if not files:
        print("Error: No tourism_*.txt files in data/")
        return None
    
    print("\nAvailable test files:")
    for i, f in enumerate(files, 1):
        print(f"  {i}. {f.name}")
    
    try:
        choice = int(input("\nSelect (number): "))
        if 1 <= choice <= len(files):
            return str(files[choice - 1])
    except ValueError:
        pass
    
    print("Invalid selection")
    return None


def convert_to_float(value):
    """Convert any value to float, handling numpy arrays/scalars."""
    try:
        if hasattr(value, 'item'):  # numpy scalar
            return float(value.item())
        elif isinstance(value, (int, float)):
            return float(value)
        else:
            return float(value)
    except:
        return None


def run_algorithm(module, problem_file):
    """Run algorithm once and return the result."""
    try:
        print("Running algorithm...")
        result = module.run(problem_file)
        
        if result and len(result) >= 2:
            fitness = result[1]
            return convert_to_float(fitness)
        return None
    except Exception as e:
        print(f"Error running algorithm: {e}")
        return None


def main():
    """Main CLI loop."""
    print("\n" + "="*60)
    print("  TOURISM ROUTE OPTIMIZATION - Algorithm Runner")
    print("="*60)
    
    # Discover algorithms
    print("\nDiscovering algorithms...")
    algorithms = discover_algorithms()
    
    if not algorithms:
        print("Error: No algorithms found in src/ directory")
        return
    
    algo_list = sorted(list(algorithms.keys()))
    
    while True:
        print("\n" + "="*60)
        print("Available algorithms:")
        for i, name in enumerate(algo_list, 1):
            print(f"  {i}. {name}")
        print(f"  {len(algo_list) + 1}. Exit")
        
        try:
            choice = int(input("\nSelect algorithm (number): "))
            
            if choice == len(algo_list) + 1:
                print("\nGoodbye!\n")
                break
            
            if choice < 1 or choice > len(algo_list):
                print("Invalid choice")
                continue
            
            selected_algo_name = algo_list[choice - 1]
            algo_file = algorithms[selected_algo_name]
            
            # Load module
            print(f"\nLoading: {selected_algo_name}")
            module = load_algorithm_module(algo_file)
            
            if module is None:
                print(f"Error: Could not load {selected_algo_name}")
                continue
            
            if not hasattr(module, 'run'):
                print(f"Error: Algorithm does not have 'run' function")
                continue
            
            # Select problem
            problem_file = select_problem()
            if problem_file is None:
                continue
            
            # Run algorithm
            print("\n" + "-"*60)
            fitness = run_algorithm(module, problem_file)
            if fitness is not None:
                print(f"\nResult: {fitness:.6f}")
            print("-"*60)
        
        except ValueError:
            print("Invalid input")
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted.\n")
        sys.exit(0)