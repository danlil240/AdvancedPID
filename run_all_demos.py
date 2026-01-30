#!/usr/bin/env python3
"""
Run All Demos - Master launcher for all PID demonstrations.

This script provides a menu to run individual demos or all of them.
"""

import sys
import os
from pathlib import Path

# Ensure we can import from the project
sys.path.insert(0, str(Path(__file__).parent))


def print_header():
    """Print welcome header."""
    print("\n" + "=" * 70)
    print("   ADVANCED PID CONTROL LIBRARY - DEMONSTRATION SUITE")
    print("=" * 70)


def print_menu():
    """Print demo menu."""
    print("\nAvailable Demonstrations:")
    print("-" * 40)
    print("  1. Basic PID Demo")
    print("     - Simple step response")
    print("     - CSV logging")
    print("     - Basic analysis")
    print()
    print("  2. Auto-Tuning Demo")
    print("     - Differential evolution tuning")
    print("     - Ziegler-Nichols comparison")
    print("     - Performance comparison")
    print()
    print("  3. Advanced Features Demo")
    print("     - Anti-windup methods")
    print("     - Derivative filtering")
    print("     - Setpoint weighting")
    print("     - Bumpless transfer")
    print()
    print("  4. Spectacular Simulations")
    print("     - 3D phase space")
    print("     - Gain performance surfaces")
    print("     - Multi-plant comparison")
    print("     - Robustness analysis")
    print()
    print("  5. Interactive Animated Demo")
    print("     - Real-time parameter adjustment")
    print("     - Live updating plots")
    print()
    print("  6. Run ALL demos")
    print("  0. Exit")
    print("-" * 40)


def run_demo(demo_name: str):
    """Run a specific demo."""
    demos = {
        '1': ('examples.demo_basic', 'Basic PID Demo'),
        '2': ('examples.demo_tuning', 'Auto-Tuning Demo'),
        '3': ('examples.demo_advanced_features', 'Advanced Features Demo'),
        '4': ('examples.demo_spectacular_simulations', 'Spectacular Simulations'),
        '5': ('examples.demo_animated', 'Interactive Animated Demo'),
    }
    
    if demo_name not in demos:
        print("Invalid selection.")
        return False
    
    module_name, title = demos[demo_name]
    
    print(f"\n{'=' * 70}")
    print(f"   Running: {title}")
    print('=' * 70)
    
    try:
        # Import and run the demo module
        import importlib
        module = importlib.import_module(module_name)
        
        if hasattr(module, 'main'):
            module.main()
        else:
            print(f"Demo module {module_name} has no main() function.")
            return False
        
        return True
        
    except ImportError as e:
        print(f"Failed to import demo: {e}")
        print("Make sure all dependencies are installed: pip install -r requirements.txt")
        return False
    except Exception as e:
        print(f"Demo error: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_demos():
    """Run all demos in sequence."""
    print("\n" + "=" * 70)
    print("   RUNNING ALL DEMONSTRATIONS")
    print("=" * 70)
    print("\nNote: Close each plot window to proceed to the next demo.\n")
    
    for i in ['1', '2', '3', '4', '5']:
        success = run_demo(i)
        if not success:
            print(f"\nDemo {i} failed. Continue anyway? (y/n): ", end='')
            response = input().strip().lower()
            if response != 'y':
                break
        print("\n" + "-" * 70)
    
    print("\n" + "=" * 70)
    print("   ALL DEMONSTRATIONS COMPLETE")
    print("=" * 70)


def main():
    """Main entry point."""
    print_header()
    
    # Check dependencies
    try:
        import numpy
        import scipy
        import matplotlib
        print("\n✓ All required dependencies are installed.")
    except ImportError as e:
        print(f"\n✗ Missing dependency: {e}")
        print("  Please run: pip install -r requirements.txt")
        return
    
    while True:
        print_menu()
        
        choice = input("\nEnter your choice (0-6): ").strip()
        
        if choice == '0':
            print("\nThank you for using the Advanced PID Control Library!")
            print("Goodbye.\n")
            break
        elif choice == '6':
            run_all_demos()
        elif choice in ['1', '2', '3', '4', '5']:
            run_demo(choice)
        else:
            print("\nInvalid choice. Please enter 0-6.")


if __name__ == "__main__":
    main()
