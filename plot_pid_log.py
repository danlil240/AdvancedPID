#!/usr/bin/env python3
"""
Convenient executable script to plot PID CSV logs.

Usage:
    python plot_pid_log.py <csv_file> [options]

Examples:
    python plot_pid_log.py pid_log.csv
    python plot_pid_log.py pid_log.csv --comprehensive
    python plot_pid_log.py pid_log.csv --components
    python plot_pid_log.py pid_log.csv --report
    python plot_pid_log.py pid_log.csv --all --save
"""

import argparse
import sys
from pathlib import Path

from pid_control.analyzer.pid_analyzer import PIDAnalyzer
from pid_control.analyzer.plots import PIDPlotter


def main():
    parser = argparse.ArgumentParser(
        description='Plot PID control logs from CSV files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s pid_log.csv
  %(prog)s pid_log.csv --comprehensive
  %(prog)s pid_log.csv --components --frequency
  %(prog)s pid_log.csv --all --save output_dir/
  %(prog)s pid_log.csv --report --output report.txt
        """
    )
    
    parser.add_argument(
        'csv_file',
        type=str,
        help='Path to CSV log file'
    )
    
    parser.add_argument(
        '-r', '--response',
        action='store_true',
        help='Plot basic response (default if no other plot specified)'
    )
    
    parser.add_argument(
        '-c', '--comprehensive',
        action='store_true',
        help='Plot comprehensive multi-panel analysis'
    )
    
    parser.add_argument(
        '-p', '--components',
        action='store_true',
        help='Plot PID component breakdown'
    )
    
    parser.add_argument(
        '-f', '--frequency',
        action='store_true',
        help='Plot frequency domain analysis'
    )
    
    parser.add_argument(
        '-s', '--saturation',
        action='store_true',
        help='Plot saturation analysis (requires output limits)'
    )
    
    parser.add_argument(
        '-a', '--all',
        action='store_true',
        help='Generate all available plots'
    )
    
    parser.add_argument(
        '--report',
        action='store_true',
        help='Generate and print analysis report'
    )
    
    parser.add_argument(
        '--analyze',
        action='store_true',
        help='Perform analysis and print metrics'
    )
    
    parser.add_argument(
        '--save',
        type=str,
        metavar='DIR',
        help='Save plots to directory instead of displaying'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        metavar='FILE',
        help='Save report to file (use with --report)'
    )
    
    parser.add_argument(
        '--output-limits',
        type=float,
        nargs=2,
        metavar=('MIN', 'MAX'),
        help='Output limits for saturation analysis'
    )
    
    parser.add_argument(
        '--dpi',
        type=int,
        default=150,
        help='DPI for saved figures (default: 150)'
    )
    
    args = parser.parse_args()
    
    csv_path = Path(args.csv_file)
    if not csv_path.exists():
        print(f"Error: CSV file not found: {csv_path}", file=sys.stderr)
        sys.exit(1)
    
    print(f"Loading PID log from: {csv_path}")
    
    try:
        analyzer = PIDAnalyzer(str(csv_path))
    except Exception as e:
        print(f"Error loading CSV: {e}", file=sys.stderr)
        sys.exit(1)
    
    print(f"Loaded {len(analyzer.data.get('timestamp', []))} samples")
    print(f"Available columns: {', '.join(analyzer.columns)}")
    print()
    
    if args.analyze or args.report:
        print("Analyzing data...")
        output_limits = tuple(args.output_limits) if args.output_limits else None
        metrics = analyzer.analyze(output_limits=output_limits)
        print("Analysis complete.")
        print()
    
    if args.report:
        report = analyzer.generate_report()
        print(report)
        
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            analyzer.save_report(str(output_path))
            print(f"\nReport saved to: {output_path}")
        print()
    
    if args.save:
        save_dir = Path(args.save)
        save_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving plots to: {save_dir}")
    
    plot_any = args.response or args.comprehensive or args.components or args.frequency or args.saturation or args.all
    
    if not plot_any and not args.report and not args.analyze:
        args.response = True
    
    figures = []
    
    try:
        if args.response or args.all:
            print("Generating response plot...")
            fig = analyzer.plot_response()
            figures.append(('response', fig))
        
        if args.comprehensive or args.all:
            print("Generating comprehensive analysis plot...")
            fig = analyzer.plot_comprehensive()
            figures.append(('comprehensive', fig))
        
        if args.components or args.all:
            print("Generating PID components plot...")
            try:
                fig = analyzer.plot_pid_components()
                figures.append(('components', fig))
            except ValueError as e:
                print(f"  Skipped: {e}")
        
        if args.frequency or args.all:
            print("Generating frequency analysis plot...")
            fig = analyzer.plot_frequency_analysis()
            figures.append(('frequency', fig))
        
        if args.saturation or args.all:
            if args.output_limits:
                print("Generating saturation analysis plot...")
                output_limits = tuple(args.output_limits)
                fig = analyzer.plot_saturation(output_limits)
                figures.append(('saturation', fig))
            else:
                print("  Skipped saturation plot: --output-limits required")
        
    except Exception as e:
        print(f"Error generating plots: {e}", file=sys.stderr)
        sys.exit(1)
    
    if args.save:
        for name, fig in figures:
            filename = f"{csv_path.stem}_{name}.png"
            filepath = save_dir / filename
            PIDPlotter.save(fig, str(filepath), dpi=args.dpi)
            print(f"  Saved: {filepath}")
        print(f"\nAll plots saved to: {save_dir}")
    else:
        if figures:
            print(f"\nDisplaying {len(figures)} plot(s)...")
            PIDPlotter.show()


if __name__ == '__main__':
    main()
