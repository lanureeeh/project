#!/usr/bin/env python
"""
Time Logger CLI Tool

A command-line utility to track and log time spent on different project phases.
Logs are appended to a markdown file (time_log.md) in the project root directory.

Usage:
    python -m src.utils.time_logger --phase "EDA" --hours 1.5 --note "missingness plots"

Arguments:
    --phase: The project phase or activity (e.g., "EDA", "Modeling", "Data Cleaning")
    --hours: Time spent in hours (can be decimal, e.g., 1.5 for 1 hour and 30 minutes)
    --note: Optional note or description of the work done
"""
import argparse
import os
import datetime
from pathlib import Path

def log_time(phase, hours, note=None):
    """
    Log time spent on a project phase to time_log.md.
    
    Args:
        phase (str): Project phase or activity
        hours (float): Time spent in hours
        note (str, optional): Description of the work done
    """
    # Get the current date and time
    now = datetime.datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M")
    
    # Find the project root (assumed to be where time_log.md should be)
    # Try to find an existing time_log.md, or create it in the current working directory
    log_file = Path("time_log.md")
    
    # Create the file with header if it doesn't exist
    if not log_file.exists():
        with open(log_file, "w") as f:
            f.write("# Project Time Log\n\n")
            f.write("| Date | Time | Phase | Hours | Notes |\n")
            f.write("|------|------|-------|-------|-------|\n")
    
    # Format the log entry
    note_str = note if note else ""
    log_entry = f"| {date_str} | {time_str} | {phase} | {hours} | {note_str} |\n"
    
    # Append the log entry to the file
    with open(log_file, "a") as f:
        f.write(log_entry)
    
    print(f"Time logged: {phase} - {hours} hours - {date_str} {time_str}")

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Log time spent on project phases")
    
    parser.add_argument("--phase", type=str, required=True, 
                        help="Project phase or activity (e.g., 'EDA', 'Modeling')")
    
    parser.add_argument("--hours", type=float, required=True,
                        help="Time spent in hours (can be decimal, e.g., 1.5)")
    
    parser.add_argument("--note", type=str, default=None,
                        help="Optional description of the work done")
    
    args = parser.parse_args()
    
    log_time(args.phase, args.hours, args.note)

if __name__ == "__main__":
    main() 