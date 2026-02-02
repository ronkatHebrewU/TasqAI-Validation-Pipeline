"""
3. Practice Challenge: "The Log Parser"
Try this offline without a code editor first to see if you can "speak" the code:

Challenge: You have a list of strings: ["INFO: Task started", "ERROR: Connection failed", "INFO: Retrying..."].

Write a function that returns a dictionary counting the occurrences of each log level (INFO, ERROR). Constraint:

Use the .split() method and a dictionary.
"""
import json
from typing import List
from collections import defaultdict


def log_parser(logs: List):
    logTypeCounter = defaultdict(int)
    for log in logs:
        logType = log.split(":")[0]
        logTypeCounter[logType] += 1

    return logTypeCounter

"""
One Last Practice: The "Yield" Filter
Let's combine the Generator concept we just discussed with your Log Parser. This mimics a real scenario where you
 
have to process a file so big it won't fit in memory.

The Challenge: Modify your function to be a Generator called stream_errors.

It should take a file path as input.

It should read the file line-by-line.

It should only yield the lines that start with "ERROR".
"""


def stream_errors(file_path):
    """
    1. Open the file safely.
    2. Iterate line by line (Memory Efficient).
    3. Check if the line starts with "ERROR".
    4. If it does, extract the message and 'yield' it.
    """
    with open('file_path', 'r') as f:
        for line in f:
            yield json.loads(line)


def show_errors(logType, file_path):
    for error in stream_errors(file_path):
        error_split = error.split(":")
        if logType == error_split[0]:
            message = error_split[1].strip()
            print(f"error message:{message}")