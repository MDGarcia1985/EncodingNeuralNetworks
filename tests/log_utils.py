# Purpose: Provide shared logging helpers for pytest result output.
# Design: Append human-readable test entries without changing assertion behavior.
# Workflow: Run a test body, capture pass or failure status, print details, and write the log.
#
# Michael Garcia
# michael@mandedesign.studio
# https://mandedesign.studio
#
# License: MIT

from datetime import datetime
from pathlib import Path
import re


LOG_PATH = Path(__file__).resolve().parent / "data" / "testLog.txt"


# Purpose: Find the next numbered entry for the test log.
# Design: Read existing log headings and continue from the highest test number.
# Workflow: Return 1 for a missing or empty log, otherwise return max number plus one.
def _next_test_number():
    if not LOG_PATH.exists():
        return 1

    matches = re.findall(r"^Test (\d+)$", LOG_PATH.read_text().strip(), re.MULTILINE)
    if not matches:
        return 1

    return max(int(match) for match in matches) + 1


# Purpose: Normalize test result details into text for the log file.
# Design: Accept strings, lists, or missing results so tests can report flexible details.
# Workflow: Return default text, pass strings through, or join iterable result lines.
def _format_results(results):
    if results is None:
        return "No additional results."

    if isinstance(results, str):
        return results

    return "\n".join(str(result) for result in results)


# Purpose: Write one structured pass or fail entry to the test log.
# Design: Include date, 24-hour time, status fields, details, and any exception summary.
# Workflow: Build the entry, ensure the log directory exists, append it, and print it.
def write_test_log(results=None, error=None):
    now = datetime.now()
    passed = error is None
    entry = (
        f"Test {_next_test_number()}\n"
        f"Date: {now:%m/%d/%Y}\n"
        f"Time: {now:%H:%M}\n"
        f"Passed: {'Yes' if passed else 'No'}\n"
        f"Failed: {'No' if passed else 'Yes'}\n\n"
        "Results:\n"
        f"{_format_results(results)}\n"
    )

    if error is not None:
        entry += f"Error: {type(error).__name__}: {error}\n"

    entry += "\n"

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as log_file:
        log_file.write(entry)

    print(entry, end="")


# Purpose: Execute a test body while guaranteeing that its result is logged.
# Design: Re-raise exceptions after logging so pytest still reports failures normally.
# Workflow: Run the callable, log success when it returns, or log failure before raising.
def run_logged_test(test_body):
    results = None
    try:
        results = test_body()
    except Exception as error:
        write_test_log(results, error)
        raise

    write_test_log(results)
