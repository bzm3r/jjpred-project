"""Utilities for profiling Python code."""

import cProfile
from collections.abc import Callable
import pstats
from io import StringIO
from typing import Any
from contextlib import contextmanager


def profile_function(
    func: Callable[..., Any], *args: Any, **kwargs: Any
) -> Any:
    """
    Profiles the execution time of a function using cProfile and then prints
    the profiling results.

    :param func: The function to be profiled.
    :param args: Positional arguments to pass to the function.
    :param kwargs: Keyword arguments to pass to the function.
    """
    # Create a profiler object
    pr = cProfile.Profile()

    # Enable profiling
    pr.enable()

    # Execute the function with the provided arguments
    result = func(*args, **kwargs)

    # Disable profiling
    pr.disable()

    # Create a StringIO object to capture the profiling output
    s = StringIO()

    # Create a Stats object and sort by cumulative time
    ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")

    # Print profiling results to the StringIO object
    ps.print_stats()

    # Output the profiling results
    print(s.getvalue())

    return result


@contextmanager
def profile_code(sort_by="cumulative", lines_to_print=None):
    """Profiles a block of code using cProfile and then prints the profiling
    results.

    Example:

    .. code-block: python
        with profile_code(sort_by="time", lines_to_print=10):
            # Some code you want to profile
            total = 0
            for i in range(10000):
                total += i
            print(total)
    """
    # Create a profiler object
    profiler = cProfile.Profile()
    profiler.enable()  # Start profiling
    yield  # This will allow the code inside the 'with' block to run
    profiler.disable()  # Stop profiling

    # Create a Stats object and print the statistics
    stats = pstats.Stats(profiler).sort_stats(sort_by)
    if lines_to_print:
        stats.print_stats(lines_to_print)
    else:
        stats.print_stats()
