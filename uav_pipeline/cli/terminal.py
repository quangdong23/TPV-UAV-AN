"""Restore the caller's POSIX terminal attributes; never consume keyboard input."""
from contextlib import contextmanager
import os
import sys


@contextmanager
def preserve_terminal():
    descriptor = None
    attributes = None
    if os.name == 'posix':
        try:
            import termios
            descriptor = os.dup(sys.stdin.fileno())
            if os.isatty(descriptor):
                attributes = termios.tcgetattr(descriptor)
        except (AttributeError, OSError, ValueError, termios.error):
            pass
    try:
        yield
    finally:
        if descriptor is not None:
            try:
                if attributes is not None:
                    # Do not flush characters the user has already typed.
                    termios.tcsetattr(descriptor, termios.TCSANOW, attributes)
            except (OSError, termios.error):
                pass  # The caller may have disconnected/closed the terminal.
            finally:
                os.close(descriptor)
