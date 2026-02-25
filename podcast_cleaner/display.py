"""Shared Rich console for CLI output."""
from __future__ import annotations

import sys

from rich.console import Console

# Use plain text when not a TTY (piped/CI) or under test runner
_force_terminal = None
if not sys.stdout.isatty():
    _force_terminal = False

console = Console(force_terminal=_force_terminal)
