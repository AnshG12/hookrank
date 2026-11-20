from __future__ import annotations

import os
import re
import sys


class Console:
    """Tiny ANSI console with graceful color fallback and basic rich-style markup."""
    enabled = sys.stdout.isatty() and os.getenv("NO_COLOR") is None
    styles = {
        "bold": "1", "dim": "2", "red": "31", "green": "32", "yellow": "33",
        "blue": "34", "magenta": "35", "cyan": "36", "bright_cyan": "96",
        "bright_blue": "94", "bright_magenta": "95", "bright_green": "92",
    }

    def print(self, value: object = "", *args: object, **kwargs: object) -> None:
        s = str(value)
        if self.enabled:
            def replace(match: re.Match[str]) -> str:
                tag = match.group(1)
                if tag.startswith("/"):
                    return "\033[0m"
                codes = [self.styles[k] for k in tag.split() if k in self.styles]
                return f"\033[{';'.join(codes)}m" if codes else ""
            s = re.sub(r"\[(/(?:[a-z_ ]*)|[a-z_ ]+)\]", replace, s)
        else:
            s = re.sub(r"\[/(?:[a-z_ ]*)\]|\[/?[a-z_ ]+\]", "", s)
        print(s, *args, **kwargs)


class Prompt:
    @staticmethod
    def ask(label: str, choices: list[str] | None = None, default: str | None = None) -> str:
        suffix = f" [{'/'.join(choices)}]" if choices else ""
        if default is not None:
            suffix += f" ({default})"
        while True:
            answer = input(f"{label}{suffix}: ").strip()
            answer = answer or (default or "")
            if not choices or answer in choices:
                return answer
            error(f"Choose one of: {', '.join(choices)}")


console = Console()


def banner() -> None:
    console.print("╭────────────────────────────────────────╮")
    console.print("│  [bold bright_cyan]HOOK[/][bold bright_magenta]RANK[/]  [dim]MICRODRAMA PROMO DISCOVERY[/]   │")
    console.print("╰────────────────────────────────────────╯")


def success(message: str) -> None:
    console.print(f"[bold green]✓[/] {message}")


def error(message: str) -> None:
    console.print(f"[bold red]✗[/] {message}")


def info(message: str) -> None:
    console.print(f"[bright_cyan]›[/] {message}")
