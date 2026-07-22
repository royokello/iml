import sys


_LOGO = """
██╗███╗   ███╗██╗
██║████╗ ████║██║
██║██╔████╔██║██║
██║██║╚██╔╝██║██║
██║██║ ╚═╝ ██║███████╗
╚═╝╚═╝     ╚═╝╚══════╝
""".strip()

def print_banner(module: str | None = None):
    print(f"\n{_LOGO}", file=sys.stderr)
    if module:
        print(f"{module.upper()}", file=sys.stderr)
    print(file=sys.stderr)
