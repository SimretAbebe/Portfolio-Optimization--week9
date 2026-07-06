# src/style_fix.py
"""Utility to safely apply a Matplotlib style.
If the requested style (e.g., 'seaborn-deep') is unavailable, it falls back to a
built‑in style that works with the current Matplotlib version.
"""
import matplotlib.pyplot as plt
import os


def _find_style_file(style_name: str) -> str | None:
    """Look for a style file in a few sensible locations and return its path.

    Search order:
    - If `style_name` is a path to an existing file, return it.
    - Workspace root (parent of `src/`) for `style_name.mplstyle` or `style_name`.
    - Current working directory for `style_name.mplstyle` or `style_name`.
    - Otherwise return None.
    """
    # 1) If it's already a valid path
    if os.path.isfile(style_name):
        return style_name

    candidates = []

    # 2) file next to the repository root (assume this file lives in src/)
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    candidates.append(os.path.join(repo_root, f"{style_name}.mplstyle"))
    candidates.append(os.path.join(repo_root, style_name))

    # 3) current working directory (useful for notebooks)
    cwd = os.getcwd()
    candidates.append(os.path.join(cwd, f"{style_name}.mplstyle"))
    candidates.append(os.path.join(cwd, style_name))

    for p in candidates:
        if os.path.isfile(p):
            return p
    return None


def apply_style(style_name: str = "seaborn-deep") -> None:
    """Apply a Matplotlib style, with a graceful fallback.

    Tries these options in order:
    - Use a built-in library style if available (e.g., 'seaborn').
    - Use a local `.mplstyle` file if present in the repo or notebook cwd.
    - Fall back to a safe built-in style.
    """
    # Prefer library styles when available
    if style_name in plt.style.available:
        plt.style.use(style_name)
        print(f"Style applied: {style_name} (library style)")
        return

    # Try to locate a .mplstyle file in repo root or current working dir
    style_path = _find_style_file(style_name)
    if style_path:
        # Quick heuristic validation: .mplstyle files should contain rcParam lines like 'key: value'.
        try:
            with open(style_path, 'r', encoding='utf-8') as f:
                contents = f.read()
            if any(line.strip().startswith('import ') or 'sns.' in line or line.strip().startswith('def ') for line in contents.splitlines()):
                # clearly not a valid .mplstyle file
                print(f"Found style file at {style_path}, but it looks invalid for .mplstyle format. Skipping.")
            else:
                try:
                    plt.style.use(style_path)
                    print(f"Style applied from file: {style_path}")
                    return
                except OSError:
                    # fall through to fallback
                    pass
        except Exception:
            # If reading fails for any reason, ignore and fall back
            pass

    # Final fallback: use matplotlib's default style which is always available
    fallback = "default"
    try:
        plt.style.use(fallback)
        print(f"'{style_name}' not found - using fallback style: '{fallback}'.")
    except Exception:
        # As a last resort, attempt 'classic'
        try:
            plt.style.use('classic')
            print(f"'{style_name}' and fallback '{fallback}' failed — using 'classic'.")
        except Exception:
            # If even that fails, silently continue
            pass
