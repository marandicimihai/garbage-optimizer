from __future__ import annotations

import sys
import subprocess
import webbrowser
from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def run_script(path: Path) -> None:
    print(f"Running: {path}")
    subprocess.run([sys.executable, str(path)], check=True)


def main() -> None:
    root = project_root()
    scripts = [
        root / "src" / "route-optimizer" / "pois.py",
        root / "src" / "route-optimizer" / "streets.py",
        root / "src" / "route-optimizer" / "bins.py",
        root / "src" / "route-optimizer" / "waste_data.py",
        root / "src" / "route-optimizer" / "trucks.py",
        root / "src" / "route-optimizer" / "visualization.py",
    ]

    for s in scripts:
        if not s.exists():
            print(f"Missing script: {s}")
            sys.exit(2)

    try:
        for s in scripts:
            run_script(s)
    except subprocess.CalledProcessError as exc:
        print(f"Script failed: {exc}")
        sys.exit(exc.returncode)

    html = root / "src" / "route-optimizer" / "generated" / "visualization.html"
    if html.exists():
        webbrowser.open_new_tab(html.as_uri())
        print(f"Opened: {html}")
    else:
        print(f"Missing output: {html}")


if __name__ == "__main__":
    main()
