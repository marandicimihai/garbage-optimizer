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
    ]
    server_script = root / "src" / "route-optimizer" / "server.py"

    for s in scripts:
        if not s.exists():
            print(f"Missing script: {s}")
            sys.exit(2)
    if not server_script.exists():
        print(f"Missing script: {server_script}")
        sys.exit(2)

    try:
        for s in scripts:
            run_script(s)
    except subprocess.CalledProcessError as exc:
        print(f"Script failed: {exc}")
        sys.exit(exc.returncode)

    server_url = "http://127.0.0.1:5002"
    print(f"Starting visualization server: {server_url}")
    webbrowser.open_new_tab(server_url)
    try:
        subprocess.run([sys.executable, str(server_script)], check=True)
    except subprocess.CalledProcessError as exc:
        print(f"Server failed: {exc}")
        sys.exit(exc.returncode)


if __name__ == "__main__":
    main()
