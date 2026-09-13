"""
Practical 1 - Verify your real Python, Jupyter, and library setup is ready
for this course. Run this after installing everything below.
Run: python check_setup.py
"""
import importlib
import sys

REQUIRED_PACKAGES = ["numpy", "pandas", "matplotlib", "sklearn", "jupyter"]


def check_python_version():
    version = sys.version_info
    print(f"Real Python version: {version.major}.{version.minor}.{version.micro}")
    ok = version.major == 3 and version.minor >= 9
    print("  OK (3.9+)" if ok else "  WARNING: this course expects Python 3.9 or newer")
    return ok


def check_package(name):
    try:
        module = importlib.import_module(name)
        version = getattr(module, "__version__", "unknown version")
        print(f"  {name}: installed ({version})")
        return True
    except ImportError:
        print(f"  {name}: NOT installed")
        return False


if __name__ == "__main__":
    print("### CHECKING YOUR REAL PYTHON SETUP ###")
    python_ok = check_python_version()

    print("\n### CHECKING REQUIRED REAL LIBRARIES ###")
    results = {pkg: check_package(pkg) for pkg in REQUIRED_PACKAGES}

    all_ok = python_ok and all(results.values())
    print("\n### REAL RESULT ###")
    if all_ok:
        print("Everything is installed. You're ready to start Practical 2.")
    else:
        missing = [pkg for pkg, ok in results.items() if not ok]
        print(f"Missing packages: {missing}")
        print(f"Fix with: pip install {' '.join(missing)}")
