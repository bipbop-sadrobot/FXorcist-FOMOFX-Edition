"""
Backward compatibility shim for fxorcist_cli.py
Redirects to the new package structure.
"""

from fxorcist.cli import main

if __name__ == '__main__':
    main()