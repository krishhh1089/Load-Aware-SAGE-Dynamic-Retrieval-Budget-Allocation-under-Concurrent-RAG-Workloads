import os

IGNORE = {
    "venv",
    ".venv",
    "__pycache__",
    ".git",
    "node_modules",
    ".idea",
    ".vscode"
}

for root, dirs, files in os.walk("."):
    dirs[:] = [d for d in dirs if d not in IGNORE]

    level = root.count(os.sep)
    indent = "    " * level

    print(f"{indent}{os.path.basename(root)}/")

    for file in files:
        print(f"{indent}    {file}")