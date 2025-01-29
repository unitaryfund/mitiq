import os

ROOT_DIR = "mitiq"


def find_manual_kwargs(filepath: str):
    with open(filepath, "r") as f:
        lines = f.readlines()

    results = []
    for i, line in enumerate(lines):
        if "kwargs:" in line:
            block = []
            for j in range(i + 1, len(lines)):
                if lines[j].strip().startswith("-"):
                    block.append(lines[j].strip())
                elif lines[j].strip() == "":
                    break
            if block:
                results.append((i, block))
    return results


def main():
    for subdir, _, files in os.walk(ROOT_DIR):
        for file in files:
            if file.endswith(".py"):
                filepath = os.path.join(subdir, file)
                matches = find_manual_kwargs(filepath)
                if matches:
                    print(f"File: {filepath}")
                    for match in matches:
                        print(f"Line {match[0]}: {match[1]}")


if __name__ == "__main__":
    main()
