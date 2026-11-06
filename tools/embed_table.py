import os

workspace = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
b64_path = os.path.join(workspace, "dist", "preflop_1755_b64.txt")

with open(b64_path, "r", encoding="utf-8") as f:
    b64 = f.read().strip()

targets = [
    os.path.join(workspace, "bot", "bot.py"),
    os.path.join(workspace, "Starter_Kit", "bot.py")
]

for path in targets:
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    target = 'EMBEDDED_PREFLOP_B64 = ""'
    assert target in content, f"Target not found in {path}"
    content = content.replace(target, f'EMBEDDED_PREFLOP_B64 = "{b64}"', 1)

    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"Successfully embedded preflop table into {path}")

