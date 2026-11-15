"""Promote APEX DISCARD v11 Dominator as canonical submission."""
import shutil, os, zipfile, hashlib

def sha256_of_file(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def main():
    src = "sparring_bots/bot_v11.py"
    assert os.path.exists(src), f"Source {src} does not exist"

    destinations = [
        "bot/bot.py",
        "Starter_Kit/bot.py",
        "dist/bot.py"
    ]

    for dst in destinations:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
        print(f"Copied {src} -> {dst}")

    # Build submission zip
    zip_path = "dist/bot_upload_v11_dominator.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write("dist/bot.py", arcname="bot.py")

    hash_val = sha256_of_file("dist/bot.py")
    zip_size = os.path.getsize(zip_path)
    file_size = os.path.getsize("dist/bot.py")

    print("\nPROMOTION COMPLETE:")
    print(f"  Canonical Entrypoint: bot/bot.py")
    print(f"  Submission File:     dist/bot.py ({file_size:,} bytes)")
    print(f"  Submission ZIP:      {zip_path} ({zip_size:,} bytes)")
    print(f"  SHA-256 Digest:      {hash_val}")

if __name__ == "__main__":
    main()
