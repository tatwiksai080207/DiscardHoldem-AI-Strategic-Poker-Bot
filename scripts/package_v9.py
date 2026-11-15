import zipfile
import hashlib
import os

src = "bot/bot_v9.py"
dst = "dist/bot_upload_v9_supreme.zip"

with zipfile.ZipFile(dst, "w", compression=zipfile.ZIP_DEFLATED) as z:
    z.write(src, arcname="bot.py")

src_data = open(src, "rb").read()
with zipfile.ZipFile(dst, "r") as z:
    zip_data = z.read("bot.py")
    assert src_data == zip_data, "Zip content mismatch!"

print(f"Created {dst}: size={os.path.getsize(dst)} bytes, sha256={hashlib.sha256(open(dst, 'rb').read()).hexdigest()}")
print(f"Contained bot.py size={len(zip_data)} bytes, sha256={hashlib.sha256(zip_data).hexdigest()}")
