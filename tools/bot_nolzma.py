import os, sys
sys.modules["lzma"] = None
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "bot"))
import bot
bot.main()
