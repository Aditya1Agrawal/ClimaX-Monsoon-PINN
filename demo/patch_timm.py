"""timm 0.6.12 (required by ClimaX) imports maxxvit, which fails on Python >= 3.11; disable that one import."""
import importlib.util
import pathlib
import sys

init = pathlib.Path(importlib.util.find_spec("timm").origin).parent / "models" / "__init__.py"
txt = init.read_text()
if sys.version_info >= (3, 11) and "\nfrom .maxxvit import *" in txt:
    init.write_text(txt.replace("\nfrom .maxxvit import *", "\n# from .maxxvit import *  # disabled for py>=3.11"))
    print("patched", init)
else:
    print("no patch needed")
