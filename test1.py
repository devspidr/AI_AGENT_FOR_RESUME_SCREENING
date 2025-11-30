# patch_fix_app_imports.py
from pathlib import Path

p = Path("app") / "app.py"
if not p.exists():
    raise SystemExit("app/app.py not found")

s = p.read_text(encoding="utf-8")

# remove the corrupted line if present
s = s.replace("from agentic from agentic import config as _config", "")

# canonical fixes
s = s.replace("import config as _config", "from agentic import config as _config")
s = s.replace("from config import TOP_K", "from agentic.config import TOP_K")
s = s.replace("from config import WEIGHT_EMBEDDING, WEIGHT_KEYWORD", "from agentic.config import WEIGHT_EMBEDDING, WEIGHT_KEYWORD")
s = s.replace("from config import", "from agentic.config import")
s = s.replace("from utils import clean_text", "from agentic.utils import clean_text")
s = s.replace("from agent import SimpleAgent", "from agentic.agent import SimpleAgent")
s = s.replace("from agentic.agent import SimpleAgent", "from agentic.agent import SimpleAgent")  # idempotent

# remove duplicate empty lines introduced accidentally
lines = s.splitlines()
out_lines = []
prev_blank = False
for L in lines:
    if L.strip() == "":
        if not prev_blank:
            out_lines.append(L)
        prev_blank = True
    else:
        out_lines.append(L)
        prev_blank = False

p.write_text("\n".join(out_lines), encoding="utf-8")
print("patched", p)
