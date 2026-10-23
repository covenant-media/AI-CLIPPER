import json
import subprocess
import sys
import time
from pathlib import Path
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel

# Windows consoles are usually cp1252. Provider error messages routinely carry
# non-Latin text (e.g. Chinese relay errors such as
# "当前分组 default 下对于模型 ... 无可用渠道") and rich's legacy Windows renderer
# used to raise UnicodeEncodeError *while reporting an unrelated failure*,
# killing the whole run. Swapping the stream to errors="replace" keeps the
# console encoding intact but never raises on an unrepresentable character.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(errors="replace")
    except Exception:
        pass

console = Console()


def _safe(text) -> str:
    """Guarantee `text` can be printed on the current terminal.

    Belt-and-braces on top of the reconfigure above: if a stream still cannot
    represent a character (odd code page, redirected pipe, ...) it is replaced
    instead of raising.
    """
    text = str(text)
    for stream in (sys.stdout, sys.stderr):
        encoding = getattr(stream, "encoding", None)
        if not encoding:
            continue
        try:
            text.encode(encoding)
        except (UnicodeEncodeError, LookupError):
            text = text.encode(encoding, "replace").decode(encoding, "replace")
    return text


class _Log:
    """Console logger that can never take the pipeline down with it.

    Messages routinely embed raw provider JSON, which may contain terminal-
    hostile sequences: non-encodable characters, or bracket pairs such as
    "[/]" that rich would try to parse as markup and reject with MarkupError.
    Both are neutralised here.
    """

    def info(self, m):    console.print(f"[cyan]•[/cyan] {escape(_safe(m))}")
    def ok(self, m):      console.print(f"[green]✓[/green] {escape(_safe(m))}")
    def warn(self, m):    console.print(f"[yellow]![/yellow] {escape(_safe(m))}")
    def error(self, m):   console.print(f"[red]✗[/red] {escape(_safe(m))}")
    def step(self, m):    console.print(Panel.fit(escape(_safe(m)), style="bold"))


log = _Log()


def run(cmd, check=True, capture=True):
    res = subprocess.run(
        cmd,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
        text=True,
    )
    if check and res.returncode != 0:
        raise RuntimeError(
            f"Command failed: {' '.join(map(str, cmd))}\n{res.stderr}"
        )
    return res.returncode, res.stdout, res.stderr


def ffprobe_duration(path: Path) -> float:
    _, out, _ = run([
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "json", str(path),
    ])
    return float(json.loads(out)["format"]["duration"])


def write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def read_json(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def fmt_tc(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


class Timer:
    def __init__(self, label):
        self.label = label
    def __enter__(self):
        self.t = time.time()
        log.info(f"{self.label}...")
        return self
    def __exit__(self, *a):
        log.ok(f"{self.label} done in {time.time() - self.t:.1f}s")
