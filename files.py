"""Files JARVIS is allowed to touch - create, write, read, search, move and *undo*.

Why it is built this way
------------------------
"Make it do things: create files, delete files" is the request; an assistant that can
delete anything on a laptop is the risk.  So every path goes through :func:`resolve_path`,
which confines writes to the roots the user granted (default ``Documents\\JARVIS``, created on
first use), and every destructive action is journal-first:

* before a delete, the file is copied into ``data/file_trash/<timestamp>/`` and only then sent
  to the Recycle Bin through :func:`winops.recycle`, so :func:`undo` can bring it back even if
  the user empties the bin;
* before a write/overwrite, the previous bytes are saved in ``data/file_backups/`` and the
  journal records the pointer, so "undo that" and "what did you just change?" have answers;
* a path *outside* the roots is never silently touched: the caller gets
  ``needs_confirmation`` plus a token, and the same call with ``confirm="yes"`` proceeds once.

Reading is generous (notes, logs, code, ``.docx`` via its zip+XML shape), writing is strict.
No third-party dependency is used, and nothing here raises - each function returns
``{ok, message, ...}`` for the model to speak.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sys
import time
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import config
import winops
from config import SETTINGS, get_logger

log = get_logger("files")

MAX_READ_CHARS = 24_000
MAX_LIST = 200
SKIP_DIRS = {"node_modules", "__pycache__", ".git", ".venv", "venv", ".cache", "appdata",
             "$recycle.bin", "program files", "program files (x86)", "windows", "system32",
             ".mypy_cache", "dist", "build", ".next", ".nuget", "downloads"}
TEXT_SUFFIXES = {".txt", ".md", ".py", ".js", ".ts", ".json", ".yaml", ".yml", ".toml", ".ini",
                 ".cfg", ".csv", ".log", ".html", ".css", ".bat", ".ps1", ".sh", ".c", ".h",
                 ".cpp", ".java", ".go", ".rs", ".rb", ".php", ".sql", ".srt", ".vtt", ".env"}
BACKUP_DIR = Path("data/file_backups")
TRASH_DIR = Path("data/file_trash")
JOURNAL = Path("data/file_journal.jsonl")

_SAFE_NAME = re.compile(r"[^\w.\- ()#&'+]+")


def _root(path: Path) -> Path:
    return config.ROOT / path if not path.is_absolute() else path


# --------------------------------------------------------------------------- roots
def roots() -> List[Path]:
    """Where JARVIS may write.  ``FILES_ROOT`` overrides the default; ``FILES_ALLOWED`` adds more."""
    out: List[Path] = []
    raw = (SETTINGS.files_root or "").strip()
    if raw:
        for chunk in re.split(r"[;,]+", raw):
            chunk = chunk.strip().strip('"')
            if chunk:
                candidate = Path(os.path.expandvars(os.path.expanduser(chunk)))
                if str(candidate) not in [str(p) for p in out]:
                    out.append(candidate)
    if not out:
        documents = Path.home() / "Documents"
        if not documents.is_dir():  # non-Windows dev box, or Documents redirected/absent
            documents = Path.home()
        out.append(documents / "JARVIS")
    extra = (SETTINGS.files_allowed or "").strip()
    for chunk in re.split(r"[;,]+", extra):
        chunk = chunk.strip().strip('"')
        if chunk:
            candidate = Path(os.path.expandvars(os.path.expanduser(chunk)))
            if str(candidate) not in [str(p) for p in out]:
                out.append(candidate)
    for folder in out:
        try:
            folder.mkdir(parents=True, exist_ok=True)
        except OSError as exc:  # read-only drive, OneDrive hiccup
            log.debug("root %s not creatable: %s", folder, exc)
    return out


def _clean_name(name: str, fallback: str = "note") -> str:
    stem = _SAFE_NAME.sub("_", (name or "").strip()).strip("._ ")[:80]
    return stem or fallback


def _result(ok: bool, message: str, **extra: Any) -> Dict[str, Any]:
    out: Dict[str, Any] = {"ok": bool(ok), "message": message}
    out.update(extra)
    return out


def _journal(entry: Dict[str, Any]) -> None:
    path = _root(JOURNAL)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"at": int(time.time()), **entry}, ensure_ascii=False, default=str) + "\n")
    except OSError as exc:  # pragma: no cover
        log.debug("file journal unavailable: %s", exc)


def journal(limit: int = 12) -> List[Dict[str, Any]]:
    path = _root(JOURNAL)
    if not path.is_file():
        return []
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines()[-limit:]:
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


# --------------------------------------------------------------------------- path resolution
@dataclass
class Located:
    """A path JARVIS is allowed to act on, plus the facts the model needs."""

    path: Path
    root: Optional[Path]
    inside: bool
    label: str
    note: str = ""

    @property
    def exists(self) -> bool:
        return self.path.exists()


def resolve_path(raw: str, for_write: bool = False, must_exist: bool = False) -> Tuple[Optional[Located], str]:
    """Turn "my notes/todo.md", "todo.md" or an absolute path into a guarded :class:`Located`.

    Bare names land in the first granted root, so "create a file called shopping.md" works
    without the user ever thinking about folders.  Anything that escapes the roots is refused
    for writes and marked ``outside`` for reads (the caller then asks before continuing).
    """
    text = (raw or "").strip().strip('"').strip("'")
    if not text:
        return None, "No file was named."
    if len(text) > 260 or "\x00" in text:
        return None, "That path is too long to be a real file."
    if re.match(r"^[a-zA-Z]:[\\/]", text) and not winops.IS_WINDOWS:
        return None, "That is a Windows path and JARVIS is not on Windows right now."
    if "\\" in text and not winops.IS_WINDOWS and "/" not in text:
        return None, (f"“{text[:40]}” looks like a Windows path; use forward slashes here.")
    if re.search(r"[<>*?|\n\r]", text) and not text.lower().startswith(("http", "file:")):
        return None, f"“{text[:40]}” contains characters Windows will not accept in a name."
    candidate = Path(os.path.expandvars(os.path.expanduser(text)))
    if not candidate.is_absolute():
        # "notes/x.md" relative to the repo is how the existing notes tool works, so honour it.
        repo_guess = config.ROOT / candidate
        base = roots()[0] if roots() else config.ROOT
        if for_write or not repo_guess.exists():
            candidate = base / candidate
        else:
            candidate = repo_guess
    try:
        resolved = candidate.resolve()
    except OSError:
        resolved = candidate
    matched: Optional[Path] = None
    outside = False
    for root in roots():
        try:
            r_root = root.resolve()
        except OSError:
            r_root = root
        if resolved == r_root or r_root in resolved.parents:
            matched = r_root
            break
        # a path under a granted root that was written with ".." in it still resolves inside
        if for_write and str(resolved).lower().startswith(str(r_root).lower()[:8]):
            outside = True
    if for_write and matched is None:
        inside_home = str(resolved).lower().startswith(str(Path.home()).lower())
        if not inside_home:
            return None, (f"I only write inside {', '.join(str(r) for r in roots())}. "
                          "Add FILES_ALLOWED=C:\\Users\\you\\Desktop to .env to give me another folder.")
        outside = True
    label = resolved.name or str(resolved)
    located = Located(path=resolved, root=matched, inside=matched is not None, label=label,
                      note=("outside my folders" if outside else ""))
    if must_exist and not resolved.exists():
        near = suggest_names(resolved.parent, resolved.stem)
        return None, (f"There is no {label} in {resolved.parent}."
                      + (f" Closest: {', '.join(near)}." if near else ""))
    return located, ""


def suggest_names(folder: Path, stem: str, limit: int = 3) -> List[str]:
    try:
        names = [p.name for p in folder.iterdir() if p.is_file()]
    except OSError:
        return []
    stem_l = stem.lower()
    scored = sorted(names, key=lambda n: (stem_l not in n.lower(), -_similar(stem_l, n.lower())))
    return [n for n in scored[:limit] if _similar(stem_l, n.lower()) > 0.3 or stem_l in n.lower()]


def _similar(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    common = sum(1 for chunk in re.findall(r"\w{3}", a) if chunk in b)
    total = max(1, len(re.findall(r"\w{3}", a)))
    ratio = common / total
    return max(ratio, 1.0 - (abs(len(a) - len(b)) / max(len(a), len(b))))


# --------------------------------------------------------------------------- guards
def _confirm_token(located: Located, action: str) -> str:
    seed = f"{action}|{located.path}|{time.strftime('%Y%m%d%H')}"
    return hashlib.sha1(seed.encode("utf-8")).hexdigest()[:8]


def _needs_confirmation(located: Located, action: str) -> Dict[str, Any]:
    #: Both answers name the fix.  The confirm branch used to say only "say confirm", which left
     #: the user re-asking every day about a folder they had decided to allow.
    return _result(False, f"Say “confirm {action}” and I will do it - {located.path} is outside the "
                          f"folders I normally touch.  Add FILES_ALLOWED={located.path.parent} to .env to "
                          f"work there without being asked.",
                   needs_confirmation=True, action=action, path=str(located.path),
                   confirm_token=_confirm_token(located, action))


def _backup(located: Located, kind: str = "write") -> Optional[Path]:
    """Copy the current bytes aside before changing them, so undo is always possible."""
    if not located.path.is_file():
        return None
    folder = _root(BACKUP_DIR if kind == "write" else TRASH_DIR) / time.strftime("%Y%m%d-%H%M%S")
    try:
        folder.mkdir(parents=True, exist_ok=True)
        target = folder / located.path.name
        #: Two edits inside the same second used to share one backup filename, so the second copy
        #: clobbered the first and "undo that" could only reach back a single step.
        if target.exists():
            stem, suffix = target.stem, target.suffix
            step = 2
            while (folder / f"{stem}-{step}{suffix}").exists():
                step += 1
            target = folder / f"{stem}-{step}{suffix}"
        shutil.copy2(located.path, target)
        return target
    except OSError as exc:  # pragma: no cover - disk full / locked file
        log.debug("backup of %s failed: %s", located.path, exc)
        return None


# --------------------------------------------------------------------------- the actions
def read(path: str, limit: int = MAX_READ_CHARS, offset: int = 0) -> Dict[str, Any]:
    located, error = resolve_path(path, must_exist=True)
    if located is None:
        return _result(False, error)
    if located.path.is_dir():
        return list_dir(str(located.path))
    try:
        size = located.path.stat().st_size
    except OSError as exc:
        return _result(False, f"I can't stat {located.label}: {exc}")
    suffix = located.path.suffix.lower()
    if suffix in {".docx", ".pptx", ".xlsx"}:
        return _read_office(located, size)
    if size > 6_000_000:
        return _result(False, f"{located.label} is {size / 1e6:.1f} MB - too big to read aloud. "
                             "Ask me for the first part, or open it yourself.")
    try:
        raw = located.path.read_bytes()
    except OSError as exc:
        return _result(False, f"I couldn't open {located.label}: {exc}")
    if b"\x00" in raw[:4096]:
        return _result(True, f"{located.label} is a binary file ({size} bytes), not text.",
                       binary=True, size=size, path=str(located.path))
    text = raw.decode("utf-8", errors="replace")
    start = max(0, int(offset or 0))
    window = text[start:start + max(200, int(limit or MAX_READ_CHARS))]
    truncated = start + len(window) < len(text)
    head = f"From {located.label}" + (f" (chars {start + 1}-{start + len(window)} of {len(text)})" if truncated else "")
    return _result(True, "\n".join([head, "", window]).strip(), path=str(located.path),
                   characters=len(text), truncated=truncated, lines=text.count("\n") + 1,
                   modified=datetime.fromtimestamp(located.path.stat().st_mtime).strftime("%d %b %H:%M"))


def _read_office(located: Located, size: int) -> Dict[str, Any]:
    """``.docx``/``.pptx`` are zipped XML, so text extraction needs no Word and no deps."""
    member = {"docx": "word/document.xml", "pptx": None, "xlsx": "xl/sharedStrings.xml"}[located.path.suffix[1:]]
    try:
        with zipfile.ZipFile(located.path) as archive:
            names = archive.namelist()
            if member and member in names:
                blob = archive.read(member).decode("utf-8", errors="ignore")
            else:
                candidates = [n for n in names if re.search(r"(document|slide\d+|sharedStrings)\.xml$", n)]
                if not candidates:
                    return _result(False, f"{located.label} has no readable text part.")
                blob = "".join(archive.read(n).decode("utf-8", errors="ignore") for n in candidates[:8])
    except (OSError, zipfile.BadZipFile) as exc:
        return _result(False, f"{located.label} is not a readable Office file: {exc}")
    text = re.sub(r"<a:p>|</w:p>|</a:p>", "\n", blob)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not text:
        return _result(True, f"{located.label} has no text in it (it may be scanned images).",
                       path=str(located.path))
    clipped = text[:MAX_READ_CHARS]
    return _result(True, f"{located.label}: {clipped}", path=str(located.path), characters=len(text),
                   truncated=len(text) > len(clipped), kind=located.path.suffix[1:])


def write(path: str, content: str = "", mode: str = "create", confirm: str = "") -> Dict[str, Any]:
    """Create or change a text file.  ``mode`` = create | overwrite | append.

    ``create`` refuses to clobber, which is what makes "make me a file called X" safe to say
    twice; the caller (or the user) has to say overwrite or append explicitly, and either way
    the previous bytes go to ``data/file_backups`` first so :func:`undo` can put them back.
    """
    located, error = resolve_path(path, for_write=True)
    if located is None:
        return _result(False, error)
    if not located.inside and (confirm or "").strip().lower() not in {"yes", "y", "confirm", "true", "1"}:
        return _needs_confirmation(located, "write there")
    mode = (mode or "create").lower().strip()
    if mode in {"new", "create", ""}:
        mode = "create"
    if mode not in {"create", "overwrite", "append"}:
        return _result(False, f"“{mode}” is not a mode I know: create, overwrite or append.")
    exists = located.path.exists()
    if mode == "create" and exists:
        return _result(False, f"{located.label} already exists. Say “overwrite {located.label}” to "
                              f"replace it or “append to {located.label}” to add to the end.",
                       exists=True, path=str(located.path))
    if mode == "overwrite" and not exists:
        mode = "create"
    text = content if content is not None else ""
    saved = _backup(located, "write") if exists else None
    try:
        located.path.parent.mkdir(parents=True, exist_ok=True)
        if mode == "append" and exists and located.path.stat().st_size:
            with located.path.open("a", encoding="utf-8") as handle:
                handle.write(text if text.startswith("\n") else "\n" + text)
        else:
            if text and not text.endswith("\n") and located.path.suffix.lower() not in {".html", ".json"}:
                text += "\n"
            with located.path.open("w", encoding="utf-8") as handle:
                handle.write(text)
    except OSError as exc:
        return _result(False, f"I couldn't write {located.label}: {exc}")
    words = len(re.findall(r"\S+", text))
    _journal({"action": mode, "path": str(located.path), "backup": str(saved) if saved else "",
              "bytes": len(text.encode("utf-8")), "words": words})
    verb = "Appended to" if mode == "append" else ("Overwrote" if saved else "Created")
    return _result(True, f"{verb} {located.label} - {located.path.stat().st_size} bytes, {words} words, "
                        f"in {located.path.parent}.", path=str(located.path), characters=len(text),
                   words=words, overwritten=bool(saved), backup=str(saved) if saved else "",
                   lines=text.count("\n") + 1)


def note(title: str, text: str = "") -> Dict[str, Any]:
    """Quick dated note in Markdown - the "note that down" path, kept next to the AI files."""
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    name = _SAFE_NAME.sub("-", (title or "note").strip()).strip("-_ ")[:48] or "note"
    folder = roots()[0] / "notes"
    path = folder / f"{datetime.now().strftime('%Y%m%d')}-{name.lower().replace(' ', '-')}.md"
    body = f"# {title or 'Note'}\n\n_{stamp}_\n\n{text or ''}\n"
    try:
        folder.mkdir(parents=True, exist_ok=True)
        path.write_text(body, encoding="utf-8")
    except OSError as exc:
        return _result(False, f"Could not save the note: {exc}")
    _journal({"action": "note", "path": str(path), "bytes": len(body)})
    return _result(True, f"Noted: {title or 'note'} saved to {path.name}.", path=str(path))


def delete(path: str, confirm: str = "", permanent: str = "") -> Dict[str, Any]:
    """Move to the Recycle Bin (and keep a private copy), so "undo that" always works."""
    located, error = resolve_path(path)
    if located is None:
        return _result(False, error)
    if not located.exists:
        near = suggest_names(located.path.parent, located.path.stem)
        return _result(False, f"There's no {located.label} in {located.path.parent}."
                              + (f" Did you mean {', '.join(near)}?" if near else ""))
    allowed = (confirm or "").strip().lower() in {"yes", "y", "confirm", "delete it", "1"}
    if not located.inside and not allowed:
        return _needs_confirmation(located, "delete")
    policy = (SETTINGS.file_delete_policy or "recycle").lower()
    if policy == "refuse":
        return _result(False, "FILE_DELETE_POLICY=refuse, so I will not delete anything. "
                              "Say it again with FILE_DELETE_POLICY=recycle in .env if you want me to.")
    if located.path.is_dir() and policy != "trash_only":
        return _result(False, f"{located.label} is a folder. Say “empty the folder {located.label}” "
                              "if you really mean everything inside it, or delete files one at a time.")
    kept = _backup(located, "trash")
    if permanent.strip().lower() in {"yes", "y", "forever", "shred"} and policy == "direct":
        try:
            shutil.rmtree(located.path) if located.path.is_dir() else located.path.unlink()
            _journal({"action": "delete-permanent", "path": str(located.path), "backup": ""})
            return _result(True, f"{located.label} deleted permanently.", path=str(located.path))
        except OSError as exc:
            return _result(False, f"Could not delete {located.label}: {exc}")
    outcome = winops.recycle([str(located.path)])
    if not outcome.get("ok"):
        # Non-Windows (or a refused shell call): the private copy still exists, so removal is safe.
        try:
            located.path.unlink() if located.path.is_file() else located.path.rmdir()
            outcome = {"ok": True, "message": f"{located.label} removed."}
        except OSError as exc:
            return _result(False, f"Windows refused to delete {located.label}: {exc}")
    _journal({"action": "delete", "path": str(located.path), "backup": str(kept) if kept else "",
              "recycled": "Recycle Bin" in str(outcome.get("message", ""))})
    return _result(True, f"{located.label} is gone" + (" (Recycle Bin, and I kept a copy you can "
                        "restore). " if kept else ". ") + f"Backup: {kept or 'none'}",
                   path=str(located.path), backup=str(kept) if kept else "", can_undo=bool(kept))


def undo(steps: int = 1) -> Dict[str, Any]:
    """Replay the journal backwards: restore deleted files, roll back writes."""
    entries = journal(limit=max(1, min(int(steps or 1), 10)) * 3)
    actions = [e for e in reversed(entries) if e.get("action") in {"delete", "write", "overwrite",
                                                                  "append", "move", "rename"}]
    restored: List[str] = []
    for entry in actions[:max(1, min(int(steps or 1), 10))]:
        backup, target = entry.get("backup") or "", entry.get("path") or ""
        if not backup or not Path(backup).is_file() or not target:
            continue
        try:
            Path(target).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(backup, target)
            restored.append(Path(target).name)
        except OSError as exc:
            return _result(False, f"Restoring {Path(target).name} failed: {exc}")
    if not restored:
        return _result(False, "There is nothing in my journal to undo - I have not changed any files "
                              "since I started, or the backup was outside my folders.")
    _journal({"action": "undo", "paths": restored})
    return _result(True, f"Restored {', '.join(restored)}.", files=restored)


def list_dir(path: str = "", pattern: str = "*", limit: int = MAX_LIST, sort: str = "name") -> Dict[str, Any]:
    if not (path or "").strip():
        folder = roots()[0]
    else:
        located, error = resolve_path(path, must_exist=False)
        folder = located.path if located else (roots()[0] if roots() else config.ROOT)
    if not folder.is_dir():
        return _result(False, f"No folder at {folder}.")
    try:
        entries = [p for p in folder.glob(pattern or "*") if not p.name.startswith(".")]
    except (OSError, re.error):
        return _result(False, f"“{pattern}” is not a pattern I can use.")
    rows: List[Dict[str, Any]] = []
    for item in entries:
        try:
            stat = item.stat()
            rows.append({"name": item.name + ("/" if item.is_dir() else ""), "path": str(item),
                         "bytes": 0 if item.is_dir() else stat.st_size,
                         "modified": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M"),
                         "kind": "folder" if item.is_dir() else item.suffix.lstrip(".").lower() or "file"})
        except OSError:
            continue
    key = {"size": lambda r: -r["bytes"], "new": lambda r: r["modified"], "name": lambda r: r["name"].lower()}.get(
        sort, lambda r: r["name"].lower())
    rows.sort(key=key)
    rows = rows[:max(1, min(int(limit or MAX_LIST), MAX_LIST))]
    if not rows:
        return _result(True, f"{folder} is empty" + (f" for “{pattern}”." if pattern != "*" else "."),
                       path=str(folder), entries=[])
    lines = [f"{r['name']}  ({r['bytes']} bytes, {r['modified']})" if r["kind"] != "folder"
             else f"{r['name']}" for r in rows]
    return _result(True, f"{len(rows)} item(s) in {folder}:\n" + "\n".join(lines),
                   path=str(folder), entries=rows[:40], count=len(rows))


def search(name: str = "", contains: str = "", root: str = "", limit: int = 25) -> Dict[str, Any]:
    """Find files by name (glob) or by a phrase inside them (text files only)."""
    bases = [Path(os.path.expandvars(os.path.expanduser(root)))] if root else roots()
    hits: List[Dict[str, Any]] = []
    seen: set = set()
    pattern = (name or "*").strip() or "*"
    if not pattern.startswith(("*", "?")) and not any(ch in pattern for ch in "*?"):
        pattern = f"*{pattern}*"
    for base in bases:
        if not base.is_dir():
            continue
        if name and not contains:
            for found in base.rglob(pattern):
                if len(hits) >= limit:
                    break
                if any(part.lower() in SKIP_DIRS for part in found.relative_to(base).parts if part != found.name):
                    continue
                if found.is_file() and str(found) not in seen:
                    seen.add(str(found))
                    hits.append({"name": found.name, "path": str(found),
                                 "bytes": found.stat().st_size if found.exists() else 0,
                                 "modified": datetime.fromtimestamp(found.stat().st_mtime).strftime("%Y-%m-%d %H:%M")})
        elif contains:
            needle = contains.lower()
            for candidate in base.rglob("*"):
                if len(hits) >= limit:
                    break
                if not candidate.is_file() or candidate.suffix.lower() not in TEXT_SUFFIXES:
                    continue
                if any(part.lower() in SKIP_DIRS for part in candidate.parts):
                    continue
                try:
                    if candidate.stat().st_size > 2_000_000:
                        continue
                    text = candidate.read_text(encoding="utf-8", errors="ignore")
                except OSError:
                    continue
                if needle in text.lower():
                    index = text.lower().find(needle)
                    snippet = re.sub(r"\s+", " ", text[max(0, index - 70):index + 130]).strip()
                    seen.add(str(candidate))
                    hits.append({"name": candidate.name, "path": str(candidate), "match": snippet,
                                 "line": text[:index].count("\n") + 1})
    if not hits:
        return _result(True, f"Nothing matching “{name or contains}” in {', '.join(str(b) for b in bases)}.",
                       results=[], count=0)
    detail = "\n".join(f"{h.get('name')}" + (f" — {h['match'][:110]}" if h.get("match") else "") for h in hits[:12])
    return _result(True, f"{len(hits)} match(es) for “{name or contains}”:\n{detail}",
                   results=hits[:limit], count=len(hits))


def move(path: str, destination: str = "", copy: str = "", rename: str = "") -> Dict[str, Any]:
    """Move/copy a file, or rename it in place."""
    located, error = resolve_path(path)
    if located is None:
        return _result(False, error)
    if not located.exists:
        return _result(False, f"There is no {located.label} to move.")
    target_raw = (rename or destination or "").strip()
    if not target_raw:
        return _result(False, "Where should I put it? e.g. move notes.md to Desktop.")
    target_loc, target_error = resolve_path(target_raw, for_write=True)
    if target_loc is None:
        return _result(False, target_error)
    if target_loc.path.is_dir() and not rename:
        target_loc.path = target_loc.path / located.path.name
    saving = copy.strip().lower() in {"yes", "y", "true", "1"}
    saved = _backup(target_loc, "write") if target_loc.path.exists() else None
    try:
        target_loc.path.parent.mkdir(parents=True, exist_ok=True)
        if saving:
            (shutil.copytree if located.path.is_dir() else shutil.copy2)(located.path, target_loc.path)
        else:
            shutil.move(str(located.path), str(target_loc.path))
    except (OSError, shutil.Error) as exc:
        return _result(False, f"Could not {'copy' if saving else 'move'} {located.label}: {exc}")
    _journal({"action": "copy" if saving else ("move" if destination else "rename"),
              "path": str(located.path), "to": str(target_loc.path), "backup": str(saved) if saved else ""})
    return _result(True, f"{'Copied' if saving else 'Moved'} {located.label} to {target_loc.path}.",
                   path=str(target_loc.path), from_=str(located.path), copied=saving)


def mkdir(path: str) -> Dict[str, Any]:
    located, error = resolve_path(path, for_write=True)
    if located is None:
        return _result(False, error)
    try:
        located.path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return _result(False, f"Could not create {located.label}: {exc}")
    _journal({"action": "mkdir", "path": str(located.path)})
    return _result(True, f"Folder ready: {located.path}.", path=str(located.path))


def disk_report(path: str = "") -> Dict[str, Any]:
    located, _error = resolve_path(path or ".", must_exist=False)
    folder = located.path if located else (roots()[0] if roots() else config.ROOT)
    try:
        usage = shutil.disk_usage(folder if folder.is_dir() else folder.parent)
    except OSError as exc:
        return _result(False, f"I can't read that drive: {exc}")
    biggest: List[Tuple[int, str]] = []
    count = 0
    total = 0
    if folder.is_dir():
        for candidate in folder.rglob("*"):
            try:
                if candidate.is_file():
                    size = candidate.stat().st_size
                    count += 1
                    total += size
                    biggest.append((size, str(candidate)))
            except OSError:
                continue
    biggest.sort(reverse=True)
    gb = 1024 ** 3
    message = (f"{folder} has {count} files using {total / gb:.2f} GB. "
               f"Drive free space: {usage.free / gb:.1f} GB of {usage.total / gb:.1f} GB.")
    if biggest:
        message += " Largest: " + ", ".join(f"{Path(p).name} ({s / 1e6:.0f} MB)" for s, p in biggest[:3]) + "."
    return _result(True, message, path=str(folder), files=count, bytes=total,
                   free_gb=round(usage.free / gb, 2), total_gb=round(usage.total / gb, 2),
                   biggest=[{"path": p, "bytes": s} for s, p in biggest[:10]])


def open_file(path: str, select: str = "") -> Dict[str, Any]:
    """Hand it to Windows: open the file, or show it in Explorer."""
    located, error = resolve_path(path)
    if located is None:
        return _result(False, error)
    if not located.exists:
        return _result(False, f"{located.label} does not exist yet.")
    if select.strip().lower() in {"yes", "y", "true", "show", "1"}:
        return winops.open_folder(str(located.path.parent), select=located.path.name)
    return winops.shell_execute(str(located.path))


def script(name: str, language: str = "python", code: str = "", run: str = "") -> Dict[str, Any]:
    """Write a runnable script and (on request) execute it - the "do crazy things" path."""
    language = (language or "python").lower()
    extensions = {"python": ".py", "py": ".py", "powershell": ".ps1", "ps": ".ps1", "batch": ".bat",
                  "bat": ".bat", "javascript": ".js", "js": ".js", "shell": ".sh", "sh": ".sh",
                  "html": ".html", "markdown": ".md", "sql": ".sql"}
    suffix = extensions.get(language, f".{language[:5]}")
    filename = _SAFE_NAME.sub("_", (name or "script").strip()).strip("._ ")[:60] or "script"
    if not filename.lower().endswith(suffix):
        filename += suffix
    folder = roots()[0] / "scripts"
    try:
        folder.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return _result(False, f"Could not create the scripts folder: {exc}")
    path = folder / filename
    shebang = {"python": "#!/usr/bin/env python3\n", "powershell": "#Requires -Version 5\n",
               "shell": "#!/bin/sh\n"}.get(language, "")
    body = (shebang if not code.startswith("#!") else "") + (code or "")
    try:
        path.write_text(body, encoding="utf-8")
    except OSError as exc:
        return _result(False, f"Could not write {filename}: {exc}")
    runner = (f".venv\\Scripts\\python.exe {path.name}" if language == "python"
              else f"the {language} interpreter")
    result = _result(True, f"{filename} saved in {folder}. Run it with {runner}.",
                     path=str(path), language=language, characters=len(body), command=runner)
    if (run or "").strip().lower() in {"yes", "y", "run it", "now", "1"}:
        out = execute_script(str(path), language)
        result["execution"] = out
        result["message"] += " " + out.get("message", "")
    return result


def execute_script(path: str, language: str = "python", timeout: int = 60) -> Dict[str, Any]:
    """Run a script JARVIS just wrote (or that already exists), capturing its output."""
    located, error = resolve_path(path)
    if located is None:
        return _result(False, error)
    if not located.path.is_file():
        return _result(False, f"No script at {located.path}.")
    suffix = located.path.suffix.lower()
    if suffix == ".py":
        cmd = [sys.executable, str(located.path)]
    elif suffix == ".ps1":
        cmd = [shutil.which("powershell") or "powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
               "-File", str(located.path)]
    elif suffix in {".bat", ".cmd"}:
        cmd = ["cmd", "/c", str(located.path)]
    elif suffix in {".sh", ".bash"}:
        cmd = ["sh", str(located.path)]
    else:
        return _result(False, f"I don't know how to run a {suffix or 'unknown'} file.")
    code, out, err = winops.run_quiet(cmd, timeout=max(5, min(int(timeout or 60), 600)))
    text = "\n".join(x for x in (out, err) if x)[:4000]
    _journal({"action": "run", "path": str(located.path), "exit": code})
    if code == 0:
        return _result(True, f"{located.label} ran cleanly." + (f" Output: {text[:1200]}" if text else ""),
                       exit_code=code, output=text, path=str(located.path))
    return _result(False, f"{located.label} exited with code {code}. {text[:1200]}", exit_code=code,
                   output=text, path=str(located.path))


def recent(limit: int = 8) -> Dict[str, Any]:
    """What I did last - the journal, spoken."""
    entries = journal(limit)
    if not entries:
        return _result(True, "I haven't touched any files yet this session.", entries=[])
    lines = []
    for entry in reversed(entries):
        when = datetime.fromtimestamp(int(entry.get("at", 0))).strftime("%H:%M") if entry.get("at") else "?"
        target = Path(str(entry.get("path", "?"))).name
        lines.append(f"{when} {entry.get('action', '?')} {target}")
    return _result(True, "Recent file changes:\n" + "\n".join(lines), entries=entries[::-1])


__all__ = ["roots", "resolve_path", "read", "write", "note", "delete", "undo", "list_dir", "search",
           "move", "mkdir", "disk_report", "open_file", "script", "execute_script", "recent",
           "journal", "Located"]
