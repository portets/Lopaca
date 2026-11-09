"""Persistent chat and document storage, independent of the GUI."""

import json
import os
import re
import shutil
import sqlite3
import time
import uuid
from contextlib import closing
from pathlib import Path

from settings import CHAT_SORT_OPTIONS, CHATS_DIR, CONVERSATION_STYLE_OPTIONS, Settings


def _read_json_object(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _write_json(path, data):
    temp_path = path.with_name(path.name + ".tmp")
    temp_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temp_path.replace(path)


class ChatStore:
    def __init__(self, root=CHATS_DIR):
        self.root = Path(root)

    def ensure(self):
        self.root.mkdir(parents=True, exist_ok=True)

    def folder(self, chat_id):
        return self.root / chat_id

    def documents(self, chat_id):
        return self.folder(chat_id) / "documents"

    def projects(self, chat_id):
        return self.folder(chat_id) / "projects"

    def disk_usage(self):
        """Sum data file sizes, excluding the installed app and not following symlinks."""
        chats, total = {}, 0
        pending = [(self.root.parent, None)]
        while pending:
            folder, chat_id = pending.pop()
            try:
                with os.scandir(folder) as entries:
                    for entry in entries:
                        try:
                            if entry.is_dir(follow_symlinks=False):
                                if folder == self.root.parent and entry.name == "app":
                                    continue
                                child_id = entry.name if folder == self.root else chat_id
                                pending.append((Path(entry.path), child_id))
                            elif entry.is_file(follow_symlinks=False):
                                size = entry.stat(follow_symlinks=False).st_size
                                total += size
                                if chat_id is not None:
                                    chats[chat_id] = chats.get(chat_id, 0) + size
                        except OSError:
                            pass  # Files may be removed while the scan is running.
            except OSError:
                pass
        return chats, total

    def load_settings(self, chat_id):
        try:
            data = _read_json_object(self.folder(chat_id) / "chat-settings.json")
        except FileNotFoundError:
            data = {}
        except OSError, json.JSONDecodeError:
            return {"created": time.time()}
        if not isinstance(data.get("created"), (int, float)):
            data["created"] = time.time()
        return data

    def save_settings(self, chat_id, settings):
        folder = self.folder(chat_id)
        folder.mkdir(parents=True, exist_ok=True)
        _write_json(folder / "chat-settings.json", settings)

    def load_app_settings(self):
        try:
            data = _read_json_object(self.root.parent / "app-settings.json")
        except OSError, json.JSONDecodeError:
            data = {}
        for key in ("pinned_chats", "unpinned_chat_order"):
            chat_order = data.get(key, [])
            data[key] = (
                list(dict.fromkeys(chat_id for chat_id in chat_order if isinstance(chat_id, str)))
                if isinstance(chat_order, list)
                else []
            )
        if data.get("chat_sort") not in dict(CHAT_SORT_OPTIONS):
            data["chat_sort"] = Settings.chat_sort
        if data.get("conversation_style") not in dict(CONVERSATION_STYLE_OPTIONS):
            data["conversation_style"] = Settings.conversation_style
        return data

    def save_app_settings(self, settings):
        self.ensure()
        _write_json(self.root.parent / "app-settings.json", settings)

    def load(self, chat_id, *, history=True):
        settings = self.load_settings(chat_id)
        data = {"id": chat_id, "title": "New chat", "created": settings["created"], "messages": []}
        for key in ("title", "created", "updated", "inference_stats"):
            if key in settings:
                data[key] = settings[key]
        path = self.folder(chat_id) / "chat-history.sqlite"
        if history and path.exists():
            with closing(sqlite3.connect(path)) as connection:
                data["messages"] = [
                    json.loads(message)
                    for (message,) in connection.execute(
                        "SELECT message FROM messages ORDER BY position"
                    )
                ]
        return data

    def save(self, data, *, message_start=0):
        chat_id = data["id"]
        if not 0 <= message_start <= len(data["messages"]):
            raise ValueError("Invalid first message position")
        history_path = self.folder(chat_id) / "chat-history.sqlite"
        if not history_path.exists():
            message_start = 0
        self.documents(chat_id).mkdir(parents=True, exist_ok=True)
        self.projects(chat_id).mkdir(exist_ok=True)
        settings_path = self.folder(chat_id) / "chat-settings.json"
        try:
            settings = _read_json_object(settings_path)
        except FileNotFoundError:
            settings = {}
        settings.update(
            {key: value for key, value in data.items() if key not in {"messages", "pinned"}}
        )
        # Keep each message's optional fields without duplicating conversation data in settings.
        with (
            closing(sqlite3.connect(history_path)) as connection,
            connection,
        ):
            connection.execute(
                "CREATE TABLE IF NOT EXISTS messages "
                "(position INTEGER PRIMARY KEY, message TEXT NOT NULL)"
            )
            connection.executemany(
                "INSERT INTO messages (position, message) VALUES (?, ?) "
                "ON CONFLICT(position) DO UPDATE SET message = excluded.message "
                "WHERE messages.message != excluded.message",
                (
                    (index, json.dumps(message, ensure_ascii=False))
                    for index, message in enumerate(data["messages"][message_start:], message_start)
                ),
            )
            connection.execute("DELETE FROM messages WHERE position >= ?", (len(data["messages"]),))
            self.save_settings(chat_id, settings)
        documents_db = self.folder(chat_id) / "documents.sqlite"
        if not documents_db.exists():
            with closing(sqlite3.connect(documents_db)) as connection:
                connection.execute("PRAGMA user_version = 1")

    def create(self):
        chat = {"id": uuid.uuid4().hex, "title": "New chat", "messages": [], "created": time.time()}
        self.save(chat)
        return chat

    def list(self):
        self.ensure()
        with os.scandir(self.root) as entries:
            chats = [
                self.load(entry.name, history=False)
                for entry in entries
                if entry.is_dir() and (Path(entry.path) / "chat-history.sqlite").exists()
            ]
        app_settings = self.load_app_settings()
        pinned_order = {
            chat_id: index for index, chat_id in enumerate(app_settings["pinned_chats"])
        }
        manual_order = {
            chat_id: index for index, chat_id in enumerate(app_settings["unpinned_chat_order"])
        }
        for chat in chats:
            chat["pinned"] = chat["id"] in pinned_order

        def sort_key(chat):
            if chat["pinned"]:
                return 0, pinned_order[chat["id"]]
            if app_settings["chat_sort"] == "recently_used":
                return 1, -chat.get("updated", chat["created"])
            if app_settings["chat_sort"] == "creation_date":
                return 1, -chat["created"]
            # New chats precede the saved order; without one, use creation date.
            return 1, manual_order.get(chat["id"], -1), -chat["created"]

        chats.sort(key=sort_key)
        return chats

    def add_documents(self, chat_id, paths):
        folder = self.documents(chat_id)
        folder.mkdir(exist_ok=True)
        for path, relative in paths:
            path = Path(path)
            if not path.is_file() or path.is_symlink():
                continue
            target = folder / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                target = folder / f"{path.stem}-{uuid.uuid4().hex[:6]}{path.suffix}"
            shutil.copyfile(path, target)

    def delete(self, chat_id):
        shutil.rmtree(self.folder(chat_id))
        settings = self.load_app_settings()
        order_keys = ("pinned_chats", "unpinned_chat_order")
        if any(chat_id in settings[key] for key in order_keys):
            for key in order_keys:
                settings[key] = [item for item in settings[key] if item != chat_id]
            self.save_app_settings(settings)


class MemoryStore:
    """Bounded user facts; models propose changes, never SQL or filesystem operations."""

    def __init__(self, path, settings):
        self.path, self.settings = Path(path), settings

    def connect(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=5)
        connection.executescript("""
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT, text TEXT NOT NULL,
                normalized TEXT NOT NULL UNIQUE, core INTEGER NOT NULL,
                source_chat TEXT, evidence TEXT, created REAL, updated REAL,
                search_terms TEXT NOT NULL
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS memory_search USING fts5(
                text, keywords, tokenize='porter unicode61'
            );
        """)
        return connection

    def records(self):
        with closing(self.connect()) as connection:
            return connection.execute(
                "SELECT id,text,core FROM memories ORDER BY core DESC,updated DESC,id DESC"
            ).fetchall()

    def apply(self, proposals, user_text, chat_id, allowed_ids):
        """Apply validated, evidenced proposals atomically."""
        if not isinstance(proposals, list):
            return []
        accepted = []
        with closing(self.connect()) as connection, connection:
            for index, proposal in enumerate(proposals[:8]):
                if not isinstance(proposal, dict):
                    continue
                action, memory_id = proposal.get("action"), proposal.get("id")
                if not isinstance(action, str) or action not in {"add", "update", "delete"}:
                    continue
                if action == "add":
                    memory_id = None
                evidence = proposal.get("evidence")
                if (
                    not isinstance(evidence, str)
                    or not evidence.strip()
                    or " ".join(evidence.casefold().split())
                    not in " ".join(user_text.casefold().split())
                ):
                    continue
                if action != "add" and (type(memory_id) is not int or memory_id not in allowed_ids):
                    continue
                text = proposal.get("text", "")
                if not isinstance(text, str):
                    continue
                text = " ".join(text.split())
                if action != "delete" and not 1 <= len(text) <= 500:
                    continue
                kind = proposal.get("kind")
                if action != "delete" and (
                    not isinstance(kind, str) or kind not in {"core", "casual"}
                ):
                    continue
                normalized = " ".join(re.findall(r"\w+", text.casefold()))
                if action != "delete" and not normalized:
                    continue
                core = kind == "core"
                rows = connection.execute("SELECT id,text,core FROM memories").fetchall()
                if action != "add" and not any(row[0] == memory_id for row in rows):
                    continue
                if action != "delete":
                    search_terms = proposal.get("search_terms")
                    if action == "update" and search_terms is None:
                        search_terms = connection.execute(
                            "SELECT search_terms FROM memories WHERE id=?", (memory_id,)
                        ).fetchone()[0]
                    if not isinstance(search_terms, str) or not search_terms.strip():
                        continue
                    search_terms = " ".join(search_terms.split())[:200]
                    duplicate = connection.execute(
                        "SELECT id FROM memories WHERE normalized=?", (normalized,)
                    ).fetchone()
                    if duplicate and duplicate[0] != memory_id:
                        continue
                    others = [row for row in rows if row[0] != memory_id]
                    if action == "add" and len(rows) >= self.settings.memory_records:
                        continue
                    if core and (
                        sum(bool(row[2]) for row in others) >= self.settings.memory_core_records
                        or sum(len(row[1]) for row in others if row[2]) + len(text)
                        > self.settings.memory_core_chars
                    ):
                        core = False
                if action == "delete":
                    connection.execute("DELETE FROM memories WHERE id=?", (memory_id,))
                    connection.execute("DELETE FROM memory_search WHERE rowid=?", (memory_id,))
                else:
                    now = time.time()
                    if action == "add":
                        memory_id = connection.execute(
                            "INSERT INTO memories "
                            "(text,normalized,core,source_chat,evidence,created,updated,"
                            "search_terms) "
                            "VALUES (?,?,?,?,?,?,?,?)",
                            (text, normalized, core, chat_id, evidence, now, now, search_terms),
                        ).lastrowid
                    else:
                        connection.execute(
                            "UPDATE memories SET text=?,normalized=?,core=?,source_chat=?,"
                            "evidence=?,updated=?,search_terms=? WHERE id=?",
                            (
                                text,
                                normalized,
                                core,
                                chat_id,
                                evidence,
                                now,
                                search_terms,
                                memory_id,
                            ),
                        )
                    connection.execute("DELETE FROM memory_search WHERE rowid=?", (memory_id,))
                    connection.execute(
                        "INSERT INTO memory_search(rowid,text,keywords) VALUES (?,?,?)",
                        (memory_id, text, re.sub(r"[_-]", " ", text + " " + search_terms)),
                    )
                accepted.append(index)
        return accepted

    def set_core(self, memory_id, core):
        with closing(self.connect()) as connection, connection:
            connection.execute("UPDATE memories SET core=? WHERE id=?", (bool(core), memory_id))
