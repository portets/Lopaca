"""Persistent chat and document storage, independent of the GUI."""

import json
import os
import shutil
import sqlite3
import time
import uuid
from contextlib import closing
from pathlib import Path

from settings import CHAT_SORT_OPTIONS, Settings

APP_DIR = Path.home() / ".local" / "share" / "lopaca"
CHATS_DIR = APP_DIR / "chats"


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

    def disk_usage(self):
        """Sum file sizes in one metadata-only scan, without following symlinks."""
        chats, total = {}, 0
        pending = [(self.root.parent, None)]
        while pending:
            folder, chat_id = pending.pop()
            try:
                with os.scandir(folder) as entries:
                    for entry in entries:
                        try:
                            if entry.is_dir(follow_symlinks=False):
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
            return {"code_agent": False, "created": time.time()}
        data["code_agent"] = data.get("code_agent", Settings.code_agent) is True
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
        return data

    def save_app_settings(self, settings):
        self.ensure()
        _write_json(self.root.parent / "app-settings.json", settings)

    def load(self, chat_id):
        settings = self.load_settings(chat_id)
        data = {"id": chat_id, "title": "New chat", "created": settings["created"], "messages": []}
        for key in ("title", "created", "updated", "inference_stats"):
            if key in settings:
                data[key] = settings[key]
        path = self.folder(chat_id) / "chat-history.sqlite"
        if path.exists():
            with closing(sqlite3.connect(path)) as connection:
                data["messages"] = [
                    json.loads(message)
                    for (message,) in connection.execute(
                        "SELECT message FROM messages ORDER BY position"
                    )
                ]
        return data

    def save(self, data):
        chat_id = data["id"]
        data.get("inference_stats", {}).pop("tokens_per_second", None)
        self.documents(chat_id).mkdir(parents=True, exist_ok=True)
        settings_path = self.folder(chat_id) / "chat-settings.json"
        try:
            settings = _read_json_object(settings_path)
        except FileNotFoundError:
            settings = {"code_agent": Settings.code_agent}
        settings.update(
            {key: value for key, value in data.items() if key not in {"messages", "pinned"}}
        )
        # Keep each message's optional fields without duplicating conversation data in settings.
        with (
            closing(sqlite3.connect(self.folder(chat_id) / "chat-history.sqlite")) as connection,
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
                    for index, message in enumerate(data["messages"])
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
                self.load(entry.name)
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
            chat["created"] = self.load_settings(chat["id"])["created"]

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
        folder.mkdir(parents=True, exist_ok=True)
        for path in map(Path, paths):
            if not path.is_file():
                continue
            target = folder / path.name
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
