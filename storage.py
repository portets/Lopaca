"""Persistent chat and document storage, independent of the GUI."""

import json
import os
import shutil
import time
import uuid
from pathlib import Path

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
        # Preserve chats from installations made before the app was renamed.
        legacy_dir = APP_DIR.with_name("llama-chat")
        if self.root == CHATS_DIR and legacy_dir.exists() and not APP_DIR.exists():
            legacy_dir.rename(APP_DIR)
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
            data = {"code_agent": False}
            self.save_settings(chat_id, data)
        except OSError, json.JSONDecodeError:
            data = {}
        data["code_agent"] = data.get("code_agent") is True
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
        pinned_chats = data.get("pinned_chats", [])
        data["pinned_chats"] = (
            list(dict.fromkeys(chat_id for chat_id in pinned_chats if isinstance(chat_id, str)))
            if isinstance(pinned_chats, list)
            else []
        )
        return data

    def save_app_settings(self, settings):
        self.ensure()
        _write_json(self.root.parent / "app-settings.json", settings)

    def load(self, chat_id):
        try:
            data = _read_json_object(self.folder(chat_id) / "chat.json")
            if isinstance(data.get("messages"), list):
                return data
        except OSError, json.JSONDecodeError:
            pass
        return {"id": chat_id, "title": "New chat", "messages": [], "created": time.time()}

    def save(self, data):
        data.get("inference_stats", {}).pop("tokens_per_second", None)
        self.documents(data["id"]).mkdir(parents=True, exist_ok=True)
        _write_json(self.folder(data["id"]) / "chat.json", data)

    def create(self):
        chat = {"id": uuid.uuid4().hex, "title": "New chat", "messages": [], "created": time.time()}
        self.save(chat)
        self.save_settings(chat["id"], {"code_agent": False})
        return chat

    def list(self):
        self.ensure()
        with os.scandir(self.root) as entries:
            chats = [
                self.load(entry.name)
                for entry in entries
                if entry.is_dir() and (Path(entry.path) / "chat.json").exists()
            ]
        pinned_order = {
            chat_id: index for index, chat_id in enumerate(self.load_app_settings()["pinned_chats"])
        }
        for chat in chats:
            chat["pinned"] = chat["id"] in pinned_order
        chats.sort(
            key=lambda chat: (
                (0, pinned_order[chat["id"]])
                if chat["pinned"]
                else (1, -chat.get("updated", chat.get("created", 0)))
            )
        )
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
        if chat_id in settings["pinned_chats"]:
            settings["pinned_chats"].remove(chat_id)
            self.save_app_settings(settings)
