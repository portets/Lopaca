"""GTK/libadwaita presentation layer. Application behavior lives in main.py."""

import re
import threading
import time
from collections import deque
from contextlib import suppress
from datetime import datetime
from pathlib import Path
from typing import ClassVar

import gi

from agentic import DocumentFiles, get_clock_format
from images import image_mime_type
from markdown_ui import (
    CODE_BLOCK_PATTERN,
    MarkdownText,
    ReplyImages,
    append_code_block,
    append_markdown,
    load_picture,
)
from model_server import (
    DOWNLOADING_MODEL,
    SAMPLING_FIELDS,
    SERVER_FIELD_CHOICES,
    SERVER_FIELDS,
    server_parameters,
)
from server_builds import (
    BACKEND_LABELS,
    cached_server_config,
    embedding_launch_parameters,
    embedding_path,
    managed_server_file,
    model_path,
    model_vision_family,
    vision_file_precision,
    vision_path,
)
from settings import (
    CHAT_SORT_OPTIONS,
    CONVERSATION_STYLE_OPTIONS,
    EMBEDDING_DOWNLOAD,
    MODEL_DOWNLOADS,
    MODEL_RECOMMENDATIONS,
    VISION_DOWNLOADS,
)

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
gi.require_version("Graphene", "1.0")
gi.require_version("Gsk", "4.0")
# GI namespace versions must be selected before importing their bindings.
from gi.repository import Adw, Gdk, Gio, GLib, GObject, Graphene, Gsk, Gtk, Pango  # noqa: E402

ATTACH_DOCUMENTS_ICON = "mail-attachment-symbolic"
OPTIONAL_FEATURES_TOOLTIP = (
    "Downloads additional dependencies for document indexing, faster downloads, "
    "and retrieval speedups."
)
ATTACHMENT_THUMBNAIL_SIZE = 64

CSS = """
/* Theme styling. Margins and padding use top, right, bottom, left.
 * Window geometry, alignment and calculated scroll/text metrics are in layout code. */

/* Window, header and shared text */
window.lopaca-main {
    color: @theme_fg_color;
}
window.lopaca-main, .transcript-scroll, .transcript {
    background-color: @theme_bg_color;
}
.app-titlebar {
    background-color: @headerbar_bg_color;
    border-bottom: none;
    box-shadow: 0 1px 1px @headerbar_shade_color;
}
.chat-prose {
    line-height: 1.4em;
}
.muted {
    color: alpha(@window_fg_color,0.62);
    font-size: 0.9em;
}
.chat-list-scroll {
    margin-bottom: 0px;
}
.chat-header {
    border-spacing: 8px;
    margin-left: 20px;
    margin-right: 20px;
    margin-top: 12px;
    margin-bottom: 10px;
}
.chat-header-titles {
    border-spacing: 3px;
}
.transcript {
    border-spacing: 2px;
    margin-top: 16px;
    margin-bottom: 18px;
    margin-left: 24px;
    margin-right: 24px;
}
.app-toolbar {
    /* Theme defaults; add overrides here when needed. */
}
.app-header-bar {
    /* Theme defaults; add overrides here when needed. */
}
.app-title {
    /* Theme defaults; add overrides here when needed. */
}
.app-layout {
    /* Theme defaults; add overrides here when needed. */
}
.chat-list-overlay {
    /* Theme defaults; add overrides here when needed. */
}
.disk-usage {
    margin-right: -8px;
    margin-left: 8px;
}
.chat-disk-usage {
    /* Theme defaults; add overrides here when needed. */
}
.total-disk-usage {
    /* Theme defaults; add overrides here when needed. */
}
.chat-pane {
    /* Theme defaults; add overrides here when needed. */
}
.chat-title {
    /* Theme defaults; add overrides here when needed. */
}
.chat-folder-path {
    /* Theme defaults; add overrides here when needed. */
}
.main-transcript-scroll {
    /* Theme defaults; add overrides here when needed. */
}
.response-status {
    /* Theme defaults; add overrides here when needed. */
}
.status-spacer {
    /* Theme defaults; add overrides here when needed. */
}
.context-usage {
    /* Theme defaults; add overrides here when needed. */
}
.generation-speed {
    /* Theme defaults; add overrides here when needed. */
}
.main-frame {
    /* Theme defaults; add overrides here when needed. */
}
.prompt-scroll {
    /* Theme defaults; add overrides here when needed. */
}
.send-stop {
    /* Theme defaults; add overrides here when needed. */
}

/* Sidebar and chat list */
.chat-list-divider {
    margin: 6px 0;
}
.sidebar {
    padding: 4px 0;
    min-width: 286px;
    border-spacing: 8px;
    margin-top: 13px;
    margin-bottom: 12px;
    margin-left: 12px;
    margin-right: 12px;
}
.sidebar, .composer-frame, .document-drop-box {
    background-color: alpha(@headerbar_shade_color,0.4);
    border: 1px solid @borders;
    border-radius: 10px;
}
.workspace-list row {
    border-radius: 8px;
    margin: 0;
    padding: 0;
}
.workspace-list:drop(active) {
    background-color: transparent;
    background-image: none;
    box-shadow: none;
    outline: none;
}
.workspace-list row:hover {
    background-color: alpha(@theme_fg_color,0.07);
}
.workspace-list row.active-chat, .workspace-list row:selected {
    background-color: alpha(@theme_selected_bg_color,0.4);
}
.chat-row {
    padding: 1px 2px 1px 10px;
    margin-top: 1px;
    margin-bottom: 1px;
    border-radius: 9px;
    border-spacing: 4px;
}
.chat-options-button {
    opacity: 0;
}
.chat-row:hover .chat-options-button, .chat-options-button:focus {
    opacity: 1;
}
.chat-options-menu modelbutton {
    font-weight: normal;
}
.chat-drop-indicator {
    min-height: 3px;
    border-radius: 2px;
    background-color: alpha(@window_fg_color,0.28);
    margin-left: 22px;
    margin-right: 20px;
}
.chat-drop-indicator:dir(rtl) {
    margin-left: 20px;
    margin-right: 22px;
}
.sidebar-header {
    border-spacing: 8px;
    margin-top: 4px;
    margin-left: 8px;
    margin-right: 6px;
}
.sidebar-header:dir(rtl) {
    margin-left: 6px;
    margin-right: 8px;
}
.main-chat-list {
    margin-left: 8px;
    margin-right: 8px;
}
.main-chat-list:dir(rtl) {
    margin-left: 12px;
    margin-right: 14px;
}
.sidebar-footer {
    border-spacing: 8px;
    margin-left: 10px;
    margin-right: 10px;
    margin-bottom: 4px;
}
.chats-heading {
    margin-right: -1px;
    margin-left: 1px;
}
.new-chat-button {
    margin-right: 4px;
    margin-left: -4px;
}
.app-settings-button {
    /* Theme defaults; add overrides here when needed. */
}
@keyframes first-launch-pulse {
    from { box-shadow: inset 0 0 0 2px alpha(@accent_color, 0.95); }
    to { box-shadow: inset 0 0 0 2px alpha(@accent_color, 0.12); }
}
.first-launch-highlight,
.first-launch-model-tab viewswitcher > button:nth-child(2) {
    box-shadow: inset 0 0 0 2px @accent_color;
    animation: first-launch-pulse 400ms ease-in-out infinite alternate;
}
.chat-list-row {
    /* Theme defaults; add overrides here when needed. */
}
.chat-list-label {
    /* Theme defaults; add overrides here when needed. */
}
.chat-list-options-button {
    /* Theme defaults; add overrides here when needed. */
}

/* Messages and reasoning */
.message-row {
    padding: 5px 0;
}
.message {
    padding: 10px 13px;
    border-radius: 14px;
    border-spacing: 6px;
}
.message label {
    caret-color: transparent;
    -gtk-secondary-caret-color: transparent;
}
.user-message {
    background-color: alpha(@accent_bg_color,0.4);
}
.message-attachment-button {
    min-width: 0;
    min-height: 0;
    padding: 2px 5px;
    font-size: 0.9em;
}
.message-attachment-content {
    border-spacing: 6px;
}
.message-attachment-button box, .message-attachment-thumbnail {
    border-radius: inherit;
}
.markdown-image {
    min-width: 0;
    min-height: 0;
    padding: 0;
}
.markdown-image picture {
    border-radius: inherit;
}
.assistant-message {
    background-color: alpha(@theme_fg_color,0.045);
}
.message-role {
    font-size: 0.78em;
    font-weight: 700;
    opacity: 0.68;
}
.message-time {
    opacity: 0;
}
.message:hover + .message-time {
    opacity: 1;
}
.thinking {
    background-color: alpha(@theme_fg_color,0.065);
    border-left: 3px solid @accent_color;
    padding: 3px 10px;
    margin-bottom: 8px;
    border-radius: 5px;
    border-spacing: 4px;
}
.thinking button {
    min-height: 28px;
    padding: 2px 4px;
}
.thinking, .thinking button, .thinking label {
    font-weight: normal;
}
.message-date-separator {
    margin-bottom: 2px;
    margin-top: 12px;
}
.message-column {
    border-spacing: 3px;
}
.reasoning-content {
    border-spacing: 6px;
}
.reasoning-body {
    border-spacing: 0px;
}
.streaming-response-current-content {
    border-spacing: 6px;
}
.message-time-label {
    /* Theme defaults; add overrides here when needed. */
}
.message-label {
    /* Theme defaults; add overrides here when needed. */
}
.reasoning-header-button {
    /* Theme defaults; add overrides here when needed. */
}
.reasoning-arrow {
    /* Theme defaults; add overrides here when needed. */
}
.model-settings-reasoning {
    /* Theme defaults; add overrides here when needed. */
}

/* Markdown, tables and code */
.markdown-text {
    border-spacing: 6px;
}
.markdown-text.markdown-fragmented {
    border-spacing: 0;
}
.markdown-block-gap {
    margin-top: 6px;
}
.markdown-quote-content {
    border-spacing: 4px;
}
.code-block-frame {
    background-color: alpha(@theme_fg_color,0.035);
    border: 1px solid alpha(@theme_fg_color,0.20);
    border-radius: 8px;
    padding: 9px 11px;
}
.code-block {
    font-family: monospace;
}
.markdown-table {
    border: 1px solid @borders;
    border-radius: 6px;
}
.table-cell {
    padding: 5px 8px;
}
.table-heading {
    font-weight: bold;
    background-color: alpha(@theme_fg_color,0.06);
    border-bottom: 1px solid @borders;
}
.table-alternate {
    background-color: alpha(@theme_fg_color,0.025);
}
.markdown-quote {
    padding: 6px 10px;
    border-left: 3px solid @accent_color;
    background-color: alpha(@theme_fg_color,0.035);
}

/* Prompt and response controls */
.thinking-dropdown {
    margin-right: 8px;
}
/* Inside the closed button, around the text and arrow */
.thinking-dropdown > button {
    padding-left: 17px;
    padding-right: 8px;
}

/* Only the text inside the closed button */
.thinking-dropdown > button label {
}

/* Inside the expanded popup's outer box */
.thinking-dropdown popover contents {
}

/* Inside each selectable option */
.thinking-dropdown popover listview row {
    padding-left: 12px;
    padding-right: 12px;
    padding-top: 6px;
    padding-bottom: 6px;
}

/* Only the text inside each option */
.thinking-dropdown popover listview row label {
    margin-left: -2px;
    margin-right: 2px;
}
.thinking-dropdown:dir(rtl) {
    margin-left: 8px;
    margin-right: 0;
}
.thinking-dropdown label {
    font-weight: normal;
    margin-left: 0px;
}
.prompt-view, .prompt-view text {
    background-color: transparent;
}
.send-stop.suggested-action:not(:disabled) {
    background-image: none;
    background-color: @accent_bg_color;
    color: @accent_fg_color;
}
.send-stop.destructive-action:not(:disabled) {
    background-image: none;
    background-color: @destructive_bg_color;
    color: @destructive_fg_color;
}
.composer {
    border-spacing: 7px;
    margin-top: 12px;
    margin-bottom: 12px;
    margin-left: 12px;
    margin-right: 12px;
}
.composer-status {
    border-spacing: 14px;
}
.composer-activity {
    border-spacing: 6px;
}
.composer-frame-row {
    border-spacing: 10px;
}
.composer-content {
    border-spacing: 3px;
    margin-top: 8px;
    margin-bottom: 8px;
    margin-left: 10px;
    margin-right: 10px;
}
.composer-actions {
    border-spacing: 8px;
}
.model-picker {
    border-spacing: 8px;
}
.model-picker > dropdown > button label {
    font-weight: normal;
}
.prompt-hint {
    /* Theme defaults; add overrides here when needed. */
}
.thinking-label {
    /* Theme defaults; add overrides here when needed. */
}
.attach-documents-button {
    margin-right: 2px;
}

/* Settings pages and shared settings controls */
.about-page .about-link-button {
    color: @window_fg_color;
}
preferencespage button,
preferencespage row.activatable:not(.settings-edit-entry):not(.memories-row),
popover.menu modelbutton {
    transition: box-shadow 120ms ease-out, background-color 120ms ease-out;
}
preferencespage button:active,
preferencespage row.activatable:not(.settings-edit-entry):not(.memories-row):active,
popover.menu modelbutton:active {
    box-shadow: inset 0 0 0 1px alpha(@window_fg_color, 0.18);
    transition-duration: 45ms;
}
button.edit-button-pressed {
    background-color: alpha(@window_fg_color, 0.16);
    box-shadow: inset 0 0 0 1px alpha(@window_fg_color, 0.23);
    transition-duration: 45ms;
}
.model-settings-advanced.model-advanced:active,
.model-settings-advanced.model-advanced:hover {
    background-color: transparent;
    box-shadow: none;
}
.delete-icon {
    -gtk-icon-size: 24px;
}
.model-settings-text {
    padding: 10px;
}
.model-advanced image.expander-row-arrow {
    -gtk-icon-transform: rotate(0turn);
}
.model-advanced:checked image.expander-row-arrow {
    -gtk-icon-transform: rotate(0.5turn);
}
.settings-control-group {
    border-spacing: 8px;
}
.model-settings-advanced-content {
    border-spacing: 18px;
    margin-top: 12px;
    margin-bottom: 12px;
    margin-left: 12px;
    margin-right: 12px;
}
.knowledge-packs-actions {
    border-spacing: 6px;
}
.memories-controls {
    border-spacing: 6px;
}
.settings-page {
    /* Theme defaults; add overrides here when needed. */
}
.settings-group {
    /* Theme defaults; add overrides here when needed. */
}
.settings-row {
    /* Theme defaults; add overrides here when needed. */
}
.optional-features-highlight {
    box-shadow: inset 0 0 0 2px @accent_color;
}
.settings-button {
    /* Theme defaults; add overrides here when needed. */
}
.settings-memories-row {
    /* Theme defaults; add overrides here when needed. */
}
.settings-choice-group {
    /* Theme defaults; add overrides here when needed. */
}
.override-delete-button {
    /* Theme defaults; add overrides here when needed. */
}
.about-page {
    /* Theme defaults; add overrides here when needed. */
}
.about-project {
    /* Theme defaults; add overrides here when needed. */
}
.about-licenses {
    /* Theme defaults; add overrides here when needed. */
}
.about-donations {
    /* Theme defaults; add overrides here when needed. */
}
.model-settings-page {
    /* Theme defaults; add overrides here when needed. */
}
.model-settings-summary {
    /* Theme defaults; add overrides here when needed. */
}
.list-add-controls {
    margin-top: 6px;
}
.model-settings-summary-row {
    /* Theme defaults; add overrides here when needed. */
}
.model-settings-advanced-group {
    /* Theme defaults; add overrides here when needed. */
}
.model-settings-advanced {
    /* Theme defaults; add overrides here when needed. */
}
.model-settings-preserve {
    /* Theme defaults; add overrides here when needed. */
}
.model-settings-options {
    /* Theme defaults; add overrides here when needed. */
}
.model-settings-extra {
    /* Theme defaults; add overrides here when needed. */
}
.model-settings-scroll {
    /* Theme defaults; add overrides here when needed. */
}
.model-settings-frame {
    /* Theme defaults; add overrides here when needed. */
}
.model-settings-actions {
    /* Theme defaults; add overrides here when needed. */
}
.model-server-status {
    /* Theme defaults; add overrides here when needed. */
}
.model-connect-button {
    /* Theme defaults; add overrides here when needed. */
}
.model-settings-row {
    /* Theme defaults; add overrides here when needed. */
}
.settings-edit-entry .edit-icon {
    opacity: 0;
    -gtk-icon-size: 1px;
    min-width: 0;
    min-height: 0;
    padding: 0;
    margin: 0;
}
.settings-subpage-toolbar {
    /* Theme defaults; add overrides here when needed. */
}
.settings-subpage-header {
    /* Theme defaults; add overrides here when needed. */
}
.settings-subpage-page {
    /* Theme defaults; add overrides here when needed. */
}
.settings-subpage-group {
    /* Theme defaults; add overrides here when needed. */
}
.knowledge-page {
    /* Theme defaults; add overrides here when needed. */
}
.knowledge-group {
    /* Theme defaults; add overrides here when needed. */
}
.knowledge-packs-merge {
    /* Theme defaults; add overrides here when needed. */
}
.knowledge-packs-button {
    /* Theme defaults; add overrides here when needed. */
}
.knowledge-packs-label {
    /* Theme defaults; add overrides here when needed. */
}
.knowledge-packs-spinner {
    /* Theme defaults; add overrides here when needed. */
}
.knowledge-packs-row {
    /* Theme defaults; add overrides here when needed. */
}
.knowledge-packs-switch {
    /* Theme defaults; add overrides here when needed. */
}
.knowledge-packs-info {
    /* Theme defaults; add overrides here when needed. */
}
.memories-row {
    /* Theme defaults; add overrides here when needed. */
}
.memories-core-button {
    /* Theme defaults; add overrides here when needed. */
}
.list-remove-button {
    /* Theme defaults; add overrides here when needed. */
}

/* Dialogs and document attachments */
.command-review-preview {
    padding: 0 4px;
    font-family: monospace;
}
.document-drop-icon {
    -gtk-icon-size: 32px;
}
.command-preview {
    border-radius: 6px;
}
.rename-chat-entry {
    margin-top: 18px;
    margin-bottom: 18px;
    margin-left: 18px;
    margin-right: 18px;
}
.rename-chat-content {
    border-spacing: 12px;
}
.rename-chat-buttons {
    border-spacing: 8px;
    margin-bottom: 18px;
    margin-left: 18px;
    margin-right: 18px;
}
.document-drop-box {
    margin-left: 12px;
    margin-right: 12px;
    margin-top: 8px;
    margin-bottom: 12px;
}
.document-drop-center {
    border-spacing: 10px;
}
.command-review-box {
    border-spacing: 10px;
    margin-top: 18px;
    margin-bottom: 18px;
    margin-left: 18px;
    margin-right: 18px;
}
.command-review-buttons {
    border-spacing: 8px;
}
.rename-chat-toolbar {
    /* Theme defaults; add overrides here when needed. */
}
.rename-chat-header {
    /* Theme defaults; add overrides here when needed. */
}
.rename-chat-cancel-button {
    /* Theme defaults; add overrides here when needed. */
}
.rename-chat-save-button {
    /* Theme defaults; add overrides here when needed. */
}
.document-drop-toolbar {
    /* Theme defaults; add overrides here when needed. */
}
.command-review-toolbar {
    /* Theme defaults; add overrides here when needed. */
}
.command-review-scroll {
    /* Theme defaults; add overrides here when needed. */
}
.command-review-frame {
    /* Theme defaults; add overrides here when needed. */
}
.command-review-cancel-button {
    /* Theme defaults; add overrides here when needed. */
}
.command-review-approve-button {
    /* Theme defaults; add overrides here when needed. */
}
.command-review-label {
    /* Theme defaults; add overrides here when needed. */
}
.command-review-legend {
    /* Theme defaults; add overrides here when needed. */
}
.command-review-notice {
    /* Theme defaults; add overrides here when needed. */
}
.command-review-warnings {
    /* Theme defaults; add overrides here when needed. */
}
"""


def message_time(timestamp, now=None, clock_format=None):
    date = datetime.fromtimestamp(timestamp)
    days = ((now or datetime.now()).date() - date.date()).days
    clock = (
        f"{date.hour:02d}:{date.minute:02d}"
        if (clock_format or get_clock_format()) == "24h"
        else f"{date.hour % 12 or 12}:{date.minute:02d} {'AM' if date.hour < 12 else 'PM'}"
    )
    if days == 0:
        return clock
    prefix = date.strftime("%a") if 0 < days < 7 else f"{date.strftime('%b')} {date.day}"
    return f"{prefix} {clock}"


def message_day(date, now=None):
    days = ((now or datetime.now()).date() - date).days
    return (
        "Today" if days == 0 else "Yesterday" if days == 1 else f"{date.strftime('%b')} {date.day}"
    )


def add_operation_indicator(row, text="", active=False, *, visible=True):
    label = Gtk.Label(label=text, valign=Gtk.Align.CENTER, visible=visible)
    label.add_css_class("knowledge-packs-label")
    spinner = Gtk.Spinner(spinning=active, valign=Gtk.Align.CENTER, visible=visible)
    spinner.add_css_class("knowledge-packs-spinner")
    row.add_suffix(label)
    row.add_suffix(spinner)
    return label, spinner


def model_file_size(size):
    return f" · {size / 10**9:.1f} GB"


EDIT_ICON = "document-edit-symbolic"
DONE_EDITING_ICON = "object-select-symbolic"


def pulse_edit_button(button):
    button.add_css_class("edit-button-pressed")

    def clear():
        button.remove_css_class("edit-button-pressed")
        return GLib.SOURCE_REMOVE

    GLib.timeout_add(200, clear)


def hint_edit_button(row, button, editing):
    gesture = Gtk.GestureClick(button=Gdk.BUTTON_PRIMARY)
    gesture.set_propagation_phase(Gtk.PropagationPhase.CAPTURE)

    def pressed(_gesture, _count, x, y):
        if editing():
            return
        target = row.pick(x, y, Gtk.PickFlags.DEFAULT)
        while target is not None and target is not row:
            if isinstance(target, (Gtk.Button, Gtk.CheckButton, Gtk.Switch)):
                return
            target = target.get_parent()
        if target is row:
            pulse_edit_button(button)

    gesture.connect("pressed", pressed)
    row.add_controller(gesture)


class DownloadAction(Gtk.Button):
    def __init__(self, label, **actions):
        super().__init__(
            valign=Gtk.Align.CENTER, sensitive=False, focus_on_click=False,
            css_classes=["knowledge-packs-button"],
        )
        self._label, self._actions = label, actions
        self._action = None
        self.connect("clicked", lambda *_: self._actions[self._action]())

    def show_action(self, action, *, enabled=True):
        self._action = action
        icon, tooltip = {
            "download": ("list-add-symbolic", f"Download {self._label}"),
            "remove": ("user-trash-symbolic", f"Remove {self._label}"),
            "update": ("software-update-available-symbolic", f"Update {self._label}"),
            "cancel": ("process-stop-symbolic", "Cancel Download"),
        }[action]
        self.set_icon_name(icon)
        self.set_tooltip_text(tooltip)
        self.set_sensitive(enabled)


def attachment_display_name(name, limit):
    name = " ".join(name.splitlines())
    if len(name) <= limit:
        return name
    suffix = "/" if name.endswith("/") else Path(name).suffix
    ellipsis = "..."
    available = limit - len(ellipsis)
    if len(suffix) >= available:
        return ellipsis + suffix[-available:]
    return name[: available - len(suffix)] + ellipsis + suffix


class BubbleLayout(Gtk.LayoutManager):
    """Limit one bubble to a fraction of its row, independent of other messages."""

    def __init__(self, user):
        super().__init__()
        self.user = user

    def do_get_request_mode(self, widget):
        return Gtk.SizeRequestMode.HEIGHT_FOR_WIDTH

    def _bubble_width(self, child, available):
        limit = available * 2 // 3 if self.user else available * 4 // 5
        natural = child.measure(Gtk.Orientation.HORIZONTAL, -1)[1]
        return max(1, min(limit, max(320, natural)))

    def do_measure(self, widget, orientation, for_size):
        # Message contents must never increase the transcript's requested width.
        if orientation == Gtk.Orientation.HORIZONTAL:
            return 0, 320, -1, -1
        child = widget.get_first_child()
        if child is None:
            return 0, 0, -1, -1
        width = self._bubble_width(child, for_size if for_size >= 0 else 480)
        return child.measure(orientation, width)

    def do_allocate(self, widget, width, height, baseline):
        child = widget.get_first_child()
        if child is None:
            return
        bubble_width = self._bubble_width(child, width)
        x = width - bubble_width if self.user else 0
        transform = Gsk.Transform.new().translate(Graphene.Point().init(x, 0))
        child.allocate(bubble_width, height, -1, transform)


class TranscriptViewport(Gtk.Viewport):
    def do_size_allocate(self, width, height, baseline):
        adjustment = self.get_vadjustment()
        previous = adjustment.get_value()
        Gtk.Viewport.do_size_allocate(self, width, height, baseline)
        # GTK can emit the scroll range change after allocating the transcript.
        # Apply the new offset before this layout pass can reach the screen.
        if adjustment.get_value() != previous:
            Gtk.Viewport.do_size_allocate(self, width, height, baseline)


class StatusLabel(Gtk.Label):
    def __init__(self, *, reserve_dots=False):
        super().__init__(xalign=0, ellipsize=Pango.EllipsizeMode.END, width_chars=1)
        self.add_css_class("response-status")
        self.add_css_class("muted")
        self._reserve_dots = reserve_dots
        self._message = ""
        self._animated = False
        self._timer = None
        self._dots = 0
        self.connect("map", lambda *_: self._start_timer())
        self.connect("unmap", lambda *_: self._stop_timer())

    def display(self, message, *, animated=False):
        if (message, animated) == (self._message, self._animated):
            return
        self._stop_timer()
        self._message, self._animated = message, animated
        self._dots = 0
        self._render()
        self.set_tooltip_text(message or None)
        self._start_timer()

    def _start_timer(self):
        if self._animated and self.get_mapped() and self._timer is None:
            self._timer = GLib.timeout_add(650, self._animate)

    def _stop_timer(self):
        if self._timer is not None:
            GLib.source_remove(self._timer)
            self._timer = None

    def _animate(self):
        self._dots = (self._dots + 1) % 4
        self._render()
        return GLib.SOURCE_CONTINUE

    def _render(self):
        dots = "." * self._dots
        if self._reserve_dots and self._animated and self._dots < 3:
            # Pango markup uses 1 as its smallest supported alpha.
            remaining = "." * (3 - self._dots)
            self.set_markup(
                f'{GLib.markup_escape_text(self._message)}{dots}'
                f'<span alpha="1">{remaining}</span>'
            )
        else:
            self.set_text(self._message + dots)


class ChatWindow(Adw.ApplicationWindow):
    __gsignals__: ClassVar = {
        "retrieval-settings-changed": (GObject.SignalFlags.RUN_LAST, None, ()),
        "optional-features-changed": (GObject.SignalFlags.RUN_LAST, None, ()),
    }

    def refresh_retrieval_settings(self):
        self.emit("retrieval-settings-changed")

    def refresh_optional_features(self):
        self.emit("optional-features-changed")
        self._update_setup_highlights()
        self.refresh_model_configuration()
        stage = self.controller.optional_features_stage
        self.set_background_activity(
            "optional_features", f"{stage} optional features" if stage else "",
            priority=30 if stage == "Downloading" else 20,
        )

    def __init__(self, app, controller):
        super().__init__(application=app, title="Lopaca")
        self.add_css_class("lopaca-main")
        self.controller, self.chat = controller, None
        self.busy = False
        self.follow_output = True
        self.current_wrapper = self.current_answer = self.current_thinking = None
        self.context_capacity = self.context_used = self.tokens_per_second = None
        self._review_pending = False
        self._earlier_prose = None
        self._history_start = 0
        self._prepend_upper = None
        self._history_load_id = None
        self._stream_lock = threading.Lock()
        self._incoming_stream = {}
        self._stream_updates = {}
        self._date_display = None
        self._stream_update_id = None
        self._first_answer_shown = self._first_thinking_shown = False
        self._thinking_pending = []
        self._thinking_flush_id = None
        self._inband_thinking = ""
        self._activity_status = "Not connected"
        self._model_activity = ""
        self._model_server_activity = ""
        self._model_setup_downloading = False
        self._model_download = {}
        self._model_switch_timer = None
        self._model_setup_dialog = self._model_remove_dialog = None
        self._welcome_pending = False
        self._app_settings_dialog = None
        self._background_activities = {}
        self._notices = deque()
        self._notification_timer = None
        self._notification_deadline = 0
        self._notification_model_busy = False
        self._running_commands = 0
        self._condensing_context = False
        self._response_row = None
        self._response_chat_id = None
        self._scroll_idle_id = None
        self._history_reveal_id = None
        self._chat_rows = []
        self._dragged_row = None
        self._drag_pointer_y = None
        self._drag_scroll_id = None
        self.set_default_size(1180, 790)
        self.set_size_request(760, 520)
        self._apply_css()
        self._build_ui()
        controller.attach(self)
        self._welcome_pending = controller.first_launch
        self.connect("map", self._start_date_timer)
        self.connect("map", lambda *_: self.update_first_launch())
        self.connect("unmap", lambda *_: GLib.source_remove(self._date_timer))
        self.chat_list.add_tick_callback(self._scroll_initial_chat)

    def _start_date_timer(self, *_):
        self._refresh_message_dates()
        self._date_timer = GLib.timeout_add_seconds(60, self._refresh_message_dates)

    def _apply_css(self):
        provider = Gtk.CssProvider()
        # Resolve the relative font size first, then align baselines to pixels.
        css = CSS
        if Gtk.check_version(4, 16, 0) is None:
            css = css.replace("line-height: 1.4em", "line-height: round(1.4em, 1px)")
        provider.load_from_string(css)
        Gtk.StyleContext.add_provider_for_display(
            Gdk.Display.get_default(), provider, Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION
        )

    def _clear_message_selection(self, _gesture, _presses, x, y):
        clicked = self.pick(x, y, Gtk.PickFlags.DEFAULT)
        while clicked is not None and not clicked.has_css_class("message"):
            clicked = clicked.get_parent()
        pending = [self.transcript]
        while pending:
            widget = pending.pop()
            if widget is clicked:
                continue
            if (
                isinstance(widget, Gtk.Label)
                and widget.get_selectable()
                and widget.get_selection_bounds()[0]
            ):
                widget.select_region(0, 0)
            child = widget.get_first_child()
            while child is not None:
                pending.append(child)
                child = child.get_next_sibling()

    def _build_ui(self):
        clicks = Gtk.GestureClick()
        clicks.set_button(0)
        clicks.set_propagation_phase(Gtk.PropagationPhase.CAPTURE)
        clicks.connect("pressed", self._clear_message_selection)
        self.add_controller(clicks)
        toolbar = Adw.ToolbarView()
        toolbar.add_css_class("app-toolbar")
        header_bar = Adw.HeaderBar()
        header_bar.add_css_class("app-header-bar")
        header_bar.add_css_class("app-titlebar")
        title = Gtk.Label(label="Lopaca")
        title.add_css_class("app-title")
        title.add_css_class("title")
        header_bar.set_title_widget(title)
        toolbar.add_top_bar(header_bar)
        root = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
        root.add_css_class("app-layout")
        toolbar.set_content(root)
        self.set_content(toolbar)
        sidebar = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        sidebar.add_css_class("sidebar")
        sidebar.set_hexpand(False)
        root.append(sidebar)
        top = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
        top.add_css_class("sidebar-header")
        heading = Gtk.Label(label="Chats", xalign=0)
        heading.add_css_class("chats-heading")
        heading.set_hexpand(True)
        heading.add_css_class("title-2")
        top.append(heading)
        new_chat_button = Gtk.Button(label="New chat", icon_name="list-add-symbolic")
        new_chat_button.add_css_class("new-chat-button")
        new_chat_button.add_css_class("suggested-action")
        new_chat_button.set_tooltip_text("New Chat")
        new_chat_button.connect("clicked", lambda *_: self.controller.new_chat())
        top.append(new_chat_button)
        sidebar.append(top)
        self.chat_list = Gtk.ListBox()
        self.chat_list.add_css_class("main-chat-list")
        self.chat_list.set_selection_mode(Gtk.SelectionMode.SINGLE)
        self.chat_list.set_activate_on_single_click(True)
        self.chat_list.add_css_class("workspace-list")
        self.chat_list.connect(
            "row-activated", lambda _list, row: self.controller.open_chat(row.chat_id)
        )
        self.chat_list.set_header_func(self._set_chat_header)
        drop_target = Gtk.DropTarget.new(str, Gdk.DragAction.MOVE)
        drop_target.connect("enter", self._chat_drag_motion)
        drop_target.connect("motion", self._chat_drag_motion)
        drop_target.connect("leave", lambda *_: self._clear_drop_indicator())
        drop_target.connect("drop", self._chat_drop)
        self.chat_list.add_controller(drop_target)
        list_overlay = Gtk.Overlay()
        list_overlay.add_css_class("chat-list-overlay")
        list_overlay.set_child(self.chat_list)
        self._drop_indicator = Gtk.Box()
        self._drop_indicator.add_css_class("chat-drop-indicator")
        self._drop_indicator.set_valign(Gtk.Align.START)
        self._drop_indicator.set_can_target(False)
        self._drop_indicator.set_visible(False)
        list_overlay.add_overlay(self._drop_indicator)
        self.chat_scroll = Gtk.ScrolledWindow()
        self.chat_scroll.add_css_class("chat-list-scroll")
        self.chat_scroll.add_css_class("undershoot-top")
        self.chat_scroll.add_css_class("undershoot-bottom")
        self.chat_scroll.set_child(list_overlay)
        self.chat_scroll.set_vexpand(True)
        sidebar.append(self.chat_scroll)
        footer = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
        footer.add_css_class("sidebar-footer")
        usage = Gtk.CenterBox()
        usage.add_css_class("disk-usage")
        usage.add_css_class("muted")
        usage.set_hexpand(True)
        self.chat_usage = Gtk.Label(label="Chat: —", xalign=1)
        self.chat_usage.add_css_class("chat-disk-usage")
        self.chat_usage.set_hexpand(True)
        self.total_usage = Gtk.Label(label="Total: —", xalign=0)
        self.total_usage.add_css_class("total-disk-usage")
        self.total_usage.set_hexpand(True)
        usage.set_start_widget(self.chat_usage)
        usage.set_center_widget(Gtk.Label(label=" | "))
        usage.set_end_widget(self.total_usage)
        footer.append(usage)
        self._settings_button = Gtk.Button(
            icon_name="emblem-system-symbolic", tooltip_text="App Settings"
        )
        self._settings_button.add_css_class("app-settings-button")
        self._settings_button.connect("clicked", lambda *_: self._show_settings())
        footer.append(self._settings_button)
        sidebar.append(footer)
        root.append(Gtk.Separator(orientation=Gtk.Orientation.VERTICAL))
        main = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        main.add_css_class("chat-pane")
        main.set_hexpand(True)
        root.append(main)
        header = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
        header.add_css_class("chat-header")
        column = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        column.add_css_class("chat-header-titles")
        column.set_hexpand(True)
        self.chat_title = Gtk.Label(label="New chat", xalign=0)
        self.chat_title.add_css_class("chat-title")
        self.chat_title.add_css_class("title-3")
        column.append(self.chat_title)
        self.folder_label = Gtk.Label(label="No documents folder", xalign=0)
        self.folder_label.add_css_class("chat-folder-path")
        self.folder_label.set_ellipsize(Pango.EllipsizeMode.END)
        self.folder_label.add_css_class("muted")
        column.append(self.folder_label)
        header.append(column)
        main.append(header)
        self.transcript = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        self.transcript.add_css_class("transcript")
        self.transcript_scroll = Gtk.ScrolledWindow()
        self.transcript_scroll.add_css_class("main-transcript-scroll")
        self.transcript_scroll.add_css_class("transcript-scroll")
        self.transcript_scroll.add_css_class("undershoot-bottom")
        self.transcript_scroll.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        self.transcript_scroll.set_child(TranscriptViewport(child=self.transcript))
        self.transcript_scroll.set_vexpand(True)
        main.append(self.transcript_scroll)
        adjustment = self.transcript_scroll.get_vadjustment()
        self._scroll_guard = False
        adjustment.connect("value-changed", self._adjustment_value_changed)
        adjustment.connect("changed", self._adjustment_changed)
        self._sync_native_scroll(self.transcript_scroll)
        compose = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        compose.add_css_class("composer")
        status_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
        status_row.add_css_class("composer-status")
        activities = Gtk.Box()
        activities.add_css_class("composer-activity")
        self.status = StatusLabel(reserve_dots=True)
        self.status.display(self._activity_status)
        self._status_separator = Gtk.Label(label="|", visible=False)
        self._status_separator.add_css_class("muted")
        self._background_status = StatusLabel()
        self._background_status.set_visible(False)
        for widget in (self.status, self._status_separator, self._background_status):
            activities.append(widget)
        status_row.append(activities)
        status_spacer = Gtk.Box()
        status_spacer.add_css_class("status-spacer")
        status_spacer.set_hexpand(True)
        status_row.append(status_spacer)
        self.context_status = Gtk.Label(label="Context: —/—", xalign=1)
        self.context_status.add_css_class("context-usage")
        self.context_status.add_css_class("muted")
        status_row.append(self.context_status)
        self.speed_status = Gtk.Label(label="— t/s", xalign=1)
        self.speed_status.add_css_class("generation-speed")
        self.speed_status.add_css_class("muted")
        status_row.append(self.speed_status)
        # Match one rendered space at the current font size.
        self.speed_status.set_margin_end(
            self.speed_status.create_pango_layout(" ").get_pixel_size()[0]
        )
        compose.append(status_row)
        row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
        row.add_css_class("composer-frame-row")
        compose.append(row)
        frame = Gtk.Frame()
        frame.add_css_class("main-frame")
        frame.add_css_class("composer-frame")
        frame.set_hexpand(True)
        row.append(frame)
        inner = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        inner.add_css_class("composer-content")
        frame.set_child(inner)
        self.prompt = Gtk.TextView()
        self.prompt.add_css_class("prompt-view")
        self.prompt.set_wrap_mode(Gtk.WrapMode.WORD_CHAR)
        self.prompt.set_accepts_tab(False)
        _, line_height = self.prompt.create_pango_layout("Ag").get_pixel_size()
        prompt_height = max(1, line_height) * 4
        self.prompt.set_size_request(-1, prompt_height)
        self.prompt.get_buffer().connect("changed", lambda *_: self.update_send_button())
        key_controller = Gtk.EventControllerKey.new()
        key_controller.connect("key-pressed", self._key_pressed)
        self.prompt.add_controller(key_controller)
        prompt_scroll = Gtk.ScrolledWindow()
        prompt_scroll.add_css_class("prompt-scroll")
        prompt_scroll.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        prompt_scroll.set_child(self.prompt)
        prompt_scroll.set_min_content_height(prompt_height)
        prompt_scroll.set_max_content_height(1160)
        inner.append(prompt_scroll)
        self.send_button = Gtk.Button(icon_name="go-up-symbolic", tooltip_text="Send")
        self.send_button.add_css_class("send-stop")
        self.send_button.add_css_class("suggested-action")
        self.send_button.set_valign(Gtk.Align.END)
        self.send_button.connect("clicked", self._send_clicked)
        actions = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
        actions.add_css_class("composer-actions")
        hint = Gtk.Label(
            label="Shift+Enter for new line · Enter to send",
            xalign=0,
            ellipsize=Pango.EllipsizeMode.END,
        )
        hint.add_css_class("prompt-hint")
        hint.set_hexpand(True)
        hint.add_css_class("muted")
        actions.append(hint)
        actions.append(self._build_model_picker())
        self.thinking_label = Gtk.Label(label="Thinking budget")
        self.thinking_label.add_css_class("thinking-label")
        self.thinking_label.add_css_class("muted")
        actions.append(self.thinking_label)
        self.thinking_dropdown = Gtk.DropDown.new_from_strings(["auto"])
        self.thinking_dropdown.add_css_class("thinking-dropdown")
        self.thinking_dropdown.set_valign(Gtk.Align.CENTER)
        self._thinking_values = [None]
        self._thinking_handler = self.thinking_dropdown.connect(
            "notify::selected", self._thinking_selected
        )
        actions.append(self.thinking_dropdown)
        attach = Gtk.Button(icon_name=ATTACH_DOCUMENTS_ICON, tooltip_text="Attach documents")
        attach.add_css_class("attach-documents-button")
        attach.connect("clicked", lambda *_: self._show_document_drop(self.chat["id"], attach=True))
        actions.append(attach)
        actions.append(self.send_button)
        inner.append(actions)
        main.append(compose)
        self.update_send_button()

    def _build_model_picker(self):
        self._model_picker = Gtk.Box(visible=False)
        self._model_picker.add_css_class("model-picker")
        model_label = Gtk.Label(label="Model:")
        model_label.add_css_class("muted")
        self._model_picker.append(model_label)
        self._model_dropdown = Gtk.DropDown(valign=Gtk.Align.CENTER)
        self._model_dropdown.set_list_factory(self._model_dropdown.get_factory())
        role_factory = Gtk.SignalListItemFactory()
        role_factory.connect("setup", lambda _, row: row.set_child(Gtk.Label(xalign=0)))
        role_factory.connect(
            "bind",
            lambda _, row: row.get_child().set_text(
                row.get_item().get_string().split(":", 1)[0].lower()
            ),
        )
        self._model_dropdown.set_factory(role_factory)
        self._model_keys = ()
        self._model_labels = ()
        self._model_handler = self._model_dropdown.connect("notify::selected", self._model_selected)
        self._model_picker.append(self._model_dropdown)
        return self._model_picker

    def _update_model_picker(self):
        keys = self.controller.installed_models()
        self._model_picker.set_visible(len(keys) > 1)
        if len(keys) < 2:
            return
        rule = MODEL_RECOMMENDATIONS[self.controller.model_recommendation()]
        roles = {}
        for role in ("balanced", "faster", "better"):
            roles.setdefault(getattr(rule, role), role.title())
        entries = self.controller.custom_model_files().get("model", {})
        keys = tuple(key for key in roles if key in keys) + self.controller.custom_models_in_order(
            (key for key in keys if key not in roles), entries
        )
        labels = tuple(
            f"{roles.get(key, 'Custom')}: {asset.name} · {asset.quantization}"
            if (asset := MODEL_DOWNLOADS.get(key)) else
            f"Custom: {entries[key]['name']}"
            + (f" · {entries[key]['quantization']}" if entries[key]["quantization"] else "")
            for key in keys
        )
        config = self.controller.model_configuration()
        key = self.controller.model_switch_target
        if key is None and config:
            path = Path(config.get("model") or ".").expanduser()
            key = next((key for key in keys if self.controller.installed_model_path(key) == path),
                       None)
        with self._model_dropdown.handler_block(self._model_handler):
            if keys != self._model_keys or labels != self._model_labels:
                self._model_dropdown.set_model(Gtk.StringList.new(labels))
                self._model_keys = keys
                self._model_labels = labels
            self._model_dropdown.set_selected(
                keys.index(key) if key in keys else Gtk.INVALID_LIST_POSITION
            )
        self._model_dropdown.set_sensitive(
            config is not None
            and not self.controller.switching_model
            and not self.controller.model_server.busy
        )
        if self.controller.switching_model:
            tooltip = "Loading the selected model"
        elif self.controller.model_switch_target is not None:
            tooltip = "Switching after the current model request finishes"
        elif config is None:
            tooltip = "The model is managed by an external server"
        else:
            tooltip = "Switch models"
        self._model_dropdown.set_tooltip_text(tooltip)

    def _model_selected(self, dropdown, _param):
        index = dropdown.get_selected()
        if index >= len(self._model_keys):
            return
        self.controller.request_model_switch(self._model_keys[index])

    def schedule_model_switch(self):
        if self._model_switch_timer is None and self.controller.try_model_switch():
            self._model_switch_timer = GLib.timeout_add(100, self._retry_model_switch)

    def _retry_model_switch(self):
        if self.controller.try_model_switch():
            return GLib.SOURCE_CONTINUE
        self._model_switch_timer = None
        return GLib.SOURCE_REMOVE

    def update_thinking_options(self):
        key, title, choices = self.controller.thinking_choices()
        self.thinking_label.set_text(title)
        self._thinking_values = [value for value, _ in choices]
        current = (
            self.controller.get_chat_settings(self.controller.chat["id"]).get(key)
            if self.controller.chat
            else None
        )
        with self.thinking_dropdown.handler_block(self._thinking_handler):
            self.thinking_dropdown.set_model(Gtk.StringList.new([label for _, label in choices]))
            self.thinking_dropdown.set_selected(
                self._thinking_values.index(current) if current in self._thinking_values else 0
            )
        self.thinking_dropdown.set_tooltip_text(title + " auto lets model choose for each prompt")

    def _thinking_selected(self, dropdown, _param):
        selected = dropdown.get_selected()
        if selected >= len(self._thinking_values):
            return
        try:
            self.controller.set_thinking(self._thinking_values[selected])
        except OSError as exc:
            self.update_thinking_options()
            self._show_alert("Could not save thinking setting", str(exc))

    def _store_stream_update(self, pending, callback, args):
        if callback == self.controller._update_inference_stats and callback in pending:
            args = tuple(
                new if new is not None else old
                for new, old in zip(args, pending[callback], strict=True)
            )
        pending[callback] = args

    def dispatch(self, callback, *args):
        if callback in (
            self.update_answer,
            self.update_thinking,
            self.controller._update_inference_stats,
        ):
            # Coalesce before crossing into GTK. A fast server otherwise queues
            # one idle callback (and a full answer snapshot) per token.
            with self._stream_lock:
                if not self._incoming_stream:
                    GLib.idle_add(self._drain_stream)
                if callback == self.update_thinking:
                    self._incoming_stream.setdefault(callback, []).append(args[0])
                else:
                    self._store_stream_update(self._incoming_stream, callback, args)
        elif callback == self.controller.finish_response:
            GLib.idle_add(self._finish_stream_update, callback, args)
        else:
            GLib.idle_add(callback, *args)

    def _drain_stream(self):
        with self._stream_lock:
            incoming, self._incoming_stream = self._incoming_stream, {}
        for callback, args in incoming.items():
            if callback == self.update_thinking:
                callback("".join(args))
            else:
                self._queue_stream_update(callback, args)
        return False

    def _queue_stream_update(self, callback, args):
        if callback == self.update_answer and not self._first_answer_shown:
            visible, reasoning, _ = self.controller.split_thinking(args[0])
            self._first_answer_shown = bool(visible.strip())
            if self._first_answer_shown or (
                not self._first_thinking_shown and reasoning and reasoning.strip()
            ):
                self._stream_updates.pop(callback, None)
                return callback(*args)
        self._store_stream_update(self._stream_updates, callback, args)
        if self._stream_update_id is None:
            self._stream_update_id = GLib.timeout_add(20, self._flush_stream_updates)
        return False

    def _flush_stream_updates(self):
        self._stream_update_id = None
        updates, self._stream_updates = self._stream_updates, {}
        for callback, args in updates.items():
            callback(*args)
        return False

    def _finish_stream_update(self, callback, args):
        if self._stream_update_id is not None:
            GLib.source_remove(self._stream_update_id)
        self._flush_stream_updates()
        callback(*args)
        return False

    def prompt_text(self, strip=True):
        buffer = self.prompt.get_buffer()
        start, end = buffer.get_bounds()
        text = buffer.get_text(start, end, True)
        return text.strip() if strip else text

    def set_prompt(self, text):
        self.prompt.get_buffer().set_text(text)

    def set_status(self, text):
        self._activity_status = text
        if not self.controller.busy and text in ("Ready", "Disconnected", "Not connected"):
            if self._model_activity == "Cancelling":
                self._model_activity = ""
            if (
                self._notification_timer is not None
                and self._notification_model_busy
                and self._notification_deadline - GLib.get_monotonic_time() > 1_000_000
            ):
                GLib.source_remove(self._notification_timer)
                self._start_notice_timer(1000)
        self._refresh_model_status()

    def set_model_activity(self, text, active=True):
        if active:
            self._model_activity = text
        elif self._model_activity == text:
            self._model_activity = ""
        self._refresh_model_status()

    def _refresh_model_status(self):
        text = self._model_server_activity or self._model_activity or self._activity_status
        self.status.display(text, animated=text not in ("Ready", "Disconnected", "Not connected"))

    def set_context_processing(self, active):
        self._condensing_context = active
        self.set_model_activity("Condensing context", active)
        self.update_send_button()

    def set_background_activity(self, source, text, priority):
        # Keep obscured jobs so the next one is visible as soon as the current one finishes.
        if text:
            self._background_activities[source] = priority, text
        else:
            self._background_activities.pop(source, None)
        self._refresh_background_status()

    def _refresh_background_status(self):
        if self._notices:
            text = self._notices[0]
        else:
            _, text = max(
                self._background_activities.values(), key=lambda activity: activity[0],
                default=(0, ""),
            )
        self._status_separator.set_visible(bool(text))
        self._background_status.set_visible(bool(text))
        self._background_status.display(text, animated=bool(text) and not self._notices)

    def notify(self, text):
        if not self._notices or self._notices[-1] != text:
            self._notices.append(text)
        if self._notification_timer is None:
            self._start_notice_timer()
            self._refresh_background_status()
        self.update_send_button()

    def _start_notice_timer(self, duration=3000):
        self._notification_model_busy = self.controller.busy
        self._notification_deadline = GLib.get_monotonic_time() + duration * 1000
        self._notification_timer = GLib.timeout_add(duration, self._expire_notice)

    def _expire_notice(self):
        self._notices.popleft()
        self._notification_timer = None
        if self._notices:
            self._start_notice_timer()
        self._refresh_background_status()
        return GLib.SOURCE_REMOVE

    def set_context_capacity(self, capacity):
        self.context_capacity = capacity
        self._refresh_inference_stats()

    def update_inference_stats(self, context_used, tokens_per_second):
        if context_used is not None:
            self.context_used = context_used
        if tokens_per_second is not None:
            self.tokens_per_second = tokens_per_second
        self._refresh_inference_stats()
        return False

    def _refresh_inference_stats(self):
        context_used = "—" if self.context_used is None else str(self.context_used)
        capacity = "—" if self.context_capacity is None else str(self.context_capacity)
        percent = (
            f" ({100 * self.context_used / self.context_capacity:.1f}%)"
            if self.context_used is not None and self.context_capacity
            else ""
        )
        self.context_status.set_text(f"Context: {context_used}/{capacity}{percent}")
        tokens_per_second = (
            "—" if self.tokens_per_second is None else f"{self.tokens_per_second:.1f}"
        )
        self.speed_status.set_text(f"{tokens_per_second} t/s")

    def set_chat_title(self, title):
        self.chat_title.set_text(title)

    def set_disk_usage(self, chat_size, total_size):
        def format_size(size):
            if size is None:
                return "—"
            for unit in ("B", "KB", "MB", "GB"):
                if size < 1024:
                    return f"{size:.0f}{unit}"
                size /= 1024
            return f"{size:.0f}TB"

        self.chat_usage.set_text(f"Chat: {format_size(chat_size)}")
        self.total_usage.set_text(f"Total: {format_size(total_size)}")

    def update_send_button(self):
        can_send = bool(self.prompt_text()) or self.controller.has_draft_attachments()
        can_stop = (
            self.busy
            and not self.controller._generation_answer_complete
            and not self.controller._condensing_context
        )
        self.send_button.set_sensitive(can_send or can_stop)
        if self.busy:
            steering = (
                self.controller._generation_streaming
                and self.controller.model_switch_target is None
                and can_stop
                and not self.controller.stop_requested
                and self.chat["id"] == self._response_chat_id
            )
            self.send_button.set_icon_name(
                "go-up-symbolic" if can_send else "window-close-symbolic"
            )
            self.send_button.set_tooltip_text(
                ("Interrupt and send" if steering else "Queue message")
                if can_send
                else "Stop generation"
            )
        else:
            self.send_button.set_tooltip_text(
                "Queue message" if self.controller.model_switch_target is not None else "Send"
            )

    def _send_clicked(self, _button):
        if self.busy and not self.prompt_text() and not self.controller.has_draft_attachments():
            self.controller.stop()
        else:
            self._send()

    def _send(self):
        if self.controller.send(self.prompt_text()):
            self.set_prompt("")
            self.update_send_button()

    def _key_pressed(self, _controller, key, _code, state):
        if key in (Gdk.KEY_Return, Gdk.KEY_KP_Enter) and not state & Gdk.ModifierType.SHIFT_MASK:
            self._send()
            return True
        return False

    def _is_near_bottom(self, adjustment):
        return adjustment.get_upper() - adjustment.get_page_size() - adjustment.get_value() <= 1

    def _sync_native_scroll(self, widget):
        if isinstance(widget, Gtk.ScrolledWindow) and not getattr(widget, "_scroll_sync", False):
            widget._scroll_sync = True
            wheel = Gtk.EventControllerScroll.new(Gtk.EventControllerScrollFlags.VERTICAL)
            wheel.set_propagation_phase(Gtk.PropagationPhase.CAPTURE)
            wheel.connect("scroll", self._prepare_native_scroll)
            widget.add_controller(wheel)
        child = widget.get_first_child()
        while child:
            self._sync_native_scroll(child)
            child = child.get_next_sibling()

    def _prepare_native_scroll(self, controller, _dx, _dy):
        # GTK ignores adjustment changes during overshoot deceleration. Cancel
        # that animation, then resync its unclamped position with the scrollbar.
        # Propagate the event so GTK supplies the native scrolling and edge glow.
        scroll = controller.get_widget()
        kinetic = scroll.get_kinetic_scrolling()
        if kinetic:
            scroll.set_kinetic_scrolling(False)
        scroll.get_vadjustment().emit("value-changed")
        if kinetic:
            scroll.set_kinetic_scrolling(True)
        return False

    def _adjustment_value_changed(self, adjustment):
        first = self.transcript.get_first_child()
        if first is not None and hasattr(first, "message_date") and first.get_opacity() == 0:
            row = first.get_next_sibling()
            if row is not None and adjustment.get_value() > first.get_height() + row.get_height():
                first.set_opacity(1)
        if not self._scroll_guard:
            self.follow_output = self._is_near_bottom(adjustment)
            self._schedule_older_history(adjustment)

    def _schedule_older_history(self, adjustment):
        if (
            self._history_load_id is None
            and self._prepend_upper is None
            and self._history_reveal_id is None
            and (self._history_start or self._earlier_prose)
            and not self.follow_output
            and adjustment.get_value() < adjustment.get_page_size()
        ):
            # Keep work out of the wheel handler. Wait for GTK to lay out
            # each prepend before considering another.
            self._history_load_id = GLib.idle_add(
                self._prepend_history, adjustment, priority=GLib.PRIORITY_LOW
            )

    def _adjustment_changed(self, adjustment):
        if self._prepend_upper is not None and adjustment.get_upper() > self._prepend_upper:
            shift = adjustment.get_upper() - self._prepend_upper
            self._set_scroll_value(adjustment, adjustment.get_value() + shift)
            self._prepend_upper = None
            self._schedule_older_history(adjustment)
        elif self.follow_output:
            # Anchor the bottom as GTK lays out newly loaded images and text.
            self._set_scroll_value(adjustment, adjustment.get_upper() - adjustment.get_page_size())

    def show_chat(self, chat, folder):
        # Release the sidebar row before refresh removes it and GTK moves focus.
        self.set_focus(self.prompt)
        self.chat = chat
        self.set_chat_title(chat["title"])
        self.folder_label.set_text(f"Chat folder: {folder}")
        self.render_history(chat["messages"])
        if self.busy and chat["id"] == self._response_chat_id:
            self._append_message_day(self._response_row.timestamp)
            self.transcript.append(self._response_row)
            self.scroll_bottom(True)
        stats = chat.get("inference_stats", {})
        self.context_used = stats.get("context_used")
        self.tokens_per_second = None
        self._refresh_inference_stats()

    def show_sidebar(self, chats, active_id):
        self._clear_drop_indicator()
        signature = (
            self.controller.chat_sort,
            tuple(
                (chat["id"], chat.get("title", "New chat"), chat.get("pinned", False))
                for chat in chats
            ),
        )
        if signature == getattr(self, "_sidebar_signature", None):
            self._dragged_row = None
            for row in self._chat_rows:
                active = row.chat_id == active_id
                if active:
                    row.add_css_class("active-chat")
                    self.chat_list.select_row(row)
                else:
                    row.remove_css_class("active-chat")
            return
        self._sidebar_signature = signature
        self._chat_rows = []
        self._dragged_row = None
        while (child := self.chat_list.get_first_child()) is not None:
            self.chat_list.remove(child)
        active = None
        for chat in chats:
            row = Gtk.ListBoxRow()
            row.add_css_class("chat-list-row")
            row.chat_id = chat["id"]
            line = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
            line.add_css_class("chat-row")
            pinned = chat.get("pinned", False)
            row.pinned = pinned
            if chat["id"] == active_id:
                row.add_css_class("active-chat")
                active = row
            label = Gtk.Label(label=chat.get("title", "New chat"), xalign=0)
            label.add_css_class("chat-list-label")
            label.set_ellipsize(Pango.EllipsizeMode.END)
            label.set_hexpand(True)
            line.append(label)
            label.set_max_width_chars(28)
            options_button = Gtk.MenuButton(icon_name="view-more-symbolic")
            options_button.add_css_class("chat-list-options-button")
            options_button.add_css_class("chat-options-button")
            options_button.set_tooltip_text("Chat Options")
            menu = Gio.Menu()
            actions = Gio.SimpleActionGroup()
            choices = (
                ("Rename Chat", lambda chat_id=chat["id"]: self._rename_chat(chat_id)),
                (
                    "Unpin Chat" if pinned else "Pin Chat",
                    lambda chat_id=chat["id"], value=not pinned: self.controller.set_pinned(
                        chat_id, value
                    ),
                ),
                ("Open Folder", lambda chat_id=chat["id"]: self.controller.open_folder(chat_id)),
                ("Add Documents", lambda chat_id=chat["id"]: self._show_document_drop(chat_id)),
                ("Chat Settings", lambda chat_id=chat["id"]: self._show_settings(chat_id)),
                ("Delete Chat", lambda chat_id=chat["id"]: self._confirm_delete_chat(chat_id)),
            )
            for index, (title, callback) in enumerate(choices):
                if index in (0, 2, 4):
                    section = Gio.Menu()
                    menu.append_section(None, section)
                action = Gio.SimpleAction.new(f"option{index}", None)
                action.connect(
                    "activate", lambda _action, _parameter, callback=callback: callback()
                )
                actions.add_action(action)
                section.append(title, f"chat.option{index}")
            options_button.insert_action_group("chat", actions)
            popover = Gtk.PopoverMenu.new_from_model(menu)
            popover.add_css_class("chat-options-menu")
            options_button.set_popover(popover)
            popover.set_halign(Gtk.Align.START)
            popover.set_has_arrow(False)
            popover.connect("closed", lambda menu: menu.set_pointing_to(None))
            line.append(options_button)
            row.set_child(line)
            right_click = Gtk.GestureClick(button=Gdk.BUTTON_SECONDARY)
            right_click.set_propagation_phase(Gtk.PropagationPhase.CAPTURE)
            right_click.connect(
                "pressed",
                lambda gesture, *_: gesture.set_state(Gtk.EventSequenceState.CLAIMED),
            )

            def open_options(gesture, _count, x, y, button=options_button):
                point = Graphene.Point().init(x, y)
                success, anchor = gesture.get_widget().compute_point(button, point)
                if success:
                    rect = Gdk.Rectangle()
                    rect.x, rect.y = round(anchor.x), round(anchor.y)
                    rect.width = rect.height = 1
                    button.get_popover().set_pointing_to(rect)
                button.popup()

            right_click.connect("released", open_options)
            row.add_controller(right_click)
            self.chat_list.append(row)
            self._chat_rows.append(row)
            if pinned or self.controller.chat_sort == "user_selectable":
                drag_source = Gtk.DragSource.new()
                drag_source.set_actions(Gdk.DragAction.MOVE)
                drag_source.connect("prepare", self._prepare_chat_drag, row)
                drag_source.connect(
                    "drag-begin",
                    lambda source, _drag, row=row: source.set_icon(
                        Gtk.WidgetPaintable.new(row), 0, 0
                    ),
                )
                drag_source.connect("drag-end", self._end_chat_drag)
                label.add_controller(drag_source)
                label.set_tooltip_text(
                    "Drag to reorder pinned chats" if pinned else "Drag to reorder chats"
                )
        if active:
            self.chat_list.select_row(active)

    def _scroll_initial_chat(self, _widget, _clock):
        row = self.chat_list.get_selected_row()
        if row is None:
            return False
        adjustment = self.chat_scroll.get_vadjustment()
        if row.get_height() <= 0 or adjustment.get_page_size() <= 0:
            return True
        page_size = adjustment.get_page_size()
        limit = max(adjustment.get_lower(), adjustment.get_upper() - page_size)
        # Start at the top; scroll only enough to reveal a selected row below the viewport.
        target = row.get_allocation().y + row.get_height() - page_size
        adjustment.set_value(max(adjustment.get_lower(), min(limit, target)))
        return False

    def _set_chat_header(self, row, previous):
        divider = None
        if previous is not None and previous.pinned and not row.pinned:
            divider = Gtk.Separator(orientation=Gtk.Orientation.HORIZONTAL)
            divider.add_css_class("chat-list-divider")
        row.set_header(divider)

    def _prepare_chat_drag(self, _source, _x, _y, row):
        self._dragged_row = row
        return Gdk.ContentProvider.new_for_value(row.chat_id)

    def _end_chat_drag(self, *_):
        self._dragged_row = None
        self._clear_drop_indicator()

    def _chat_drop_position(self, y):
        if self._dragged_row is None:
            return None, False
        rows = [row for row in self._chat_rows if row.pinned == self._dragged_row.pinned]
        for row in rows:
            if y < row.get_allocation().y + row.get_height() / 2:
                return row, False
        # Empty space below the rows is also a drop area for the final position.
        return (rows[-1], True) if rows else (None, False)

    def _chat_drag_motion(self, _target, _x, y):
        row, after = self._chat_drop_position(y)
        if row is None:
            return Gdk.DragAction(0)
        # Keep viewport coordinates stable while the list moves under the pointer.
        self._drag_pointer_y = y - self.chat_scroll.get_vadjustment().get_value()
        if self._drag_scroll_id is None:
            self._drag_scroll_id = GLib.timeout_add(30, self._scroll_chat_drag)
        top = row.get_allocation().y + (row.get_height() - 3 if after else 0)
        self._drop_indicator.set_margin_top(max(0, top))
        self._drop_indicator.set_visible(True)
        return Gdk.DragAction.MOVE

    def _scroll_chat_drag(self):
        if self._dragged_row is None or self._drag_pointer_y is None:
            self._drag_scroll_id = None
            return False
        adjustment = self.chat_scroll.get_vadjustment()
        page_size = adjustment.get_page_size()
        pinned_height = (
            sum(row.get_height() for row in self._chat_rows if row.pinned)
            if self._dragged_row.pinned
            else None
        )
        if pinned_height is not None and pinned_height <= page_size:
            return True
        edge = min(40, page_size / 3)
        if edge <= 0:
            return True
        if self._drag_pointer_y < edge:
            step = -16 * min(1, (edge - self._drag_pointer_y) / edge)
        elif self._drag_pointer_y > page_size - edge:
            step = 16 * min(1, (self._drag_pointer_y - page_size + edge) / edge)
        else:
            return True
        value = adjustment.get_value()
        limit = max(adjustment.get_lower(), adjustment.get_upper() - page_size)
        if pinned_height is not None:
            # Include the divider and its spacing, ending before the first unpinned row.
            pinned_end = next(
                (row.get_allocation().y for row in self._chat_rows if not row.pinned),
                pinned_height,
            )
            limit = min(limit, pinned_end - page_size)
        next_value = max(adjustment.get_lower(), min(limit, value + step))
        if next_value != value:
            adjustment.set_value(next_value)
            self._chat_drag_motion(None, 0, next_value + self._drag_pointer_y)
        return True

    def _clear_drop_indicator(self):
        self._drop_indicator.set_visible(False)
        if self._drag_scroll_id is not None:
            GLib.source_remove(self._drag_scroll_id)
            self._drag_scroll_id = None
        self._drag_pointer_y = None

    def _chat_drop(self, _target, chat_id, _x, y):
        row, after = self._chat_drop_position(y)
        self._clear_drop_indicator()
        if self._dragged_row is None or chat_id != self._dragged_row.chat_id:
            return False
        return self.controller.move_chat(chat_id, row.chat_id, after) if row is not None else False

    def render_history(self, messages):
        if self._history_load_id is not None:
            GLib.source_remove(self._history_load_id)
            self._history_load_id = None
        if self._history_reveal_id is not None:
            self.transcript_scroll.remove_tick_callback(self._history_reveal_id)
        self.transcript_scroll.set_opacity(0)
        while (child := self.transcript.get_first_child()) is not None:
            self.transcript.remove(child)
        # A handful of very long replies can exceed a screenful many times over.
        self._history_start = len(messages)
        chars = 0
        for message in reversed(messages[-32:]):
            self._history_start -= 1
            chars += len(message.get("content", ""))
            if chars >= 6000:
                break
        self._prepend_upper = None
        self._earlier_prose = None
        for index, message in enumerate(messages[self._history_start :]):
            self._append_history_message(message, attach=True, page_prose=index == 0)
        self._refresh_message_dates()
        first = self.transcript.get_first_child()
        if first is not None:
            first.set_opacity(1)
        self.scroll_bottom(True)
        first_frame = True

        def reveal(scroll, _clock):
            nonlocal first_frame
            if first_frame:
                first_frame = False
                return True
            if self._scroll_idle_id is not None:
                return True
            scroll.set_opacity(1)
            self._history_reveal_id = None
            return False

        # Allow layout and its deferred scroll update before exposing the rebuilt viewport.
        self._history_reveal_id = self.transcript_scroll.add_tick_callback(reveal)

    def _prepend_history(self, adjustment):
        self._history_load_id = None
        if self.follow_output or adjustment.get_value() >= adjustment.get_page_size():
            return False
        # Keep the current rows in place on screen when older rows gain height.
        self._prepend_upper = adjustment.get_upper()
        if self._earlier_prose is not None:
            grew = next(self._earlier_prose, None)
            if grew is not None:
                if not grew:
                    self._prepend_upper = None
                    self._schedule_older_history(adjustment)
                return False
            self._earlier_prose = None
        if not self._history_start:
            self._prepend_upper = None
            return False
        self._history_start -= 1
        message = self.chat["messages"][self._history_start]
        timestamp = message["timestamp"]
        day = datetime.fromtimestamp(timestamp).date()
        header = self.transcript.get_first_child()
        if header.message_date != day:
            header = self._message_day_label(day)
            self.transcript.prepend(header)
        wrapper = self._append_history_message(message, attach=False, page_prose=True)
        self.transcript.insert_child_after(wrapper.message_row, header)
        return False

    def _append_history_message(self, message, *, attach, page_prose):
        role = message.get("role")
        text = message.get("content", "")
        earlier = ""
        if page_prose and role == "assistant" and len(text) > 16000:
            text = self.controller.visible_answer(text)
            # Fences, notes and reference definitions can affect later paragraphs.
            if not re.search(r"!\[|[`~<>]|^ {0,3}\[.+?\]:|^ {4}|^\t", text, re.MULTILINE):
                split = text.rfind("\n\n", 0, len(text) - 2000)
                if split > 0:
                    earlier, text = text[:split], text[split + 2 :]
        wrapper = self.append_message(
            role,
            text,
            message.get("reasoning_content", ""),
            message["timestamp"],
            live=False,
            attach=attach,
            attachments=message.get("attachments", ()),
        )
        if earlier:
            self._earlier_prose = self._older_prose_sections(wrapper, earlier)
        return wrapper

    @staticmethod
    def _older_prose_sections(wrapper, text):
        first_body = wrapper.get_first_child().get_next_sibling()
        if first_body.has_css_class("thinking"):
            first_body = first_body.get_next_sibling()
        after = first_body.get_prev_sibling()
        # Reveal from the nearest paragraph backwards. A large plain paragraph
        # prepares natural line breaks before adding its labels individually.
        for paragraph in reversed(text.split("\n\n")):
            if not paragraph.strip():
                continue
            section = MarkdownText()
            for grew in section.prepend_prose(paragraph, first_body.get_width()):
                if grew and section.get_parent() is None:
                    wrapper.insert_child_after(section, after)
                yield grew

    def _refresh_message_dates(self):
        now = datetime.now()
        clock_format = get_clock_format()
        display = (now.date(), now.astimezone().tzinfo, clock_format)
        # Each row already has its label. Revisit the transcript only when the
        # local day, time zone or desktop clock preference changes.
        if display == self._date_display:
            return True
        self._date_display = display
        child = self.transcript.get_first_child()
        while child is not None:
            if hasattr(child, "message_date"):
                child.set_text(message_day(child.message_date, now))
                if child.message_date != now.date():
                    child.set_opacity(1)
            elif hasattr(child, "timestamp"):
                child.time_label.set_text(message_time(child.timestamp, now, clock_format))
            child = child.get_next_sibling()
        return True

    def _append_message_day(self, timestamp):
        day = datetime.fromtimestamp(timestamp).date()
        previous = self.transcript.get_last_child()
        if previous is None or datetime.fromtimestamp(previous.timestamp).date() != day:
            if previous is not None:
                self.transcript.get_first_child().set_opacity(1)
            separator = self._message_day_label(day)
            separator.set_opacity(0 if previous is None else 1)
            self.transcript.append(separator)

    @staticmethod
    def _message_day_label(day):
        separator = Gtk.Label(label=message_day(day))
        separator.add_css_class("message-date-separator")
        separator.message_date = day
        separator.add_css_class("muted")
        # Add one measured text line to the fixed date spacing declared in CSS.
        separator.set_margin_bottom(separator.get_layout().get_pixel_size()[1])
        return separator

    def append_message(
        self, role, text, reasoning="", timestamp=None, *, live=True, attach=True, attachments=()
    ):
        if role == "assistant":
            text = self.controller.visible_answer(text)
        timestamp = datetime.now().timestamp() if timestamp is None else timestamp
        if attach:
            self._append_message_day(timestamp)
        if live and attach:
            self._refresh_message_dates()
        row = Gtk.Box()
        row.add_css_class("message-row")
        row.set_layout_manager(BubbleLayout(role == "user"))
        row.set_hexpand(True)
        wrapper = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        wrapper.add_css_class("message")
        wrapper.set_valign(Gtk.Align.START)
        wrapper.add_css_class("user-message" if role == "user" else "assistant-message")
        column = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        column.add_css_class("message-column")
        column.append(wrapper)
        time_label = Gtk.Label(label=message_time(timestamp))
        time_label.add_css_class("message-time-label")
        time_label.add_css_class("muted")
        time_label.add_css_class("message-time")
        time_label.set_halign(Gtk.Align.END if role == "user" else Gtk.Align.START)
        column.append(time_label)
        row.append(column)
        row.timestamp = timestamp
        row.time_label = time_label
        wrapper.message_row = row
        label = Gtk.Label(label="You" if role == "user" else "Assistant", xalign=0)
        label.add_css_class("message-label")
        label.add_css_class("message-role")
        wrapper.append(label)
        if reasoning:
            self._create_thinking_widget(wrapper, reasoning, True)
        if text:
            extractor = self.controller.commands_in if role == "assistant" else None
            images = self._reply_images(self.chat["id"]) if role == "assistant" else None
            append_markdown(wrapper, text, extractor, self._review_command, images=images)
        if role == "user" and attachments:
            wrapper.append(self._message_attachment_button(self.chat["id"], attachments))
        if attach:
            self.transcript.append(row)
        if role == "user" and attach and live:
            self.scroll_bottom(force=True)
        return wrapper

    def _message_attachment_button(self, chat_id, attachments):
        button = Gtk.Button(
            has_frame=False, halign=Gtk.Align.END, css_classes=["message-attachment-button"]
        )
        # llama-server accepts several image parts in a single vision request.
        show_previews = len(attachments) <= 3 and all(map(image_mime_type, attachments))
        if show_previews:
            label = "Image" if len(attachments) == 1 else "Images"
            tooltip = ["Open documents folder"]
        elif len(attachments) == 1:
            label = attachment_display_name(attachments[0], 32)
            name = attachment_display_name(attachments[0], 64)
            tooltip = [name if name != label else "Open documents folder"]
        else:
            label = "Multiple documents"
            ordered = sorted(
                attachments, key=lambda name: (not name.endswith("/"), name.casefold())
            )
            tooltip = [attachment_display_name(name, 64) for name in ordered[:20]]
            if len(ordered) > 20:
                tooltip.append("...")
        button.set_tooltip_text("\n".join(tooltip))
        header = Gtk.Box(css_classes=["message-attachment-content"])
        header.append(Gtk.Image(icon_name=ATTACH_DOCUMENTS_ICON))
        header.append(Gtk.Label(label=label, ellipsize=Pango.EllipsizeMode.MIDDLE))
        content = header
        if show_previews:
            content = Gtk.Box(
                orientation=Gtk.Orientation.VERTICAL, css_classes=["message-attachment-content"]
            )
            content.append(header)
            thumbnails = Gtk.Box(css_classes=["message-attachment-content"])
            folder = self.controller.store.documents(chat_id)
            for name in attachments:
                thumbnails.append(self._attachment_thumbnail(folder, name))
            content.append(thumbnails)
        button.set_child(content)
        button.connect("clicked", lambda _: self.controller.open_folder(chat_id, documents=True))
        return button

    @staticmethod
    def _attachment_thumbnail(folder, name):
        size = ATTACHMENT_THUMBNAIL_SIZE
        picture = Gtk.Picture(
            content_fit=Gtk.ContentFit.CONTAIN,
            overflow=Gtk.Overflow.HIDDEN,
            halign=Gtk.Align.CENTER,
            valign=Gtk.Align.CENTER,
            width_request=size,
            height_request=size,
            alternative_text=name,
            tooltip_text=name,
            css_classes=["message-attachment-thumbnail"],
        )

        def display(pixbuf):
            # Clip the picture's actual bounds, including for non-square thumbnails.
            picture.set_size_request(pixbuf.get_width(), pixbuf.get_height())
            picture.set_paintable(Gdk.Texture.new_for_pixbuf(pixbuf))

        load_picture(
            picture, lambda: DocumentFiles(folder).image(name), (size, size), display, picture.hide
        )
        return picture

    def _reply_images(self, chat_id):
        return ReplyImages(DocumentFiles(self.controller.store.documents(chat_id)))

    def _create_thinking_widget(self, wrapper, text, completed):
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        box.add_css_class("thinking")
        header_button = Gtk.Button()
        header_button.add_css_class("reasoning-header-button")
        header_button.set_has_frame(False)
        header_button.set_halign(Gtk.Align.START)
        content = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
        content.add_css_class("reasoning-content")
        arrow = Gtk.Label(label="▸" if completed else "▾")
        arrow.add_css_class("reasoning-arrow")
        content.append(arrow)
        content.append(Gtk.Label(label="Thinking", xalign=0))
        header_button.set_child(content)
        body = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        body.add_css_class("reasoning-body")
        body.thinking_arrow = arrow
        body.saved_text = text if completed else None
        if not completed:
            self._set_thinking_body(body, text)
        body.set_visible(not completed)
        header_button.connect("clicked", lambda _button: self._toggle_thinking(body))
        box.append(header_button)
        box.append(body)
        if wrapper is self.current_wrapper and self.current_answer is not None:
            wrapper.insert_child_after(box, wrapper.get_first_child())
        else:
            wrapper.append(box)
        return body

    def _toggle_thinking(self, body):
        self.follow_output = False
        if body.saved_text is not None:
            self._set_thinking_body(body, body.saved_text)
            body.saved_text = None
        body.set_visible(not body.get_visible())
        body.thinking_arrow.set_text("▾" if body.get_visible() else "▸")

    def _append_thinking_body(self, body, text):
        """Keep finished text layouts cached; only edit the small trailing label."""
        remaining = body.tail_text + text
        while remaining:
            if body.tail_label is None:
                label = Gtk.Label(xalign=0, yalign=0)
                label.set_wrap(True)
                label.set_wrap_mode(Pango.WrapMode.WORD_CHAR)
                label.set_max_width_chars(90)
                label.set_selectable(True)
                label.add_css_class("chat-prose")
                body.append(label)
                body.tail_label = label
            if len(remaining) <= 2000:
                body.tail_text = remaining
                body.tail_label.set_text(remaining)
                break
            boundary = remaining.rfind("\n", 1000, 2000)
            if boundary < 0:
                boundary = remaining.rfind(" ", 1000, 2000)
            boundary = boundary + 1 if boundary >= 0 else 2000
            body.tail_label.set_text(remaining[:boundary].removesuffix("\n"))
            remaining = remaining[boundary:]
            body.tail_label = None
            body.tail_text = ""

    def _set_thinking_body(self, body, text):
        while (child := body.get_first_child()) is not None:
            body.remove(child)
        body.tail_label = None
        body.tail_text = ""
        self._append_thinking_body(body, text)

    def begin_response(self, timestamp=None, *, chat_id=None):
        self.busy = True
        self.send_button.set_sensitive(True)
        self.send_button.set_icon_name("window-close-symbolic")
        self.send_button.set_tooltip_text("Stop generation")
        self.send_button.remove_css_class("suggested-action")
        self.send_button.add_css_class("destructive-action")
        self.set_status("Waiting")
        chat_id = chat_id or self.chat["id"]
        self.current_wrapper = self.append_message(
            "assistant", "", timestamp=timestamp, attach=chat_id == self.chat["id"]
        )
        self.current_answer = MarkdownText(streaming=True, images=self._reply_images(chat_id))
        self.current_content = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        self.current_content.add_css_class("streaming-response-current-content")
        self.current_content.append(self.current_answer)
        self.current_wrapper.append(self.current_content)
        self._answer_committed = 0
        self._first_answer_shown = self._first_thinking_shown = False
        self._stream_code = None
        self._thinking_collapsed = False
        self.current_thinking = None
        if chat_id == self.chat["id"]:
            self.scroll_bottom(True)
        self._response_row = self.current_wrapper.message_row
        self._response_chat_id = chat_id
        self.update_send_button()

    def update_thinking(self, text):
        if text:
            self.set_status("Thinking")
        if self.current_thinking is None:
            self.current_thinking = self._create_thinking_widget(self.current_wrapper, "", False)
        self._thinking_pending.append(text)
        if not self._first_thinking_shown and text.strip():
            self._first_thinking_shown = True
            if self._thinking_flush_id is not None:
                GLib.source_remove(self._thinking_flush_id)
            return self._flush_thinking()
        if self._thinking_flush_id is None:
            self._thinking_flush_id = GLib.timeout_add(40, self._flush_thinking)
        return False

    def _flush_thinking(self):
        self._thinking_flush_id = None
        text = "".join(self._thinking_pending)
        self._thinking_pending.clear()
        self._append_thinking_body(self.current_thinking, text)
        if self._response_row.get_parent() == self.transcript:
            self.scroll_bottom()
        return False

    def _replace_thinking(self, text):
        if self.current_thinking is None:
            self.current_thinking = self._create_thinking_widget(self.current_wrapper, "", False)
        if text == self._inband_thinking:
            return
        if text.startswith(self._inband_thinking):
            self.update_thinking(text[len(self._inband_thinking) :])
        else:
            self._thinking_pending.clear()
            self._set_thinking_body(self.current_thinking, text)
        self._inband_thinking = text

    def update_answer(self, text):
        text, thinking, completed = self.controller.split_thinking(text)
        if thinking is not None:
            self._replace_thinking(thinking.strip())
            self.set_status("Responding" if completed else "Thinking")
            if completed:
                self._collapse_thinking()
        elif text:
            self.set_status("Responding")
            self._collapse_thinking()
        self._render_streamed_answer(text)
        if self._response_row.get_parent() == self.transcript:
            self.scroll_bottom()
        return False

    def _render_streamed_answer(self, text):
        # Commit complete fences once, keeping buttons stable while the tail streams.
        tail = text[self._answer_committed :]
        end = max((match.end() for match in CODE_BLOCK_PATTERN.finditer(tail)), default=0)
        if end:
            if self._stream_code is not None:
                self.current_content.remove(self._stream_code[0])
                self._stream_code = None
            self.current_content.remove(self.current_answer)
            append_markdown(
                self.current_content,
                tail[:end],
                self.controller.commands_in,
                self._review_command,
                images=self.current_answer.images,
            )
            self.current_content.append(self.current_answer)
            self._answer_committed += end
            tail = tail[end:]
        opening = re.search(
            r"^ {0,3}(?:```|~~~)([^\n]*)(?:\n(.*))?$", tail, re.DOTALL | re.MULTILINE
        )
        prose = tail[: opening.start()] if opening else tail
        self.current_answer.set_visible(bool(prose))
        self.current_answer.set_text(prose)
        if opening:
            language, code = opening[1].strip(), opening[2] or ""
            if self._stream_code is None:
                self._stream_code = append_code_block(self.current_content, code, language)
            else:
                self._stream_code[1].language = language
                self._stream_code[1].set_text(code)

    def _collapse_thinking(self):
        if self.current_thinking is None or self._thinking_collapsed:
            return
        self._thinking_collapsed = True
        self.current_thinking.set_visible(False)
        self.current_thinking.thinking_arrow.set_text("▸")

    def finish_response(self, answer, reasoning):
        visible = self._response_row.get_parent() == self.transcript
        timestamp = self.chat["messages"][-1]["timestamp"]
        day_changed = datetime.fromtimestamp(self._response_row.timestamp).date() != (
            datetime.fromtimestamp(timestamp).date()
        )
        self._response_row.timestamp = timestamp
        self._response_row.time_label.set_text(message_time(timestamp))
        if reasoning and self.current_thinking is None:
            self.current_thinking = self._create_thinking_widget(self.current_wrapper, "", True)
        answer = answer.strip()
        if (
            self.current_answer.images.requests
            and not self._answer_committed
            and self._stream_code is None
            and not CODE_BLOCK_PATTERN.search(answer)
        ):
            self.current_answer.set_text(answer, finished=True)
            self.current_answer.set_visible(bool(answer))
        else:
            self.current_wrapper.remove(self.current_content)
            append_markdown(
                self.current_wrapper,
                answer,
                self.controller.commands_in,
                self._review_command,
                images=self.current_answer.images,
            )
        if self.current_thinking:
            self._collapse_thinking()
            if self.current_thinking.get_visible():
                self._set_thinking_body(self.current_thinking, reasoning.strip())
            else:
                self.current_thinking.saved_text = reasoning.strip()
        if not visible or day_changed:
            self.render_history(self.chat["messages"])
        return self.end_response()

    def end_response(self):
        if self._thinking_flush_id is not None:
            GLib.source_remove(self._thinking_flush_id)
            self._thinking_flush_id = None
        self._thinking_pending.clear()
        self._inband_thinking = ""
        if self._response_row.get_parent() == self.transcript:
            self.scroll_bottom()
        self._response_row = self._response_chat_id = None
        self.current_wrapper = self.current_content = None
        self.current_answer = self.current_thinking = None
        self._stream_code = None
        self.busy = False
        self.send_button.set_icon_name("go-up-symbolic")
        self.send_button.set_tooltip_text("Send")
        self.send_button.remove_css_class("destructive-action")
        self.send_button.add_css_class("suggested-action")
        self.update_send_button()
        self.set_status(self.controller.idle_status())
        return False

    def scroll_bottom(self, force=False):
        if force:
            self.follow_output = True
        if self.follow_output and self._scroll_idle_id is None:
            self._scroll_idle_id = GLib.idle_add(
                self._scroll_to_bottom, priority=GLib.PRIORITY_HIGH_IDLE
            )

    def _scroll_to_bottom(self):
        self._scroll_idle_id = None
        if not self.follow_output:
            return False
        adjustment = self.transcript_scroll.get_vadjustment()
        self._set_scroll_value(adjustment, adjustment.get_upper() - adjustment.get_page_size())
        return False

    def _set_scroll_value(self, adjustment, value):
        self._scroll_guard = True
        adjustment.set_value(value)
        self._scroll_guard = False

    @staticmethod
    def _closeable_alert(heading, body):
        alert = Adw.AlertDialog(heading=heading, body=body)
        alert.add_response("close", "Close")
        alert.set_default_response("close")
        alert.set_close_response("close")
        return alert

    def _show_alert(self, heading, body):
        self._closeable_alert(heading, body).present(self)

    def update_first_launch(self):
        self._update_setup_highlights()
        self.emit("optional-features-changed")
        if not self._welcome_pending or not self.get_mapped():
            return
        self._welcome_pending = False
        self._closeable_alert(
            "Welcome to Lopaca!",
            "Go to settings to configure a model, after you get acquainted check out the "
            "Knowledge Packs and other settings.",
        ).present(self)

    def _update_setup_highlights(self):
        active = (
            self.controller.first_launch and self.controller.connected is False
            and not self.controller.external_model_server
        )
        if active:
            folder = self.controller.store.root.parent
            config = self.controller.model_configuration() or {}
            executable = config.get("executable")
            active = not (
                (executable and Path(executable).expanduser().is_file())
                or cached_server_config(folder)
            )
        dialog = self._app_settings_dialog
        targets = [
            (self._settings_button, "first-launch-highlight", active and dialog is None),
            (
                self._settings_button, "optional-features-highlight",
                self._offer_optional_features(),
            ),
        ]
        model_visible = False
        if dialog is not None:
            page = dialog.get_visible_page()
            model_visible = page is not None and page.get_name() == "model"
            targets.append((dialog, "first-launch-model-tab", active and not model_visible))
        if hasattr(self, "_configured_server"):
            row = self._configured_server
            targets.append(
                (row, "first-launch-highlight", active and model_visible and row.get_activatable())
            )
        for widget, css_class, highlight in targets:
            if highlight:
                widget.add_css_class(css_class)
            else:
                widget.remove_css_class(css_class)

    def _app_settings_closed(self, dialog):
        dialog.remove_css_class("first-launch-model-tab")
        self._app_settings_dialog = None
        self._update_setup_highlights()

    def _rename_chat(self, chat_id):
        chat = self.controller.store.load_settings(chat_id)
        dialog = Adw.Dialog(title="Rename Chat", content_width=380)
        toolbar = Adw.ToolbarView()
        toolbar.add_css_class("rename-chat-toolbar")
        header = Adw.HeaderBar()
        header.add_css_class("rename-chat-header")
        toolbar.add_top_bar(header)
        entry = Gtk.Entry(text=chat.get("title", "New chat"))
        entry.add_css_class("rename-chat-entry")
        content = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        content.add_css_class("rename-chat-content")
        content.append(entry)
        buttons = Gtk.Box(halign=Gtk.Align.END)
        buttons.add_css_class("rename-chat-buttons")
        cancel_button = Gtk.Button(label="Cancel")
        cancel_button.add_css_class("rename-chat-cancel-button")
        cancel_button.connect("clicked", lambda *_: dialog.close())
        buttons.append(cancel_button)
        content.append(buttons)
        toolbar.set_content(content)
        dialog.set_child(toolbar)
        save_button = Gtk.Button(label="Save")
        save_button.add_css_class("rename-chat-save-button")
        save_button.add_css_class("suggested-action")

        def save(*_):
            self.controller.rename_chat(chat_id, entry.get_text())
            dialog.close()

        save_button.connect("clicked", save)
        entry.connect("activate", save)
        buttons.append(save_button)
        dialog.present(self)
        entry.grab_focus()

    def _show_settings(self, chat_id=None):
        dialog = Adw.PreferencesDialog(
            title="App Settings" if chat_id is None else "Chat Settings",
            content_width=720,
            content_height=640,
        )
        dialog.set_search_enabled(False)
        page = Adw.PreferencesPage(
            title="General" if chat_id is None else "Chat Settings",
            icon_name="user-available-symbolic",
        )
        page.add_css_class("settings-page")
        page.connect("map", self._sync_native_scroll)
        dialog.add(page)
        if chat_id is None:
            page.add(self._optional_features_group(dialog))
        group = Adw.PreferencesGroup()
        group.add_css_class("settings-group")
        page.add(group)
        settings = self.controller.get_chat_settings(chat_id)
        group.set_title("Default Chat Settings" if chat_id is None else "Chat Settings")
        if chat_id is not None:
            page.set_description(
                "Changes override app defaults for this chat. 'X' denotes an override."
            )
        overrides = self.controller.get_chat_overrides(chat_id) if chat_id is not None else set()
        markers = []
        setting_controls = {}
        recall_switches = {}
        recall_handler = None

        def update_recall_switches(*_):
            active = self.controller.embedding_ready()
            configured = self.controller.embedding_configured()
            for key, (row, button, handler, title, tooltip) in recall_switches.items():
                with button.handler_block(handler):
                    button.set_active(active or settings[key] is True)
                row.set_sensitive(not active)
                row.set_tooltip_text("Handled by the embedding model." if active else tooltip)
                if key in ("rag_model_indexing", "history_model_indexing"):
                    row.set_title(title if configured else f"{title} (SLOW)")

        def connect_recall_switches(*_):
            nonlocal recall_handler
            recall_handler = self.connect("retrieval-settings-changed", update_recall_switches)
            update_recall_switches()

        def disconnect_recall_switches(*_):
            nonlocal recall_handler
            if recall_handler is not None:
                self.disconnect(recall_handler)
                recall_handler = None

        def update_markers():
            for marker, key, button in markers:
                active = not isinstance(button, Gtk.CheckButton) or button.get_active()
                overridden = key in overrides and active
                marker.set_opacity(1 if overridden else 0)
                marker.set_sensitive(overridden)

        def delete_override(_button, key):
            try:
                self.controller.set_setting(key, None, chat_id)
            except OSError as exc:
                self._show_alert("Could not delete override", str(exc))
                return
            settings[key] = self.controller.get_chat_settings(chat_id)[key]
            overrides.discard(key)
            for button, handler, choice in setting_controls[key]:
                with button.handler_block(handler):
                    button.set_active(
                        settings[key] is True if choice is None else settings[key] == choice
                    )
            update_markers()

        def add_control(row, button, key):
            if chat_id is None:
                row.add_suffix(button)
            else:
                box = Gtk.Box()
                box.add_css_class("settings-control-group")
                marker = Gtk.Button(tooltip_text="Delete override", valign=Gtk.Align.CENTER)
                marker.add_css_class("override-delete-button")
                marker.set_child(
                    Gtk.Image(icon_name="window-close-symbolic", css_classes=["delete-icon"])
                )
                marker.add_css_class("flat")
                marker.connect("clicked", delete_override, key)
                box.append(marker)
                box.append(button)
                row.add_suffix(box)
                markers.append((marker, key, button))
                update_markers()
            row.set_activatable_widget(button)

        def add_switch(key, title, tooltip, *, inverted=False):
            row = Adw.ActionRow(title=title)
            row.add_css_class("settings-row")
            button = Gtk.Switch(active=(settings[key] is True) != inverted, valign=Gtk.Align.CENTER)
            button.add_css_class("settings-button")
            add_control(row, button, key)
            row.set_tooltip_text(tooltip)

            def changed(row, _param):
                try:
                    value = row.get_active() != inverted
                    self.controller.set_setting(key, value, chat_id)
                    settings[key] = value
                    overrides.add(key)
                    update_markers()
                except OSError as exc:
                    with row.handler_block(handler):
                        row.set_active((settings[key] is True) != inverted)
                    self._show_alert("Could not save settings", str(exc))

            handler = button.connect("notify::active", changed)
            setting_controls[key] = [(button, handler, False if inverted else None)]
            if key in ("rag_query_expansion", "rag_model_indexing", "history_model_indexing"):
                recall_switches[key] = (row, button, handler, title, tooltip)
            group.add(row)

        if chat_id is None:
            add_switch(
                "global_memories", "Global Memories", "Recall and learn memories across chats."
            )
            memories_row = Adw.ActionRow(title="Manage Memories", activatable=True)
            memories_row.add_css_class("settings-memories-row")
            memories_row.add_suffix(Gtk.Image(icon_name="go-next-symbolic"))
            memories_row.connect("activated", lambda *_: self._show_memories(dialog))
            group.add(memories_row)
            add_switch(
                "global_chat_histories",
                "Global Chat Histories",
                "Recall relevant discussions from other chats.",
            )
        else:
            add_switch(
                "global_memories",
                "Exclude From Memories",
                "Turn off memory recall and learning in this chat.",
                inverted=True,
            )
            add_switch(
                "exclude_from_history",
                "Exclude From History",
                "Keep this chat out of cross-chat history searches.",
            )
        add_switch(
            "rag_query_expansion",
            "Expand Retrieval Searches",
            "Ask the current model for related search terms.",
        )
        add_switch(
            "rag_model_indexing",
            "Model-Assisted Document Indexing",
            "Modestly improves document info retrieval. CAN BE SLOW, best used with t/s > 80.",
        )
        add_switch(
            "history_model_indexing",
            "Model-Assisted History Indexing",
            "Improves search in archived messages in this chat and in other chats. "
            "CAN BE SLOW, best used with t/s > 80.",
        )
        page.connect("map", connect_recall_switches)
        page.connect("unmap", disconnect_recall_switches)
        update_recall_switches()

        def add_choices(title, options, key, setter):
            choice_group = Adw.PreferencesGroup(title=title)
            choice_group.add_css_class("settings-choice-group")
            page.add(choice_group)
            controls = {}

            def choice_changed(button, value):
                if not button.get_active():
                    return
                try:
                    setter(value)
                    settings[key] = value
                    overrides.add(key)
                    update_markers()
                except OSError as exc:
                    previous_button, previous_handler = controls[settings[key]]
                    with previous_button.handler_block(previous_handler):
                        previous_button.set_active(True)
                    self._show_alert("Could not save settings", str(exc))

            radio_group = None
            for value, label in options:
                button = Gtk.CheckButton(valign=Gtk.Align.CENTER)
                button.add_css_class("settings-button")
                if radio_group is None:
                    radio_group = button
                else:
                    button.set_group(radio_group)
                button.set_active(value == settings[key])
                handler = button.connect("toggled", choice_changed, value)
                controls[value] = button, handler
                row = Adw.ActionRow(title=label)
                row.add_css_class("settings-row")
                add_control(row, button, key)
                choice_group.add(row)
            setting_controls[key] = [
                (button, handler, value) for value, (button, handler) in controls.items()
            ]

        add_choices(
            "Default Conversation Style" if chat_id is None else "Conversation Style",
            CONVERSATION_STYLE_OPTIONS,
            "conversation_style",
            lambda value: self.controller.set_setting("conversation_style", value, chat_id),
        )
        if chat_id is None:
            add_choices(
                "Unpinned Chat Sort",
                CHAT_SORT_OPTIONS,
                "chat_sort",
                self.controller.set_chat_sort,
            )
            dialog.add(self._model_settings_page())
            dialog.add(self._knowledge_settings_page(dialog))
            dialog.add(self._about_settings_page())
            self._app_settings_dialog = dialog
            dialog.connect("notify::visible-page", self._settings_tab_changed)
            dialog.connect("closed", self._app_settings_closed)
            self._update_setup_highlights()
        dialog.present(self)

    def _settings_tab_changed(self, dialog, _property):
        if page := dialog.get_visible_page():
            if hasattr(page, "scroll_to_top"):
                page.scroll_to_top()
            else:
                child = page.get_first_child()
                while child and not isinstance(child, Gtk.ScrolledWindow):
                    child = child.get_first_child()
                if child:
                    adjustment = child.get_vadjustment()
                    adjustment.set_value(adjustment.get_lower())
        self._update_setup_highlights()

    def _about_settings_page(self):
        def link_row(title, subtitle, url):
            row = Adw.ActionRow(title=title, subtitle=subtitle)
            link = Gtk.LinkButton(uri=url, label="")
            link.add_css_class("about-link-button")
            link.set_child(Gtk.Image(icon_name="external-link-symbolic"))
            link.set_valign(Gtk.Align.CENTER)
            link.set_tooltip_text(url)
            row.add_suffix(link)
            row.set_activatable_widget(link)
            return row

        page = Adw.PreferencesPage(title="About", icon_name="preferences-system-details-symbolic")
        page.add_css_class("about-page")
        about = Adw.PreferencesGroup(title="Lopaca")
        about.add_css_class("about-project")
        about.add(
            link_row(
                "Created by Timothy Porter",
                "Lopaca developer and maintainer (2026-present)",
                "https://github.com/portets",
            )
        )
        page.add(about)

        licenses = Adw.PreferencesGroup(title="Licenses")
        licenses.add_css_class("about-licenses")
        page.add(licenses)
        for title, notice in (
            (
                "Lopaca",
                "Copyright © Timothy Porter · "
                '<a href="https://www.gnu.org/licenses/agpl-3.0.en.html"> GNU Affero General '
                "Public License v3.0</a>",
            ),
            (
                "llama.cpp llama-server",
                'Copyright © The ggml authors · <a href="https://github.com/ggml-org/llama.cpp/'
                'blob/master/LICENSE">MIT License</a>',
            ),
            (
                "Qwen Models",
                'Copyright © Alibaba Cloud · <a href="https://huggingface.co/Qwen/Qwen3.8-27B/'
                'blob/main/LICENSE">Apache 2.0 License</a>',
            ),
            (
                "Wikipedia knowledge packs",
                "Adapted from Wikipedia contributors · "
                '<a href="https://creativecommons.org/licenses/by-sa/4.0/"'
                ">CC BY-SA 4.0 License</a>",
            ),
        ):
            licenses.add(Adw.ActionRow(title=title, subtitle=notice, use_markup=True))

        donations = Adw.PreferencesGroup(title="Donate")
        donations.add_css_class("about-donations")
        donations.add(link_row("Buy Me a Coffee", "Thanks!", "https://buymeacoffee.com/portets"))
        page.add(donations)
        return page

    def update_model_server(self, text, busy):
        if text != DOWNLOADING_MODEL or not busy:
            if not self.controller.model_file_download:
                self._model_download.pop("model", None)
            self._model_download.pop("server", None)
        self._model_setup_downloading = busy and text == DOWNLOADING_MODEL
        self._model_server_activity = "Connecting to model" if busy else ""
        self._refresh_model_status()
        if hasattr(self, "_model_server_status"):
            self._model_server_status.set_label(text)
            self._model_connect_button.set_sensitive(not busy)
        self.refresh_model_configuration()

    def update_model_download(self, stage, server, percent):
        if stage in ("vision", "embedding") and percent is None:
            self._model_download.pop(stage, None)
        else:
            self._model_download[stage] = server, percent
        self.refresh_model_configuration()

    def _optional_features_group(self, dialog):
        group = Adw.PreferencesGroup()
        group.add_css_class("settings-group")
        row = Adw.ActionRow(title="Download Optional Features (Recommended)", activatable=True)
        row.add_css_class("model-settings-summary-row")
        row.add_css_class("optional-features-highlight")
        row.set_tooltip_text(OPTIONAL_FEATURES_TOOLTIP)
        arrow = Gtk.Image(icon_name="go-next-symbolic", valign=Gtk.Align.CENTER)
        row.add_suffix(arrow)
        indicator = add_operation_indicator(row, visible=False)
        row.connect("activated", lambda *_: self.controller.download_optional_features())
        group.add(row)

        def refresh(*_):
            stage = self.controller.optional_features_stage
            group.set_visible(self._offer_optional_features())
            row.set_activatable(stage is None)
            arrow.set_visible(stage is None)
            self._update_download_indicator(indicator, stage or "", bool(stage), None)

        handler = self.connect("optional-features-changed", refresh)
        dialog.connect("closed", lambda *_: self.disconnect(handler))
        refresh()
        return group

    def _offer_optional_features(self):
        return (
            not self.controller.first_launch and not self.controller.optional_features_installed()
        )

    def confirm_model_download(self, *, reconfigure=False):
        if self._model_setup_dialog is not None or self.controller.model_setup_busy():
            return
        rule = MODEL_RECOMMENDATIONS[self.controller.model_recommendation()]
        model = MODEL_DOWNLOADS[rule.balanced]
        folder = self.controller.store.root.parent
        settings = self.controller.store.load_app_settings()
        config = settings.get("model_server", {})
        downloads = []
        selected = config.get("model")
        custom_model = bool(
            selected and selected in settings.get("custom_model_files", {}).get("model", {})
            and Path(selected).expanduser().is_file()
        )
        if not custom_model and not model_path(folder, rule.balanced).is_file():
            downloads.append(f"{model.name} ({model.quantization}) · {model.size / 10**9:.2f} GB")
        family = None if custom_model else model.vision_family
        if rule.vision and family and family not in config.get("vision_disabled", ()):
            precision = self.controller._recommended_vision_precision
            vision = VISION_DOWNLOADS[family][precision]
            if not vision_path(folder, family, precision).is_file():
                downloads.append(
                    f"Qwen {family} Vision ({precision}) · {vision.size / 10**9:.2f} GB"
                )
        embedding = settings.get("embedding_model", {})
        if (
            embedding.get("enabled") is not False
            and not embedding_path(folder).is_file()
            and rule.embedding
        ):
            selected = Path(embedding.get("path") or ".")
            with suppress(RuntimeError):
                selected = selected.expanduser()
            if not selected.is_file():
                downloads.append(
                    f"{EMBEDDING_DOWNLOAD.name} · {EMBEDDING_DOWNLOAD.size / 10**9:.2f} GB"
                )
        intro = (
            "Switch to an automatically configured model? "
            if reconfigure
            else "No model loaded, would you like Lopaca to download and configure a model? "
        )
        details = "\n\nEstimated downloads\n" + "\n".join(downloads) if downloads else ""
        alert = Adw.AlertDialog(
            follows_content_size=False,
            content_width=480,
            heading="Set up a local model?",
            body=f"{intro}You can also configure a model in Settings → Model → Advanced.{details}",
        )
        alert.add_response("cancel", "Cancel")
        alert.add_response("yes", "Yes")
        alert.set_default_response("cancel")
        alert.set_close_response("cancel")
        extras = Gtk.CheckButton(label="Download optional features? (Recommended)", active=True)
        extras.set_tooltip_text(OPTIONAL_FEATURES_TOOLTIP)
        extras.set_sensitive(
            not self.controller.optional_features_installed()
            and self.controller.optional_features_stage is None
        )
        alert.set_extra_child(extras)

        def chosen(alert, result):
            self._model_setup_dialog = None
            if alert.choose_finish(result) == "yes":
                self.controller.download_default_model(
                    reconfigure=reconfigure, optional_features=extras.get_active()
                )
            self.refresh_model_configuration()

        self._model_setup_dialog = alert
        alert.choose(self, None, chosen)

    def refresh_model_configuration(self):
        downloading = (
            self._model_setup_downloading
            or self.controller.model_file_download
            or self.controller.vision_download_active
            or self.controller.embedding_download_active
            or any(percent is not None for _, percent in self._model_download.values())
        )
        self.set_background_activity(
            "model_download", "Downloading files" if downloading else "", 30,
        )
        self._update_model_picker()
        if not hasattr(self, "_configured_server"):
            return
        config = self.controller.model_configuration()
        external = config is None
        config = config or {}
        settings = self.controller.store.load_app_settings()
        if hasattr(self, "_custom_server_add"):
            self._custom_server_add.set_sensitive(not external)
        downloaded_server, server_percent = self._model_download.get("server", (None, None))
        folder = self.controller.store.root.parent
        executable = config.get("executable")
        manual = bool(executable and not managed_server_file(folder, executable))
        server_config = downloaded_server or (
            config if executable and not manual else
            self.controller.managed_model_server_config(settings) or {}
        )
        self._auto_configure_manual = manual
        busy = self.controller.model_setup_busy()
        can_download = False
        managed = server_config.get("executable")
        server_available = managed and Path(managed).expanduser().is_file()
        downloading_server = server_percent is not None or (
            "server" not in self._model_download
            and self._model_server_activity == DOWNLOADING_MODEL
        )
        if downloading_server:
            server = "llama.cpp", "", True
        elif external and not self.controller.external_model_server:
            server = "Manually configured by user", "", False
        elif server_available:
            server = (
                "llama.cpp", self._model_server_subtitle(server_config), not (manual or external)
            )
        else:
            can_download = not (busy or external)
            server = "Click here to automatically download and set-up", "", can_download

        self._configured_server.set_activatable(can_download)
        self._server_download_arrow.set_visible(can_download)
        title, subtitle, sensitive = server
        self._configured_server.set_title(title)
        self._configured_server.set_subtitle(subtitle)
        self._configured_server.set_sensitive(sensitive)
        self._update_setup_highlights()
        self._update_download_indicator(
            self._server_download_indicator,
            "Downloading server",
            downloading_server,
            server_percent,
        )
        selected_config = settings.get("model_server", {}) if external else config
        self._refresh_model_rows(selected_config, downloaded_server, external)
        self._refresh_custom_file_rows(settings, selected_config, external)
        self._refresh_model_parameter_defaults()

    @staticmethod
    def _model_server_subtitle(info):
        version = re.sub(
            r", commit [^)]+(?=\))|^(b\d+)-[a-f0-9]+$",
            r"\1",
            info.get("version") or "",
        )
        backend = info.get("backend") or ""
        return " | ".join(filter(None, (version, BACKEND_LABELS.get(backend, backend.upper()))))

    def _model_parameter_config(self, config):
        profile = self.controller.model_recommendation()
        balanced = MODEL_RECOMMENDATIONS[profile].balanced
        model = config.get("model") or MODEL_DOWNLOADS[balanced].filename
        return {**config, "recommendation": profile, "model": model}

    def _refresh_model_parameter_defaults(self):
        if not getattr(self, "_model_parameter_rows", None):
            return
        settings = self.controller.store.load_app_settings()
        draft = settings.get("advanced_model_settings", {}).get("model_server", {})
        config = self._model_parameter_config(settings.get("model_server", {}) | draft)
        identity = config["recommendation"], config["model"]
        if identity == self._model_parameter_identity:
            return
        self._model_parameter_identity = identity
        self._model_parameter_defaults = server_parameters(
            {"recommendation": config["recommendation"], "model": config["model"]}
        )
        self._show_model_parameter_values(server_parameters(config), self._model_parameter_rows)

    def _show_model_parameter_values(self, values, keys):
        for key in keys:
            row, handler, reset = self._model_parameter_rows[key]
            with row.handler_block(handler):
                value = str(values[key])
                if choices := SERVER_FIELD_CHOICES.get(key):
                    self._set_model_parameter_choice(row, choices, value)
                else:
                    row.set_text(value)
            reset.set_visible(self._has_model_override(key, value))
            if key == "port":
                self._update_external_server_title(value)

    def _update_external_server_title(self, value):
        port = str(value).strip() or "9931"
        self._external_server_row.set_title(f"Use external model server at {port}")

    def _has_model_override(self, key, value):
        default = self._model_parameter_defaults[key]
        value = value.strip()
        if value and isinstance(default, (float, int)):
            with suppress(ValueError):
                return float(value) != default
        return bool(value and value != str(default))

    @staticmethod
    def _set_model_parameter_choice(row, choices, value):
        # Keep a manually configured value visible until the user replaces it.
        if value not in choices:
            choices = (*choices, value)
        row.set_model(Gtk.StringList.new(choices))
        row.set_selected(choices.index(value))

    def _reset_model_parameters(self, keys):
        overrides = dict.fromkeys(keys, "")
        try:
            self.controller.save_advanced_model_settings("model_server", overrides)
        except OSError as exc:
            self._show_alert("Could not save settings", str(exc))
            return
        settings = self.controller.store.load_app_settings()
        draft = settings.get("advanced_model_settings", {}).get("model_server", {})
        config = self._model_parameter_config(settings.get("model_server", {}) | draft)
        self._show_model_parameter_values(server_parameters(config), keys)

    @staticmethod
    def _model_reset_button(title, reset):
        label = f"Reset {title}"
        button = Gtk.Button(
            icon_name="view-refresh-symbolic", tooltip_text=label, valign=Gtk.Align.CENTER,
            focus_on_click=False,
        )
        button.add_css_class("flat")
        button.update_property([Gtk.AccessibleProperty.LABEL], [label])
        button.connect("clicked", lambda *_: reset())
        return button

    def _add_model_reset_button(self, group, reset):
        button = self._model_reset_button(group.get_title(), reset)
        group.set_header_suffix(button)
        return button

    @staticmethod
    def _update_download_indicator(indicator, text, active, percent):
        label, spinner = indicator
        label.set_label(f"{text}: {percent}%" if percent is not None else f"{text}…")
        label.set_visible(active)
        spinner.set_visible(active)
        spinner.set_spinning(active)

    def _refresh_model_rows(self, config, downloaded_server, external):
        profile = (downloaded_server or {}).get(
            "recommendation"
        ) or self.controller.model_recommendation()
        rule = MODEL_RECOMMENDATIONS[profile]
        folder = self.controller.store.root.parent
        selected = Path(config.get("model") or ".").expanduser()
        downloading, percent = self._model_download.get("model", (None, None))
        file_job = self.controller.model_file_download
        setup_busy = self.controller.model_setup_busy()
        recommended = {role: getattr(rule, role) for role in self._model_rows}
        recommended_keys = set(recommended.values())
        rows = [
            (key, self._model_rows[role], False) for role, key in recommended.items()
        ]
        rows.extend((key, widgets, True) for key, widgets in self._extra_model_rows.items())
        for key, widgets, extra in rows:
            row, action, switch, indicator, handler = widgets
            asset = MODEL_DOWNLOADS[key]
            path = model_path(folder, key)
            installed = path.is_file()
            active = installed and path == selected
            in_progress = percent is not None if key == downloading else key == file_job
            if extra:
                visible = key not in recommended_keys and (installed or in_progress)
                row.set_visible(visible)
                if not visible:
                    continue
            row.set_subtitle(
                ("" if extra else f"{asset.name} · ")
                + f"{asset.publisher} | {asset.quantization}"
                + model_file_size(asset.size)
            )
            if installed:
                row.remove_css_class("dim-label")
            else:
                row.add_css_class("dim-label")
            cancellable = not installed and (key == file_job or (setup_busy and in_progress))
            cancelling = cancellable and self.controller.download_cancels["model"].is_set()
            if cancellable:
                action.show_action("cancel", enabled=not cancelling)
            elif installed:
                action.show_action(
                    "remove", enabled=not (in_progress or setup_busy or external)
                )
            else:
                action.show_action("download", enabled=not setup_busy and not file_job)
            switch.set_visible(installed)
            switch.set_sensitive(installed and not setup_busy and not external)
            with switch.handler_block(handler):
                switch.set_active(active)
            self._update_download_indicator(
                indicator, "Cancelling" if cancelling else "Downloading", in_progress,
                percent if key == downloading and not cancelling else None,
            )
        self._refresh_vision_rows(config, rule, external)
        self._refresh_embedding_row(folder)

    def _refresh_embedding_row(self, folder):
        active = self.controller.embedding_download_active
        _, percent = self._model_download.get("embedding", (None, None))
        config = self.controller.store.load_app_settings().get("embedding_model", {})
        self._embedding_parameters.set_visible(bool(config.get("path")))
        selected = Path(config.get("path") or ".")
        with suppress(RuntimeError):
            selected = selected.expanduser()
        row, action, switch, indicator, handler = self._embedding_row
        installed = embedding_path(folder).is_file()
        available = selected == embedding_path(folder) and selected.is_file()
        if installed or available:
            row.remove_css_class("dim-label")
        else:
            row.add_css_class("dim-label")
        cancelling = active and self.controller.download_cancels["embedding"].is_set()
        if active:
            action.show_action("cancel", enabled=not cancelling)
        else:
            action.show_action("remove" if installed else "download")
        switch.set_visible(installed or available)
        switch.set_sensitive((installed or available) and not active)
        with switch.handler_block(handler):
            switch.set_active(available and config.get("enabled", True))
        self._update_download_indicator(
            indicator, "Cancelling" if cancelling else "Downloading", active,
            None if cancelling else percent,
        )

    def _embedding_choice(self):
        group = Adw.PreferencesGroup(title="Embedding")
        group.add_css_class("model-settings-summary")
        asset = EMBEDDING_DOWNLOAD
        row = Adw.ActionRow(
            title=asset.name,
            subtitle=asset.quantization + model_file_size(asset.size),
            use_markup=False,
        )
        row.set_tooltip_text(
            "Allows Lopaca to use an embedding model to drastically speed up and improve "
            "certain features. This runs on the CPU without interfering with the main loaded model."
        )
        self._embedding_row = self._model_row_controls(
            row,
            "Embedding Model",
            lambda: self._change_embedding("Install", self.controller.download_embedding),
            lambda active: self._change_embedding(
                "Enable" if active else "Disable",
                lambda: self.controller.select_embedding(active),
            ),
            lambda: self._change_embedding("Remove", self.controller.delete_embedding),
            "embedding",
        )
        group.add(row)
        return group

    def _change_embedding(self, action, apply):
        recommended = MODEL_RECOMMENDATIONS[self.controller.model_recommendation()].embedding
        enabling = action in ("Install", "Enable")
        if enabling == recommended:
            if action == "Remove":
                self._confirm_model_removal(EMBEDDING_DOWNLOAD.name, apply)
            else:
                apply()
            return
        # The switch should show the current state while the user decides.
        self.refresh_model_configuration()
        message = (
            "Gives better results but not recommended for users without a GPU."
            if enabling else "The embedding model gives Lopaca better and faster results."
        )
        self._confirm_model_action("Are you sure?", message, action, apply)

    def _refresh_vision_rows(self, config, rule, external):
        folder = self.controller.store.root.parent
        families = {
            MODEL_DOWNLOADS[key].vision_family for key in (rule.balanced, rule.faster, rule.better)
        }
        selected = model_vision_family(folder, config.get("model"))
        projector = config.get("mmproj")
        default_precision = config.get(
            "vision_precision", self.controller._recommended_vision_precision
        )
        downloading, percent = self._model_download.get("vision", (None, None))
        explicit = self.controller.vision_download_active
        busy = self.controller.vision_file_jobs_busy()
        model_available = bool(selected and Path(config["model"]).expanduser().is_file())
        visible = False
        for family, (row, action, switch, indicator, handler) in self._vision_rows.items():
            installed = [
                variant
                for variant in VISION_DOWNLOADS[family]
                if vision_path(folder, family, variant).is_file()
            ]
            row_visible = bool(family in families or family == selected or installed)
            row.set_visible(row_visible)
            visible |= row_visible
            compatible = family == selected and not external and model_available
            switch.set_visible(bool(installed))
            switch.set_sensitive(compatible and not busy and bool(installed))
            if compatible:
                row.remove_css_class("dim-label")
            else:
                row.add_css_class("dim-label")
            enabled = bool(
                installed
                and (
                    family != selected or not projector
                    or vision_file_precision(folder, family, projector)
                )
                and family not in config.get("vision_disabled", ())
            )
            with switch.handler_block(handler):
                switch.set_active(enabled)
            job = downloading or explicit
            in_progress = job is not None and job[0] == family
            cancelling = in_progress and self.controller.download_cancels["vision"].is_set()
            if in_progress:
                action.show_action("cancel", enabled=downloading is not None and not cancelling)
            else:
                action.show_action("remove" if installed else "download", enabled=not busy)
            precision = job[1] if in_progress else default_precision
            if not in_progress:
                chosen = vision_file_precision(folder, family, projector) if compatible else None
                precision = chosen or (
                    installed[0] if installed and precision not in installed else precision
                )
            subtitle = precision
            if asset := VISION_DOWNLOADS[family].get(precision):
                subtitle += model_file_size(asset.size)
            row.set_subtitle(subtitle)
            self._update_download_indicator(
                indicator, "Cancelling" if cancelling else "Downloading", in_progress,
                percent if job == downloading and not cancelling else None,
            )
        self._vision_group.set_visible(visible or "mmproj" in self._custom_file_groups)

    def _vision_choices(self):
        self._vision_group = Adw.PreferencesGroup(title="Vision")
        self._vision_group.add_css_class("model-settings-summary")
        self._vision_rows = {}
        for family, size in (("3.8", "27B"), ("3.5", "9B")):
            row = Adw.ActionRow(title=f"Qwen {family} Vision {size}", use_markup=False)
            row.set_tooltip_text(
                "Provides the ability for the model to view images. NOTE: "
                "this doesn't give the model the ability to create images."
            )
            self._vision_rows[family] = self._model_row_controls(
                row,
                "Vision Files",
                lambda family=family: self.controller.download_vision(family),
                lambda active, family=family: self._change_vision(
                    "Enable" if active else "Disable",
                    lambda: self.controller.select_vision(family, active),
                ),
                lambda family=family: self._change_vision(
                    "Remove",
                    lambda: self.controller.delete_vision(family),
                ),
                "vision",
            )
            self._vision_group.add(row)
        self._add_custom_file_button(self._vision_group, "mmproj", "vision file")
        return self._vision_group

    def _change_vision(self, action, apply):
        if action == "Enable":
            apply()
            return
        self.refresh_model_configuration()
        self._confirm_model_action(
            "Are you sure?",
            "Vision models are not actually loaded and used until you send the assistant an image.",
            action, apply,
        )

    def _model_row_controls(self, row, label, on_download, on_select, on_remove, kind):
        row.add_css_class("knowledge-packs-row")
        indicator = add_operation_indicator(row, visible=False)
        switch = Gtk.Switch(valign=Gtk.Align.CENTER)
        switch.add_css_class("knowledge-packs-switch")
        action = DownloadAction(
            label, download=on_download, remove=on_remove,
            cancel=lambda: self.controller.cancel_model_download(kind),
        )
        row.add_suffix(switch)
        row.add_suffix(action)
        handler = switch.connect("notify::active", lambda button, _: on_select(button.get_active()))
        return row, action, switch, indicator, handler

    def _model_choices(self):
        group = Adw.PreferencesGroup(title="Configured Local Model")
        group.add_css_class("model-settings-summary")
        self._model_rows = {}

        def model_key(role):
            downloaded_server, _ = self._model_download.get("server", (None, None))
            profile = (downloaded_server or {}).get(
                "recommendation"
            ) or self.controller.model_recommendation()
            return getattr(MODEL_RECOMMENDATIONS[profile], role)

        for role in ("balanced", "faster", "better"):
            row = Adw.ActionRow(title=role.title(), use_markup=False)
            self._model_rows[role] = self._model_row_controls(
                row,
                "Model",
                lambda role=role: self.controller.download_model(model_key(role)),
                lambda active, role=role: self.controller.select_model(model_key(role), active),
                lambda role=role: self._confirm_model_delete(model_key(role)),
                "model",
            )
            group.add(row)
        self._extra_model_rows = {}
        for key, asset in MODEL_DOWNLOADS.items():
            row = Adw.ActionRow(title=asset.name, use_markup=False, visible=False)
            self._extra_model_rows[key] = self._model_row_controls(
                row,
                "Model",
                lambda key=key: self.controller.download_model(key),
                lambda active, key=key: self.controller.select_model(key, active),
                lambda key=key: self._confirm_model_delete(key),
                "model",
            )
        self._add_custom_file_button(group, "model", "model")
        return group

    @staticmethod
    def _add_list_button(group, tooltip):
        button = Gtk.Button(icon_name="list-add-symbolic", tooltip_text=tooltip)
        button.add_css_class("knowledge-packs-button")
        footer = Gtk.Box(halign=Gtk.Align.END, css_classes=["list-add-controls"])
        footer.append(button)
        group.add(footer)
        return button

    def _add_custom_file_button(self, group, kind, label):
        button = self._add_list_button(group, f"Add custom {label}")
        self._custom_file_groups[kind] = group

        def choose(*_):
            button.set_sensitive(False)
            chooser = Gtk.FileDialog(title=f"Add custom {label}")
            if kind != "executable":
                file_filter = Gtk.FileFilter(name="GGUF files")
                file_filter.add_suffix("gguf")
                filters = Gio.ListStore.new(Gtk.FileFilter)
                filters.append(file_filter)
                chooser.set_filters(filters)

            def finished():
                button.set_sensitive(True)
                self.refresh_model_configuration()

            def selected(dialog, result):
                try:
                    file = dialog.open_finish(result)
                    filename = file.get_path() if file else None
                    if not filename:
                        raise ValueError("Choose a local file.")
                except GLib.Error as exc:
                    if not exc.matches(Gtk.dialog_error_quark(), Gtk.DialogError.DISMISSED):
                        self._show_alert("Could not select file", exc.message)
                except ValueError as exc:
                    self._show_alert("Could not select file", str(exc))
                else:
                    self.controller.add_custom_model_file(kind, filename, finished)
                    return
                finished()

            chooser.open(self, None, selected)

        button.connect("clicked", choose)
        return button

    @staticmethod
    def _list_remove_button(tooltip):
        button = Gtk.Button(tooltip_text=tooltip, valign=Gtk.Align.CENTER)
        button.set_child(Gtk.Image(icon_name="window-close-symbolic", css_classes=["delete-icon"]))
        button.add_css_class("flat")
        button.add_css_class("list-remove-button")
        return button

    def _custom_file_row(self, kind, path, info):
        if kind == "executable":
            subtitle = self._model_server_subtitle(info)
        else:
            metadata = " | ".join(filter(None, (info.get("publisher"), info.get("quantization"))))
            size = model_file_size(info["size"]) if info.get("size") else ""
            subtitle = (metadata + size).removeprefix(" · ")
        row = Adw.ActionRow(
            title=info["name"], subtitle=subtitle or Path(path).name, use_markup=False,
        )
        row.add_css_class("knowledge-packs-row")
        row.set_tooltip_text(path)
        toggle = Gtk.Switch(valign=Gtk.Align.CENTER)
        toggle.add_css_class("knowledge-packs-switch")
        remove = self._list_remove_button("Remove custom entry")

        def toggled(button, _):
            active = button.get_active()
            if kind == "mmproj":
                self._change_vision(
                    "Enable" if active else "Disable",
                    lambda: self.controller.select_custom_model_file(kind, path, active),
                )
            else:
                self.controller.select_custom_model_file(kind, path, active)
            self.refresh_model_configuration()

        toggle_handler = toggle.connect("notify::active", toggled)
        def remove_file(*_):
            def apply():
                self.controller.select_custom_model_file(kind, path, False, remove=True)

            if kind == "mmproj":
                self._change_vision("Remove", apply)
            else:
                apply()

        remove.connect("clicked", remove_file)
        row.add_suffix(toggle)
        row.add_suffix(remove)
        return row, toggle, toggle_handler, remove

    def _refresh_custom_file_rows(self, settings, config, external):
        files = settings.get("custom_model_files", {})
        busy = self.controller.model_setup_busy() or self.controller.switching_model
        profile = self.controller.model_recommendation()
        rule = MODEL_RECOMMENDATIONS[profile]
        recommended = {rule.balanced, rule.faster, rule.better}
        for kind, group in self._custom_file_groups.items():
            entries = files.get(kind, {})
            rows = self._custom_file_rows[kind]
            order = tuple(entries)
            if kind == "model":
                extras = (key for key in self._extra_model_rows if key not in recommended)
                order = self.controller.custom_models_in_order((*extras, *order), entries)
            if order != self._custom_file_order.get(kind):
                for key in self._custom_file_order.get(kind, ()):
                    row = rows[key][0] if key in rows else self._extra_model_rows[key][0]
                    group.remove(row)
                rows.clear()
                for key in order:
                    if key in entries:
                        rows[key] = self._custom_file_row(kind, key, entries[key])
                    row = rows[key][0] if key in rows else self._extra_model_rows[key][0]
                    group.add(row)
                self._custom_file_order[kind] = order
            for path, (row, toggle, handler, remove) in rows.items():
                selected = config.get(kind) == path
                if kind == "model" and self.controller.model_switch_target is not None:
                    selected = self.controller.model_switch_target == path
                available = Path(path).expanduser().is_file()
                if available:
                    row.remove_css_class("dim-label")
                else:
                    row.add_css_class("dim-label")
                with toggle.handler_block(handler):
                    toggle.set_active(bool(selected))
                toggle.set_sensitive(
                    available and not busy and not external
                    and (kind != "mmproj" or bool(config.get("model")))
                )
                remove.set_sensitive(not busy and (not external or not selected))

    def _confirm_model_delete(self, key):
        asset = MODEL_DOWNLOADS[key]
        self._confirm_model_removal(
            f"{asset.name} ({asset.quantization})", lambda: self.controller.delete_model(key)
        )

    def _confirm_model_removal(self, name, remove):
        self._confirm_model_action(
            "Remove model?", f"Remove {name} from this computer?", "Remove", remove
        )

    def _confirm_model_action(self, heading, message, action, apply):
        if self._model_remove_dialog is not None:
            return
        alert = Adw.AlertDialog(heading=heading, body=message)
        alert.add_response("cancel", "Cancel")
        alert.add_response("apply", action)
        alert.set_default_response("cancel")
        alert.set_close_response("cancel")
        if action == "Remove":
            alert.set_response_appearance("apply", Adw.ResponseAppearance.DESTRUCTIVE)

        def chosen(alert, result):
            self._model_remove_dialog = None
            if alert.choose_finish(result) == "apply":
                apply()
                self.refresh_model_configuration()

        self._model_remove_dialog = alert
        alert.choose(self, None, chosen)

    @staticmethod
    def _settings_edit_button(title, set_editing, *, editing=False, done_label=None):
        edit_label = f"Edit {title}"
        done_label = done_label or f"Done editing {title}"
        edit = Gtk.Button(
            icon_name=DONE_EDITING_ICON if editing else EDIT_ICON,
            tooltip_text=done_label if editing else edit_label,
            valign=Gtk.Align.CENTER,
        )
        edit.add_css_class("flat")

        def toggle(*_):
            nonlocal editing
            if set_editing(not editing) is False:
                return
            editing = not editing
            edit.set_icon_name(DONE_EDITING_ICON if editing else EDIT_ICON)
            edit.set_tooltip_text(done_label if editing else edit_label)
            if not editing:
                edit.grab_focus()

        edit.connect("clicked", toggle)
        edit.connect("clicked", pulse_edit_button)
        return edit

    @staticmethod
    def _settings_entry_row(title, value):
        row = Adw.EntryRow(title=title, text=value, editable=False, activatable=False)
        row.add_css_class("settings-edit-entry")
        return row

    @staticmethod
    def _set_settings_entry_editing(row, editing, *, select_text=False):
        field = row.get_delegate()
        row.set_editable(editing)
        field.set_focusable(editing)
        field.set_can_target(editing)
        if editing:
            row.grab_focus_without_selecting()
            if select_text:
                row.select_region(0, -1)
        else:
            position = row.get_position()
            row.select_region(position, position)

    def _model_entry_row(self, title, value, reset):
        row = self._settings_entry_row(title, value)
        row.add_css_class("model-settings-row")
        row.add_suffix(reset)
        self._set_settings_entry_editing(row, False)
        edit = self._settings_edit_button(
            title, lambda editing: self._set_settings_entry_editing(row, editing, select_text=True)
        )
        row.connect(
            "entry-activated", lambda *_: edit.emit("clicked") if row.get_editable() else None
        )
        row.add_suffix(edit)
        hint_edit_button(row, edit, row.get_editable)
        return row

    def _model_settings_page(self):
        page = Adw.PreferencesPage(name="model", title="Model", icon_name="network-server-symbolic")
        page.add_css_class("model-settings-page")
        page.connect("map", self._sync_native_scroll)
        settings = self.controller.store.load_app_settings()
        drafts = settings.get("advanced_model_settings", {})
        config = settings.get("model_server", {}) | drafts.get("model_server", {})
        embedding = settings.get("embedding_model", {}) | drafts.get("embedding_model", {})
        self._custom_file_groups = {}
        self._custom_file_rows = {kind: {} for kind in ("executable", "model", "mmproj")}
        self._custom_file_order = {}

        def save(key, value):
            try:
                self.controller.save_advanced_model_settings("model_server", {key: value})
            except OSError as exc:
                self._show_alert("Could not save settings", str(exc))

        section = Adw.PreferencesGroup(title="Configured Local Server")
        section.add_css_class("model-settings-summary")
        self._configured_server = Adw.ActionRow(use_markup=False)
        self._configured_server.add_css_class("model-settings-summary-row")
        section.add(self._configured_server)
        page.add(section)
        self._server_download_arrow = Gtk.Image(
            icon_name="go-next-symbolic", valign=Gtk.Align.CENTER
        )
        self._configured_server.add_suffix(self._server_download_arrow)
        self._server_download_indicator = add_operation_indicator(
            self._configured_server, visible=False
        )
        self._configured_server.connect(
            "activated",
            lambda *_: self.confirm_model_download(reconfigure=self._auto_configure_manual),
        )
        self._custom_server_add = self._add_custom_file_button(
            section, "executable", "llama-server"
        )
        page.add(self._model_choices())
        page.add(self._vision_choices())
        page.add(self._embedding_choice())
        page.connect("map", lambda *_: self.refresh_model_configuration())

        advanced_group = Adw.PreferencesGroup()
        advanced_group.add_css_class("model-settings-advanced-group")
        advanced = Adw.ExpanderRow(title="Advanced", expanded=False)
        advanced.add_css_class("model-settings-advanced")
        advanced.add_css_class("model-advanced")
        advanced_group.add(advanced)
        page.add(advanced_group)
        advanced_content = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        advanced_content.add_css_class("model-settings-advanced-content")
        advanced.add_row(Gtk.ListBoxRow(
            child=advanced_content, activatable=False, selectable=False, focusable=False
        ))

        reasoning = Adw.PreferencesGroup()
        reasoning.add_css_class("model-settings-reasoning")
        advanced_content.append(reasoning)
        self._external_server_row = Adw.SwitchRow(
            active=bool(config.get("external")),
            subtitle="Press Connect to apply changes.",
        )
        self._update_external_server_title(server_parameters(config)["port"])
        self._external_server_row.connect(
            "notify::active", lambda row, _: save("external", row.get_active())
        )
        reasoning.add(self._external_server_row)
        preserve_row = Adw.ActionRow(
            title="Preserve Reasoning",
            subtitle="Include earlier thinking in future replies when the model supports it.",
        )
        preserve = Gtk.Switch(
            active=self.controller.get_chat_settings()["preserve_reasoning"] is True,
            valign=Gtk.Align.CENTER,
        )
        preserve_row.set_activatable_widget(preserve)
        preserve_row.add_css_class("model-settings-preserve")
        preserve_default = self.controller.settings.preserve_thinking
        preserve_reset = self._model_reset_button(
            "Preserve Reasoning", lambda: preserve.set_active(preserve_default)
        )
        preserve_reset.set_visible(preserve.get_active() != preserve_default)
        preserve_row.add_suffix(preserve_reset)
        preserve_row.add_suffix(preserve)

        def save_reasoning(row, _pspec):
            active = row.get_active()
            try:
                self.controller.set_setting("preserve_reasoning", active)
            except OSError as exc:
                with row.handler_block(reasoning_handler):
                    row.set_active(not active)
                self._show_alert("Could not save settings", str(exc))
            preserve_reset.set_visible(row.get_active() != preserve_default)

        reasoning_handler = preserve.connect("notify::active", save_reasoning)
        reasoning.add(preserve_row)
        self._model_parameter_rows = {}
        defaults = self._model_parameter_config(config)
        self._model_parameter_identity = defaults["recommendation"], defaults["model"]
        self._model_parameter_defaults = server_parameters(
            {"recommendation": defaults["recommendation"], "model": defaults["model"]}
        )
        values = server_parameters(defaults)
        for heading, fields in (
            ("Launch Parameters", SERVER_FIELDS),
            ("Sampling Parameters", SAMPLING_FIELDS),
        ):
            options = Adw.PreferencesGroup(
                title=heading,
                description="Clear a text field to restore its default. "
                "Press Connect to apply changes.",
            )
            keys = tuple(key for key, _, _ in fields)
            self._add_model_reset_button(
                options, lambda keys=keys: self._reset_model_parameters(keys)
            )
            options.add_css_class("model-settings-options")
            advanced_content.append(options)
            for key, title, hint in fields:
                value = str(values[key])
                reset = self._model_reset_button(
                    title, lambda key=key: self._reset_model_parameters((key,))
                )
                reset.set_visible(self._has_model_override(key, value))
                if choices := SERVER_FIELD_CHOICES.get(key):
                    row = Adw.ActionRow(title=title, use_markup=False)
                    row.add_css_class("model-settings-row")
                    widget = Gtk.DropDown(valign=Gtk.Align.CENTER)
                    row.add_suffix(reset)
                    row.add_suffix(widget)
                    self._set_model_parameter_choice(widget, choices, value)

                    def selected(entry, _, key=key, reset=reset):
                        if item := entry.get_selected_item():
                            value = item.get_string()
                            save(key, value)
                            reset.set_visible(self._has_model_override(key, value))

                    handler = widget.connect("notify::selected-item", selected)
                else:
                    widget = row = self._model_entry_row(title, value, reset)

                    def changed(entry, _, key=key, reset=reset):
                        value = entry.get_text()
                        save(key, value)
                        reset.set_visible(self._has_model_override(key, value))
                        if key == "port":
                            self._update_external_server_title(value)

                    handler = row.connect("notify::text", changed)
                row.set_tooltip_text(f"--{key}: {hint}")
                self._model_parameter_rows[key] = widget, handler, reset
                options.add(row)

        extra = Adw.PreferencesGroup(
            title="Additional Launch Parameters",
            description="For example: --reasoning on --reasoning-budget 8192. "
            "Quote paths containing spaces.",
        )
        extra.add_css_class("model-settings-extra")
        advanced_content.append(extra)
        self._launch_parameters_box(
            extra, config.get("extra", ""), "", lambda value: save("extra", value)
        )
        actions = Adw.PreferencesGroup()
        actions.add_css_class("model-settings-actions")
        self._model_server_status = Gtk.Label(
            label=self.controller.model_server.status, wrap=True, xalign=0
        )
        self._model_server_status.add_css_class("model-server-status")
        self._model_server_status.add_css_class("dim-label")
        actions.add(self._model_server_status)
        self._model_connect_button = Gtk.Button(
            label="Connect", halign=Gtk.Align.END, sensitive=not self.controller.model_server.busy
        )
        self._model_connect_button.add_css_class("model-connect-button")
        self._model_connect_button.add_css_class("suggested-action")
        self._model_connect_button.connect(
            "clicked", lambda *_: self.controller.connect_model_server()
        )
        actions.add(self._model_connect_button)
        self._embedding_parameters = Adw.PreferencesGroup(title="Embedding Launch Parameters")
        self._embedding_parameters.add_css_class("model-settings-extra")

        def save_embedding_parameters(value):
            try:
                self.controller.save_advanced_model_settings(
                    "embedding_model", {"parameters": value}
                )
            except OSError as exc:
                self._show_alert("Could not save settings", str(exc))

        embedding_defaults = embedding_launch_parameters()
        self._launch_parameters_box(
            self._embedding_parameters, embedding.get("parameters") or embedding_defaults,
            embedding_defaults, save_embedding_parameters,
        )
        advanced_content.append(self._embedding_parameters)
        advanced_content.append(actions)
        self.refresh_model_configuration()
        return page

    def _launch_parameters_box(self, group, value, default, save):
        text = Gtk.TextView(wrap_mode=Gtk.WrapMode.WORD_CHAR)
        text.add_css_class("model-settings-text")
        text.add_css_class("prompt-view")
        buffer = text.get_buffer()
        buffer.set_text(value)
        title = group.get_title()
        reset = self._model_reset_button(title, lambda: buffer.set_text(default))
        reset.set_visible(bool(value.strip() and value.strip() != default.strip()))

        def changed(buffer):
            value = buffer.get_text(*buffer.get_bounds(), False)
            overridden = bool(value.strip() and value.strip() != default.strip())
            save(value if overridden else "")
            reset.set_visible(overridden)

        buffer.connect("changed", changed)
        group.set_header_suffix(reset)
        scroll = Gtk.ScrolledWindow(
            min_content_height=95, max_content_height=200, propagate_natural_height=True, child=text
        )
        scroll.add_css_class("model-settings-scroll")
        frame = Gtk.Frame(child=scroll)
        frame.add_css_class("model-settings-frame")
        frame.add_css_class("composer-frame")
        group.add(frame)

    def _show_memories(self, settings_dialog):
        toolbar = Adw.ToolbarView()
        toolbar.add_css_class("settings-subpage-toolbar")
        header = Adw.HeaderBar()
        header.add_css_class("settings-subpage-header")
        toolbar.add_top_bar(header)
        page = Adw.PreferencesPage()
        page.add_css_class("settings-subpage-page")
        page.connect("map", self._sync_native_scroll)
        group = Adw.PreferencesGroup()
        group.add_css_class("settings-subpage-group")
        page.add(group)
        toolbar.set_content(page)
        group.set_description(
            "Edit facts, select Core for always-present memories, or delete them. "
            "Click the pencil to edit, then the checkmark to save."
        )
        add = self._add_list_button(group, "Add memory")

        def add_row(memory_id=None, text="", core=False):
            draft = memory_id is None
            summary = Adw.ActionRow(
                title=text, title_lines=0, use_markup=False, activatable=False, visible=not draft,
            )
            summary.add_css_class("memories-row")
            entry = self._settings_entry_row("Memory", text)
            entry.set_visible(draft)
            controls = Gtk.Box(css_classes=["memories-controls"])
            core_button = Gtk.CheckButton(
                label="Core", active=bool(core), css_classes=["memories-core-button"]
            )
            remove = self._list_remove_button("Discard new memory" if draft else "Delete memory")

            def set_editing(editing):
                nonlocal memory_id
                current, target = (summary, entry) if editing else (entry, summary)
                if not editing:
                    text = " ".join(entry.get_text().split())
                    try:
                        saved = self.controller.change_memory(
                            memory_id, text, core_button.get_active()
                        )
                    except Exception as exc:
                        self._show_alert("Could not change memory", str(exc))
                        return False
                    if not saved or saved[1] != text:
                        self._show_alert(
                            "Memory was not saved", "Check for duplicate or empty text."
                        )
                        return False
                    if memory_id is None:
                        memory_id = saved[0]
                        add.set_sensitive(True)
                        remove.set_tooltip_text("Delete memory")
                    with core_button.handler_block(core_handler):
                        core_button.set_active(bool(saved[2]))
                    summary.set_title(text)
                current.remove(controls)
                target.add_suffix(controls)
                current.set_visible(False)
                target.set_visible(True)
                self._set_settings_entry_editing(entry, editing)
                return True

            def core_changed(button):
                if memory_id is None:
                    return
                try:
                    self.controller.set_memory_core(memory_id, button.get_active())
                except Exception as exc:
                    with button.handler_block(core_handler):
                        button.set_active(not button.get_active())
                    self._show_alert("Could not change core memory", str(exc))

            def remove_memory(_button):
                if memory_id is None:
                    add.set_sensitive(True)
                else:
                    try:
                        self.controller.change_memory(memory_id, "", False, delete=True)
                    except Exception as exc:
                        self._show_alert("Could not remove memory", str(exc))
                        return
                group.remove(summary)
                group.remove(entry)

            core_handler = core_button.connect("toggled", core_changed)
            edit = self._settings_edit_button(
                "memory", set_editing, editing=draft, done_label="Save memory"
            )
            hint_edit_button(summary, edit, entry.get_visible)
            entry.connect("entry-activated", lambda *_: edit.emit("clicked"))
            remove.connect("clicked", remove_memory)
            controls.append(edit)
            controls.append(core_button)
            controls.append(remove)
            (entry if draft else summary).add_suffix(controls)
            group.add(summary)
            group.add(entry)
            self._set_settings_entry_editing(entry, draft)

        for record in self.controller.list_memories():
            add_row(*record)

        def start_draft(button):
            button.set_sensitive(False)
            add_row()

        add.connect("clicked", start_draft)
        settings_dialog.push_subpage(Adw.NavigationPage.new(toolbar, "Global Memories"))

    def _confirm_delete_chat(self, chat_id):
        alert = Adw.AlertDialog(
            heading="Delete Chat?",
            body="Delete this chat and its chat folder? This cannot be undone.",
        )
        alert.add_response("cancel", "Cancel")
        alert.add_response("delete", "Delete Chat")
        alert.set_response_appearance("delete", Adw.ResponseAppearance.DESTRUCTIVE)
        alert.set_default_response("cancel")
        alert.set_close_response("cancel")

        def confirmed(dialog, result):
            if dialog.choose_finish(result) == "delete":
                self.controller.delete_chat(chat_id)

        alert.choose(self, None, confirmed)

    def _knowledge_settings_page(self, dialog):
        page = Adw.PreferencesPage(title="Knowledge", icon_name="application-x-addon-symbolic")
        page.add_css_class("knowledge-page")
        page.connect("map", self._sync_native_scroll)
        group = Adw.PreferencesGroup(title="Knowledge Packs")
        group.add_css_class("knowledge-group")
        page.add(group)

        def load(page):
            page.disconnect(load_handler)
            self._populate_knowledge_packs(dialog, group)

        load_handler = page.connect("map", load)
        return page

    def _refresh_knowledge_activity(self):
        activities = []
        for job in self._knowledge_jobs:
            if not job.get("active") or job["kind"] == "toggle":
                continue
            match job["kind"]:
                case "delete":
                    activities.append((20, "Removing knowledge packs"))
                case "merge":
                    activities.append((20, "Merging knowledge packs"))
                case _:
                    status = job["status"]
                    if status.startswith("Cancelling"):
                        activities.append((20, "Cancelling download"))
                    elif "Downloading" in status:
                        activities.append((30, "Downloading files"))
                    else:
                        activities.append((20, "Installing knowledge packs"))
        priority, text = max(activities, key=lambda activity: activity[0], default=(0, ""))
        self.set_background_activity("knowledge", text, priority)

    def _populate_knowledge_packs(self, dialog, group):
        group.set_description("Download and manage knowledge packs.")
        from packs import compatibility_error, load_catalog

        try:
            catalog = load_catalog()
        except OSError, ValueError:
            catalog = {}
        rows = []
        indicators = {}
        merge = Gtk.Button(
            label="Merge Installed Packs",
            tooltip_text="Reduces disk usage, REQUIRES LONG PROCESSING TIME",
        )
        merge.add_css_class("knowledge-packs-merge")
        actions = Gtk.Box()
        actions.add_css_class("knowledge-packs-actions")
        actions.append(merge)
        group.set_header_suffix(actions)
        if not hasattr(self, "_knowledge_jobs"):
            self._knowledge_jobs = []
        self._knowledge_refresh = None
        is_open = True
        merge_confirming = False

        def closed(*_):
            nonlocal is_open
            is_open = False
            if self._knowledge_refresh == refresh:
                self._knowledge_refresh = None

        dialog.connect("closed", closed)

        def refresh(job=None):
            if not is_open:
                return
            if job is not None:
                # Percentage ticks don't change pack membership or controls.
                if label := indicators.get(id(job)):
                    label.set_text(job["status"])
                return
            indicators.clear()
            for row in rows:
                group.remove(row)
            rows.clear()
            jobs = self._knowledge_jobs
            installed = self.controller.knowledge.list_packs()
            merge.set_sensitive(
                len(installed) >= 2 and not merge_confirming
                and not any(job["kind"] == "merge" for job in jobs)
            )
            covered = {
                key: manifest["title"]
                for _, manifest, _ in installed
                for key in manifest.get("contains", [])
            }
            for key, manifest, enabled in installed:
                if manifest.get("hidden"):
                    continue
                subtitle = format_size(manifest["size_bytes"])
                release = catalog.get(key)
                compatible = release is not None and not compatibility_error(release)
                newer = compatible and release["release_sequence"] > manifest.get(
                    "release_sequence", 0
                )
                if newer:
                    subtitle += " · Update available"
                error = manifest.get("compatibility_error")
                row = Adw.ActionRow(
                    title=manifest["title"], subtitle=error or subtitle, use_markup=False
                )
                row.add_css_class("knowledge-packs-row")
                if manifest.get("contains"):
                    info = Gtk.Label(label="Hover for pack information", valign=Gtk.Align.CENTER)
                    info.add_css_class("knowledge-packs-info")
                    titles = manifest.get("constituent_titles", {})
                    info.set_tooltip_text(
                        "\n".join(
                            titles.get(k, catalog.get(k, {}).get("title", k))
                            for k in manifest["contains"]
                        )
                    )
                    row.add_suffix(info)
                pending = next((job for job in jobs if job.get("key") == key), None)
                preferred = manifest.get("preferred_enabled", enabled)
                switch = Gtk.Switch(
                    active=pending.get("enabled", preferred) if pending else preferred,
                    valign=Gtk.Align.CENTER, focus_on_click=False,
                )
                switch.add_css_class("knowledge-packs-switch")
                if error:
                    switch.set_sensitive(False)
                if manifest.get("unused_by"):
                    row.add_css_class("dim-label")
                    switch.set_sensitive(False)
                    row.set_tooltip_text(
                        f"This pack is unused while {manifest['unused_by']} is enabled."
                    )
                if pending:
                    if pending["kind"] != "toggle":
                        add_status(row, pending)
                    switch.set_sensitive(False)

                def toggled(button, _, k=key, title=manifest["title"]):
                    enqueue({
                        "kind": "toggle", "key": k, "title": title,
                        "enabled": button.get_active(),
                    })

                switch.connect("notify::active", toggled)
                if compatible and (error or newer) and not pending:
                    row.add_suffix(pack_action(release, None, update=True))
                row.add_suffix(switch)
                row.add_suffix(
                    pack_action(
                        release,
                        pending,
                        remove=lambda k=key, title=manifest["title"]: enqueue(
                            {"kind": "delete", "key": k, "title": title}
                        ),
                    )
                )
                group.add(row)
                rows.append(row)
            installed_ids = {key for key, _, _ in installed}
            for key, release in catalog.items():
                if key in installed_ids or release.get("hidden"):
                    continue
                error = compatibility_error(release)
                row = Adw.ActionRow(
                    title=release["title"],
                    subtitle=error or format_size(release["size_bytes"]),
                    use_markup=False,
                )
                row.add_css_class("knowledge-packs-row")
                row.add_css_class("dim-label")
                pending = next((job for job in jobs if job.get("key") == key), None)
                if pending:
                    add_status(row, pending)
                if key in covered:
                    row.set_tooltip_text(f"Included in {covered[key]}.")
                row.add_suffix(pack_action(release, pending, covered=key in covered))
                group.add(row)
                rows.append(row)
            for job in jobs:
                if job["kind"] != "merge":
                    continue
                row = Adw.ActionRow(title=job["title"], use_markup=False)
                row.add_css_class("knowledge-packs-row")
                add_status(row, job)
                group.add(row)
                rows.append(row)

        def format_size(size):
            for unit in ("B", "KB", "MB", "GB", "TB"):
                if size < 1024 or unit == "TB":
                    return f"{size:.1f} {unit}" if unit != "B" else f"{size} B"
                size /= 1024
            return "Unknown size"

        def pack_action(release, pending, *, remove=None, update=False, covered=False):
            def download():
                enqueue(
                    {
                        "kind": "download",
                        "key": release["pack_id"],
                        "title": release["title"],
                        "release": release,
                    }
                )

            def cancel_download():
                if pending["cancel"].is_set():
                    return
                pending["cancel"].set()
                if pending.get("active"):
                    pending["status"] = "Cancelling…"
                else:
                    self._knowledge_jobs.remove(pending)
                    start_next()
                notify()

            button = DownloadAction(
                "Pack", download=download, update=download, remove=remove, cancel=cancel_download
            )
            if pending and "cancel" in pending and (
                pending.get("stage") is None or "Downloading" in pending["stage"]
            ):
                button.show_action("cancel", enabled=not pending["cancel"].is_set())
            elif remove:
                button.show_action("remove", enabled=not pending)
            else:
                button.show_action(
                    "update" if update else "download",
                    enabled=not (pending or covered or compatibility_error(release)),
                )
            return button

        def add_status(row, job):
            label, _ = add_operation_indicator(row, job["status"], job.get("active", False))
            indicators[id(job)] = label

        def notify(job=None):
            # Percentage-only updates change the pack row, not the composer activity.
            if job is None:
                self._refresh_knowledge_activity()
            if self._knowledge_refresh:
                self._knowledge_refresh(job)

        def enqueue(job):
            key = job.get("key")
            if any(existing.get("key") == key for existing in self._knowledge_jobs):
                return
            if job["kind"] in ("download", "dependencies"):
                job["cancel"] = threading.Event()
            job["status"] = "Waiting…"
            self._knowledge_jobs.append(job)
            notify()
            start_next()

        def start_next():
            if any(j.get("active") and j["kind"] == "merge" for j in self._knowledge_jobs):
                return
            for job in self._knowledge_jobs:
                if job.get("active"):
                    continue
                if job["kind"] == "merge" and any(j.get("active") for j in self._knowledge_jobs):
                    return
                start_job(job)
                if job["kind"] == "merge":
                    return

        def start_job(job):
            job["active"] = True
            job["status"] = {
                "download": "Downloading…",
                "dependencies": "Downloading shared Wikipedia data…",
                "delete": "Deleting…",
                "merge": "Merging…",
                "toggle": None,
            }[job["kind"]]
            notify()

            last_update = 0.0

            def progress(stage, current, total, title):
                nonlocal last_update
                now = time.monotonic()
                if current is not None and current != total and now - last_update < 0.2:
                    return
                last_update = now

                def update():
                    if "cancel" in job and job["cancel"].is_set():
                        return
                    same_stage = job.get("stage") == stage and (not title or title == job["title"])
                    job["stage"] = stage
                    if title:
                        job["title"] = title
                    job["status"] = (
                        f"{stage}: {current * 100 // total}%"
                        if current is not None and total
                        else f"{stage}…"
                    )
                    notify(job if same_stage else None)

                self.dispatch(update)

            def finish(error):
                self._knowledge_jobs.remove(job)
                notify()
                if error:
                    self._show_alert("Knowledge pack operation failed", error)
                start_next()

            def run():
                error = None
                stopped = job["cancel"].is_set if "cancel" in job else lambda: False
                try:
                    if job["kind"] == "download":
                        self.controller.knowledge.download_pack(
                            job["release"], progress, stopped=stopped
                        )
                    elif job["kind"] == "dependencies":
                        self.controller.knowledge.ensure_dependencies(progress, stopped=stopped)
                    elif job["kind"] == "merge":
                        self.controller.knowledge.merge_all(progress, list(catalog.values()))
                    else:
                        self.controller.knowledge.change(job["key"], job.get("enabled"))
                except InterruptedError as exc:
                    if not stopped():
                        error = str(exc)
                except Exception as exc:
                    error = str(exc)
                self.dispatch(finish, error)

            threading.Thread(target=run, daemon=True).start()

        def confirm_merge(*_):
            nonlocal merge_confirming
            if merge_confirming:
                return
            merge_confirming = True
            merge.set_sensitive(False)
            alert = Adw.AlertDialog(
                heading="Merge Installed Packs?",
                body=(
                    "Merging packs reduces disk usage. CAN TAKE HOURS TO COMPLETE. "
                    "Merged packs cannot be unmerged once completed. Lopaca "
                    "and your existing packs will remain usable during merging."
                ),
            )
            alert.add_response("cancel", "Cancel")
            alert.add_response("proceed", "Proceed")
            alert.set_default_response("cancel")
            alert.set_close_response("cancel")

            def chosen(alert, result):
                nonlocal merge_confirming
                merge_confirming = False
                if alert.choose_finish(result) == "proceed":
                    enqueue({"kind": "merge", "title": "Merging Installed Packs"})
                else:
                    notify()

            alert.choose(self, None, chosen)

        merge.connect("clicked", confirm_merge)
        self._knowledge_refresh = refresh
        refresh()

        def fetch_catalog():
            try:
                updated = load_catalog(refresh=True)
            except (OSError, ValueError) as exc:

                def failed(error=str(exc)):
                    if is_open:
                        group.set_description(f"Could not refresh catalog: {error}")

                self.dispatch(failed)
                return

            def apply_catalog():
                nonlocal catalog
                catalog = updated
                if self.controller.knowledge.missing_dependencies():
                    installed = self.controller.knowledge.list_packs()
                    consumer = next((k for k, m, _ in installed if m.get("requires")), None)
                    if consumer and not any(j.get("key") == consumer for j in self._knowledge_jobs):
                        enqueue(
                            {
                                "kind": "dependencies",
                                "key": consumer,
                                "title": "Shared Wikipedia data",
                            }
                        )
                if is_open:
                    group.set_description("Download and manage knowledge packs.")
                refresh()

            self.dispatch(apply_catalog)

        threading.Thread(target=fetch_catalog, daemon=True).start()

    def _show_document_drop(self, chat_id, *, attach=False):
        dialog = Adw.Dialog(title="Add Documents", content_width=520)
        toolbar = Adw.ToolbarView()
        toolbar.add_css_class("document-drop-toolbar")
        toolbar.add_top_bar(Adw.HeaderBar())
        box = Gtk.Box(
            orientation=Gtk.Orientation.VERTICAL, valign=Gtk.Align.FILL, height_request=160
        )
        box.add_css_class("document-drop-box")
        center = Gtk.Box(
            orientation=Gtk.Orientation.VERTICAL, valign=Gtk.Align.CENTER, vexpand=True
        )
        center.add_css_class("document-drop-center")
        center.append(
            Gtk.Image(icon_name="folder-download-symbolic", css_classes=["document-drop-icon"])
        )
        center.append(
            Gtk.Label(
                label="Drag and drop files and folders here to add to this chat",
                wrap=True,
                justify=Gtk.Justification.CENTER,
            )
        )
        box.append(center)
        target = Gtk.DropTarget.new(Gdk.FileList, Gdk.DragAction.COPY)

        def dropped(_target, files, _x, _y):
            paths = [path for file in files.get_files() if (path := file.get_path())]
            if not paths:
                return False
            self.controller.add_documents(paths, chat_id, attach=attach)
            dialog.close()
            return True

        target.connect("drop", dropped)
        box.add_controller(target)
        toolbar.set_content(box)
        dialog.set_child(toolbar)
        dialog.present(self)

    def show_document_warning(self, paths):
        detail = (
            "Readable text, supported documents, and supported images can be added. "
            "These files are unsupported or could not be read and were not copied:\n\n"
            + "\n".join(str(path) for path in paths)
        )
        self._show_alert("Some documents could not be added", detail)

    def _review_command(self, button, command):
        if self._review_pending:
            self.notify("Command review already in progress.")
            return
        self._review_pending = True
        chat_id = self.chat["id"]
        button.set_sensitive(False)

        def reviewed(result, error):
            self._review_pending = False
            button.set_sensitive(True)
            if error:
                self._show_alert("Could not inspect command", error)
            elif self.chat["id"] != chat_id:
                self.notify("Chat changed; review the command again.")
            else:
                self._command_reviewed(result)

        self.controller.review_command_async(command, reviewed)

    def _command_reviewed(self, result):
        kind, payload, risks, command = result
        if kind == "file":
            path, contents = payload
            self._show_command_approval(
                "Create this file?",
                f"File: {path}",
                command,
                "Create file",
                lambda: self.controller.create_file(path, contents),
                risks,
            )
            return
        if kind == "error":
            self.notify(payload)
            self._show_alert("Cannot prepare command", payload)
            return
        self._show_command_approval(
            "Run this command?",
            "Review the exact command before approving.",
            command,
            "Run command",
            lambda: self.controller.run_command(payload),
            risks,
        )

    def _show_command_approval(self, title, description, command, action_label, action, risks):
        # GNOME attaches modal windows to their parent's center while resizing.
        # Use a normal transient window and block parent interaction explicitly.
        dialog = Adw.Window(title=title, transient_for=self, modal=False, destroy_with_parent=True)
        toolbar = Adw.ToolbarView()
        toolbar.add_css_class("command-review-toolbar")
        toolbar.add_top_bar(Adw.HeaderBar())
        dialog.set_content(toolbar)

        def close_preview(*_):
            self.set_sensitive(True)
            dialog.destroy()

        def close_requested(*_):
            self.set_sensitive(True)
            return False

        dialog.connect("close-request", close_requested)
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        box.add_css_class("command-review-box")
        toolbar.set_content(box)
        for text in (
            description,
            f"Working directory: {self.controller.store.projects(self.chat['id'])}",
        ):
            label = Gtk.Label(label=text, xalign=0)
            label.add_css_class("command-review-label")
            label.set_wrap(True)
            label.set_wrap_mode(Pango.WrapMode.WORD_CHAR)
            label.set_selectable(True)
            box.append(label)
        risk_details = list(dict.fromkeys(f"{risk.level.title()}: {risk.reason}" for risk in risks))
        risk_summary = "\n".join(risk_details)
        if risks:
            legend = Gtk.Label(
                label="POTENTIALLY DANGEROUS CODE DETECTED\n"
                "Basic scan only; unhighlighted code can still be dangerous.",
                xalign=0,
            )
            legend.add_css_class("command-review-legend")
            legend.set_wrap(True)
            box.append(legend)
        lines = command.count("\n") + 1
        if lines > 16:
            notice = Gtk.Label(
                label=f"↕ {lines} lines, scroll inside the preview to review the full command.",
                xalign=0,
            )
            notice.add_css_class("command-review-notice")
            notice.set_wrap(True)
            notice.add_css_class("heading")
            box.append(notice)
        preview = Gtk.TextView(editable=False, cursor_visible=False)
        preview.add_css_class("command-review-preview")
        preview.set_wrap_mode(Gtk.WrapMode.NONE)
        buffer = preview.get_buffer()
        buffer.set_text(command)
        tags = {}
        for level, color in (
            ("warning", "rgba(255,226,0,0.6)"),
            ("danger", "rgba(229,57,53,0.6)"),
        ):
            shade = Gdk.RGBA()
            shade.parse(color)
            tags[level] = buffer.create_tag(f"risk-{level}", background_rgba=shade)
        for risk in risks:
            buffer.apply_tag(
                tags[risk.level],
                buffer.get_iter_at_offset(risk.start),
                buffer.get_iter_at_offset(risk.end),
            )
        if risks:
            preview.set_tooltip_text(risk_summary)
        context = preview.get_pango_context()
        font = context.get_font_description().copy()
        font.set_family("monospace")
        if font.get_size() <= 0:
            font.set_size(11 * Pango.SCALE)
        metrics = context.get_metrics(font, None)
        line_height = max(
            1, (metrics.get_ascent() + metrics.get_descent() + Pango.SCALE - 1) // Pango.SCALE
        )
        layout = preview.create_pango_layout(command)
        layout.set_font_description(font)
        command_width, _ = layout.get_pixel_size()
        # Include dialog margins, frame, and a vertical scrollbar in the initial width.
        dialog_width = max(440, min(1000, command_width + 64))
        scroll = Gtk.ScrolledWindow()
        scroll.add_css_class("command-review-scroll")
        # AUTOMATIC reserves a vertical scrollbar's minimum length even when
        # hidden. EXTERNAL keeps short previews at their actual text height;
        # wheel scrolling still works if a two-line preview is manually shrunk.
        vertical_policy = (
            Gtk.PolicyType.EXTERNAL
            if lines <= 2
            else Gtk.PolicyType.ALWAYS
            if lines > 16
            else Gtk.PolicyType.AUTOMATIC
        )
        scroll.set_policy(Gtk.PolicyType.AUTOMATIC, vertical_policy)
        scroll.set_overlay_scrolling(False)
        preview_height = min(lines, 16) * line_height
        scroll.set_min_content_height(line_height)
        scroll.set_vexpand(True)
        scroll.set_child(preview)
        frame = Gtk.Frame()
        frame.add_css_class("command-review-frame")
        frame.add_css_class("command-preview")
        frame.set_vexpand(True)
        frame.set_child(scroll)
        box.append(frame)
        buttons = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
        buttons.add_css_class("command-review-buttons")
        buttons.set_halign(Gtk.Align.END)
        cancel_button = Gtk.Button(label="Cancel")
        cancel_button.add_css_class("command-review-cancel-button")
        cancel_button.connect("clicked", close_preview)
        buttons.append(cancel_button)
        approve_button = Gtk.Button(label=action_label)
        approve_button.add_css_class("command-review-approve-button")
        approve_button.add_css_class(
            "destructive-action"
            if any(risk.level == "danger" for risk in risks)
            else "suggested-action"
        )

        def confirmed(alert, result):
            approve_button.set_sensitive(True)
            try:
                response = alert.choose_finish(result)
            except GLib.Error:
                return
            if response == "proceed":
                close_preview()
                action()

        def approve(_button):
            if not risks:
                close_preview()
                action()
                return
            alert = Adw.AlertDialog(
                heading="Potentially Dangerous Commands",
                body="Warning: potentially dangerous commands detected. Are you sure?",
            )
            warnings = Gtk.Label(label=risk_summary, xalign=0, wrap=True, selectable=True)
            warnings.add_css_class("command-review-warnings")
            warnings.set_wrap_mode(Pango.WrapMode.WORD_CHAR)
            alert.set_extra_child(warnings)
            alert.set_content_width(460)
            alert.set_follows_content_size(True)
            alert.add_response("cancel", "Cancel")
            alert.add_response("proceed", "Proceed")
            alert.set_response_appearance("proceed", Adw.ResponseAppearance.DESTRUCTIVE)
            alert.set_default_response("cancel")
            alert.set_close_response("cancel")
            approve_button.set_sensitive(False)
            # An embedded alert cannot grow beyond a short command preview.
            alert.choose(None, None, confirmed)
            alert_window = alert.get_root()
            alert_window.set_transient_for(dialog)
            alert_window.set_modal(True)
            alert_window.set_destroy_with_parent(True)

        approve_button.connect("clicked", approve)
        buttons.append(approve_button)
        box.append(buttons)
        # Measure the controls instead of leaving a fixed amount of empty window space.
        _, natural_height, _, _ = toolbar.measure(Gtk.Orientation.VERTICAL, dialog_width)
        horizontal_height = 0
        if command_width > dialog_width - 64:
            horizontal_height = max(
                16, scroll.get_hscrollbar().measure(Gtk.Orientation.VERTICAL, -1)[1]
            )
        dialog.set_default_size(
            dialog_width, natural_height + preview_height - line_height + horizontal_height
        )
        self.set_sensitive(False)
        dialog.present()

    def set_command_running(self, active):
        self._running_commands += 1 if active else -1
        self.set_background_activity(
            "commands", "Running command" if self._running_commands else "", 15,
        )

    def show_command_result(self, output, exit_code, note):
        message = (
            "Command completed successfully."
            if exit_code == 0
            else f"Command failed with exit code {exit_code}."
        )
        if note:
            message += f"\n\n{note}"
        elif output:
            message += f"\n\n```text\n{output}\n```"
        self.append_message("assistant", message)
        self.set_command_running(False)
        self.scroll_bottom()
        return False

    def show_command_error(self, error):
        self.set_command_running(False)
        self.notify(f"Command failed: {error}")
        self.append_message("assistant", f"Command could not be started: {error}")
        self.scroll_bottom()
        return False


class Lopaca(Adw.Application):
    def __init__(self, controller):
        super().__init__(application_id="app.lopaca.Lopaca")
        self.controller = controller
        GLib.set_application_name("Lopaca")

    def do_shutdown(self):
        self.controller.cancel_chat_warm()
        self.controller._model_file_stop.set()
        self.controller.recall_embeddings.close()
        self.controller.model_server.close()
        Adw.Application.do_shutdown(self)

    def do_activate(self):
        window = self.props.active_window
        if not window:
            window = ChatWindow(self, self.controller)
        window.present()
