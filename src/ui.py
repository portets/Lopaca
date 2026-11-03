"""GTK/libadwaita presentation layer. Application behavior lives in main.py."""

import html
import re

import gi

from settings import CHAT_SORT_OPTIONS, CONVERSATION_STYLE_OPTIONS

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
gi.require_version("Graphene", "1.0")
gi.require_version("Gsk", "4.0")
# GI namespace versions must be selected before importing their bindings.
from gi.repository import Adw, Gdk, Gio, GLib, Graphene, Gsk, Gtk, Pango  # noqa: E402

CSS = """
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
.sidebar {
    margin: 4px;
    padding: 4px 0;
    min-width: 286px;
}
.sidebar, .composer-frame {
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
    padding: 1px 5px;
    border-radius: 9px;
}
.chat-options-button {
    opacity: 0;
    padding: 0;
    min-width: 20px;
    min-height: 20px;
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
}
.message-row {
    padding: 5px 0;
}
.message {
    padding: 10px 13px;
    border-radius: 14px;
}
.user-message {
    background-color: alpha(@accent_bg_color,0.4);
}
.assistant-message {
    background-color: alpha(@theme_fg_color,0.045);
}
.message-role {
    font-size: 0.78em;
    font-weight: 700;
    opacity: 0.68;
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
.thinking {
    background-color: alpha(@theme_fg_color,0.065);
    border-left: 3px solid @accent_color;
    padding: 3px 10px;
    margin-bottom: 8px;
    border-radius: 5px;
}
.thinking button {
    min-height: 28px;
    padding: 2px 4px;
}
.thinking, .thinking button, .thinking label {
    font-weight: normal;
}
.thinking-dropdown label {
    font-weight: normal;
}
.muted {
    color: alpha(@window_fg_color,0.62);
    font-size: 0.9em;
}
.prompt-view, .prompt-view text {
    background-color: transparent;
}
.command-preview {
    border-radius: 6px;
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
"""

_CODE_BLOCK_PATTERN = re.compile(r"```[^\n]*\n(.*?)```", re.DOTALL)


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


def set_markdown_text(label, text):
    """Use identical safe inline formatting for streamed and saved messages."""
    value = html.escape(text)
    value = re.sub(r"^#{1,6} (.*)$", r"<b>\1</b>", value, flags=re.MULTILINE)
    value = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", value)
    value = re.sub(r"`([^`]+)`", r"<tt>\1</tt>", value)
    label.set_markup(value)


def table_cells(line):
    """Split Markdown cells, preserving escaped pipes as literal text."""
    cells = re.split(r"(?<!\\)\|", line.strip())
    if not cells[0]:
        cells.pop(0)
    if cells and not cells[-1]:
        cells.pop()
    return [cell.strip().replace(r"\|", "|") for cell in cells]


def markdown_blocks(text):
    """Yield prose and pipe tables; fenced code is handled by append_markdown."""
    lines = text.splitlines(keepends=True)
    start = index = 0
    while index + 1 < len(lines):
        header, separators = table_cells(lines[index]), table_cells(lines[index + 1])
        if (
            not re.search(r"(?<!\\)\|", lines[index])
            or len(header) != len(separators)
            or not separators
            or not all(re.fullmatch(r":?-+:?", cell) for cell in separators)
        ):
            index += 1
            continue
        if start < index:
            yield "prose", "".join(lines[start:index]).strip("\n")
        alignment = [
            0.5
            if cell.startswith(":") and cell.endswith(":")
            else 1.0
            if cell.endswith(":")
            else 0.0
            for cell in separators
        ]
        rows = [header]
        index += 2
        while (
            index < len(lines)
            and lines[index].strip()
            and ("|" in lines[index] or len(header) == 1)
        ):
            cells = table_cells(lines[index])
            rows.append((cells + [""] * len(header))[: len(header)])
            index += 1
        yield "table", (alignment, rows)
        start = index
    if start < len(lines):
        yield "prose", "".join(lines[start:]).strip("\n") if start else text


def markdown_label(text=""):
    label = Gtk.Label(xalign=0, yalign=0)
    label.set_wrap(True)
    label.set_wrap_mode(Pango.WrapMode.WORD_CHAR)
    label.set_selectable(True)
    set_markdown_text(label, text)
    return label


class HorizontalScrollFrame(Gtk.Frame):
    """Bordered Markdown content with a scrollbar at its visible bottom edge."""

    def __init__(self, child):
        super().__init__()
        self.scroll = Gtk.ScrolledWindow()
        self.scroll.set_policy(Gtk.PolicyType.EXTERNAL, Gtk.PolicyType.NEVER)
        self.scroll.set_propagate_natural_width(True)
        self.scroll.set_propagate_natural_height(True)
        self.scroll.set_child(child)
        self.bar = Gtk.Scrollbar(
            orientation=Gtk.Orientation.HORIZONTAL, adjustment=self.scroll.get_hadjustment()
        )
        self.bar.set_valign(Gtk.Align.END)
        self.bar.set_visible(False)
        self.overlay = Gtk.Overlay(child=self.scroll)
        self.overlay.add_overlay(self.bar)
        self.overlay.set_clip_overlay(self.bar, True)
        self.set_child(self.overlay)
        self._scroll_connections = []
        self._outer_scroll = None
        self.connect("map", self._connect_scrollbar)
        self.connect("unmap", self._disconnect_scrollbar)
        if isinstance(child, Gtk.Label):
            child.connect("notify::label", lambda *_: self.update_content_width())
            self.update_content_width()

    def update_content_width(self, child=None):
        """Scroll at the content's bounded natural width instead of wrapping to the bubble."""
        if child is None:
            child = self.scroll.get_child().get_child()
        child.set_size_request(-1, -1)
        child.set_size_request(child.measure(Gtk.Orientation.HORIZONTAL, -1)[1], -1)

    def _connect_scrollbar(self, _widget):
        self._outer_scroll = self.get_ancestor(Gtk.ScrolledWindow)
        adjustments = [(self.scroll.get_hadjustment(), "changed")]
        if self._outer_scroll is not None:
            vertical = self._outer_scroll.get_vadjustment()
            adjustments.extend([(vertical, "changed"), (vertical, "value-changed")])
        for adjustment, signal in adjustments:
            # Reposition before the next paint; deferring this shows the old position briefly.
            handler = adjustment.connect(signal, self._update_scrollbar)
            self._scroll_connections.append((adjustment, handler))
        self._update_scrollbar()

    def _disconnect_scrollbar(self, _widget):
        for adjustment, handler in self._scroll_connections:
            adjustment.disconnect(handler)
        self._scroll_connections.clear()
        self._outer_scroll = None

    def _update_scrollbar(self, *_args):
        horizontal = self.scroll.get_hadjustment()
        overflow = horizontal.get_upper() > horizontal.get_page_size() + 1
        self.bar.set_visible(overflow)
        bar_height = (
            self.bar.measure(Gtk.Orientation.VERTICAL, -1)[1] - self.bar.get_margin_bottom()
            if overflow
            else 0
        )
        self.scroll.set_margin_bottom(bar_height)
        margin = 0
        if overflow and self._outer_scroll is not None:
            content = self._outer_scroll.get_child()
            if isinstance(content, Gtk.Viewport):
                content = content.get_child()
            valid, position = self.overlay.compute_point(content, Graphene.Point().init(0, 0))
            if valid:
                # Include the content margin and match GTK's truncated scroll offset.
                vertical = self._outer_scroll.get_vadjustment()
                below_view = (
                    position.y
                    + content.get_margin_top()
                    + self.overlay.get_height()
                    - (int(vertical.get_value()) + vertical.get_page_size())
                )
                margin = min(
                    max(0, round(below_view)), max(0, self.overlay.get_height() - bar_height)
                )
        self.bar.set_margin_bottom(margin)
        return False


class MarkdownTable(HorizontalScrollFrame):
    """Keep columns aligned and update only changed cells during streaming."""

    def __init__(self, data):
        self.grid = Gtk.Grid()
        super().__init__(self.grid)
        self.add_css_class("markdown-table")
        self.cells = []
        self.data = ([], [])
        self.set_data(data)

    def set_data(self, data):
        alignment, rows = data
        previous_alignment, previous_rows = self.data
        if alignment != previous_alignment:
            while (child := self.grid.get_first_child()) is not None:
                self.grid.remove(child)
            self.cells.clear()
            previous_rows = []
        while len(self.cells) > len(rows):
            for label in self.cells.pop():
                self.grid.remove(label)
        for row_index, row in enumerate(rows):
            if row_index == len(self.cells):
                labels = []
                for column, xalign in enumerate(alignment):
                    label = markdown_label()
                    label.set_max_width_chars(500)
                    label.set_xalign(xalign)
                    label.set_hexpand(True)
                    label.add_css_class("table-cell")
                    if row_index == 0:
                        label.add_css_class("table-heading")
                    elif row_index % 2 == 0:
                        label.add_css_class("table-alternate")
                    self.grid.attach(label, column, row_index, 1, 1)
                    labels.append(label)
                self.cells.append(labels)
            for column, value in enumerate(row):
                if row_index >= len(previous_rows) or previous_rows[row_index][column] != value:
                    label = self.cells[row_index][column]
                    set_markdown_text(label, value)
                    self.update_content_width(label)
        self.data = data


class MarkdownText(Gtk.Box):
    """Share prose/table rendering between stored messages and the streaming tail."""

    def __init__(self, text=""):
        super().__init__(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        self.blocks = []
        self.set_text(text)

    def set_text(self, text):
        blocks = list(markdown_blocks(text))
        for index, (kind, value) in enumerate(blocks):
            if index >= len(self.blocks) or self.blocks[index][0] != kind:
                while len(self.blocks) > index:
                    self.remove(self.blocks.pop()[1])
                child = MarkdownTable(value) if kind == "table" else markdown_label(value)
                self.append(child)
            else:
                _, child, previous = self.blocks[index]
                if value != previous:
                    if kind == "table":
                        child.set_data(value)
                    else:
                        set_markdown_text(child, value)
            block = (kind, child, value)
            if index == len(self.blocks):
                self.blocks.append(block)
            else:
                self.blocks[index] = block
        while len(self.blocks) > len(blocks):
            self.remove(self.blocks.pop()[1])


def append_code_block(parent, text):
    label = Gtk.Label(label=text, xalign=0, yalign=0)
    label.set_wrap(True)
    label.set_wrap_mode(Pango.WrapMode.WORD_CHAR)
    label.set_max_width_chars(500)
    label.set_selectable(True)
    label.add_css_class("code-block")
    frame = HorizontalScrollFrame(label)
    frame.add_css_class("code-block-frame")
    parent.append(frame)
    return frame, label


def append_markdown(parent, text, command_extractor=None, command_handler=None):
    """Append safe Markdown, tables, and fenced code or command blocks."""

    last = 0
    for match in _CODE_BLOCK_PATTERN.finditer(text):
        if text[last : match.start()]:
            parent.append(MarkdownText(text[last : match.start()]))
        commands = command_extractor(match.group(0)) if command_extractor else []
        if commands:
            for command in commands:
                button = Gtk.Button(label="Review Command")
                button.set_halign(Gtk.Align.START)
                button.connect("clicked", command_handler, command)
                parent.append(button)
        else:
            append_code_block(parent, match.group(1).rstrip("\n"))
        last = match.end()
    if text[last:] or not last:
        parent.append(MarkdownText(text[last:]))


class ChatWindow(Adw.ApplicationWindow):
    def __init__(self, app, controller):
        super().__init__(application=app, title="Lopaca")
        self.add_css_class("lopaca-main")
        self.controller, self.chat = controller, None
        self.busy = self.agent_mode = False
        self.follow_output = True
        self.current_wrapper = self.current_answer = self.current_thinking = None
        self.context_capacity = self.context_used = self.tokens_per_second = None
        self._thinking_pending = []
        self._thinking_flush_id = None
        self._inband_thinking = ""
        self._status_text = None
        self._status_timer = None
        self._status_dots = 0
        self._activity_status = "Not Connected"
        self._processing_documents = False
        self._notification_timer = None
        self._notification_deadline = 0
        self._notification_model_busy = False
        self._response_row = None
        self._response_chat_id = None
        self._scroll_idle_id = None
        self._chat_rows = []
        self._dragged_row = None
        self._drag_pointer_y = None
        self._drag_scroll_id = None
        self.set_default_size(1180, 790)
        self.set_size_request(760, 520)
        self._apply_css()
        self._build_ui()
        controller.attach(self)
        self.chat_list.add_tick_callback(self._scroll_initial_chat)

    def _apply_css(self):
        provider = Gtk.CssProvider()
        provider.load_from_data(CSS.encode())
        Gtk.StyleContext.add_provider_for_display(
            Gdk.Display.get_default(), provider, Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION
        )

    def _build_ui(self):
        toolbar = Adw.ToolbarView()
        header_bar = Adw.HeaderBar()
        header_bar.add_css_class("app-titlebar")
        title = Gtk.Label(label="Lopaca")
        title.add_css_class("title")
        header_bar.set_title_widget(title)
        toolbar.add_top_bar(header_bar)
        root = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
        toolbar.set_content(root)
        self.set_content(toolbar)
        sidebar = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        sidebar.add_css_class("sidebar")
        sidebar.set_size_request(252, -1)
        sidebar.set_hexpand(False)
        sidebar.set_margin_top(16)
        sidebar.set_margin_bottom(14)
        sidebar.set_margin_start(12)
        sidebar.set_margin_end(10)
        root.append(sidebar)
        top = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        top.set_margin_top(4)
        top.set_margin_start(8)
        top.set_margin_end(6)
        heading = Gtk.Label(label="Chats", xalign=0)
        heading.set_hexpand(True)
        heading.add_css_class("title-2")
        top.append(heading)
        new_chat_button = Gtk.Button(label="New chat", icon_name="list-add-symbolic")
        new_chat_button.add_css_class("suggested-action")
        new_chat_button.set_tooltip_text("New Chat")
        new_chat_button.connect("clicked", lambda *_: self.controller.new_chat())
        top.append(new_chat_button)
        sidebar.append(top)
        self.chat_list = Gtk.ListBox()
        self.chat_list.set_selection_mode(Gtk.SelectionMode.SINGLE)
        self.chat_list.set_activate_on_single_click(True)
        self.chat_list.add_css_class("workspace-list")
        self.chat_list.connect(
            "row-activated", lambda _list, row: self.controller.open_chat(row.chat_id)
        )
        self.chat_list.set_margin_start(14)
        self.chat_list.set_margin_end(12)
        self.chat_list.set_header_func(self._set_chat_header)
        drop_target = Gtk.DropTarget.new(str, Gdk.DragAction.MOVE)
        drop_target.connect("enter", self._chat_drag_motion)
        drop_target.connect("motion", self._chat_drag_motion)
        drop_target.connect("leave", lambda *_: self._clear_drop_indicator())
        drop_target.connect("drop", self._chat_drop)
        self.chat_list.add_controller(drop_target)
        list_overlay = Gtk.Overlay()
        list_overlay.set_child(self.chat_list)
        self._drop_indicator = Gtk.Box()
        self._drop_indicator.add_css_class("chat-drop-indicator")
        self._drop_indicator.set_valign(Gtk.Align.START)
        self._drop_indicator.set_can_target(False)
        self._drop_indicator.set_visible(False)
        self._drop_indicator.set_margin_start(22)
        self._drop_indicator.set_margin_end(20)
        list_overlay.add_overlay(self._drop_indicator)
        self.chat_scroll = Gtk.ScrolledWindow()
        self.chat_scroll.add_css_class("undershoot-top")
        self.chat_scroll.add_css_class("undershoot-bottom")
        self.chat_scroll.set_child(list_overlay)
        self.chat_scroll.set_vexpand(True)
        self.chat_scroll.set_margin_bottom(0)
        sidebar.append(self.chat_scroll)
        footer = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        footer.set_margin_start(10)
        footer.set_margin_end(10)
        footer.set_margin_bottom(4)  # Smaller values move disk usage and settings downward.
        usage = Gtk.CenterBox()
        usage.add_css_class("muted")
        usage.set_hexpand(True)
        self.chat_usage = Gtk.Label(label="Chat: —", xalign=1)
        self.chat_usage.set_hexpand(True)
        self.total_usage = Gtk.Label(label="Total: —", xalign=0)
        self.total_usage.set_hexpand(True)
        usage.set_start_widget(self.chat_usage)
        usage.set_center_widget(Gtk.Label(label=" | "))
        usage.set_end_widget(self.total_usage)
        footer.append(usage)
        settings_button = Gtk.Button(
            icon_name="emblem-system-symbolic", tooltip_text="App Settings"
        )
        settings_button.add_css_class("flat")
        settings_button.connect("clicked", lambda *_: self._show_app_settings())
        footer.append(settings_button)
        sidebar.append(footer)
        root.append(Gtk.Separator(orientation=Gtk.Orientation.VERTICAL))
        main = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        main.set_hexpand(True)
        root.append(main)
        header = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        header.set_margin_start(20)
        header.set_margin_end(20)
        header.set_margin_top(12)
        header.set_margin_bottom(10)
        column = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=3)
        column.set_hexpand(True)
        self.chat_title = Gtk.Label(label="New chat", xalign=0)
        self.chat_title.add_css_class("title-3")
        column.append(self.chat_title)
        self.folder_label = Gtk.Label(label="No documents folder", xalign=0)
        self.folder_label.set_ellipsize(Pango.EllipsizeMode.END)
        self.folder_label.add_css_class("muted")
        column.append(self.folder_label)
        header.append(column)
        main.append(header)
        self.transcript = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2)
        self.transcript.add_css_class("transcript")
        self.transcript.set_margin_top(16)
        self.transcript.set_margin_bottom(18)
        self.transcript.set_margin_start(24)
        self.transcript.set_margin_end(24)
        self.transcript_scroll = Gtk.ScrolledWindow()
        self.transcript_scroll.add_css_class("transcript-scroll")
        self.transcript_scroll.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        self.transcript_scroll.set_child(self.transcript)
        self.transcript_scroll.set_vexpand(True)
        main.append(self.transcript_scroll)
        adjustment = self.transcript_scroll.get_vadjustment()
        self._scroll_guard = False
        adjustment.connect("value-changed", self._adjustment_value_changed)
        adjustment.connect("changed", self._adjustment_changed)
        wheel = Gtk.EventControllerScroll.new(Gtk.EventControllerScrollFlags.VERTICAL)
        wheel.set_propagation_phase(Gtk.PropagationPhase.CAPTURE)
        wheel.connect("scroll", self._prepare_native_scroll)
        self.transcript_scroll.add_controller(wheel)
        compose = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=7)
        compose.set_margin_start(14)
        compose.set_margin_end(14)
        compose.set_margin_top(14)
        compose.set_margin_bottom(18)
        status_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=14)
        self.status = Gtk.Label(label=" Not Connected", xalign=0)
        self.status.add_css_class("muted")
        status_row.append(self.status)
        status_spacer = Gtk.Box()
        status_spacer.set_hexpand(True)
        status_row.append(status_spacer)
        self.context_status = Gtk.Label(label="Context: —/—", xalign=1)
        self.context_status.add_css_class("muted")
        status_row.append(self.context_status)
        self.speed_status = Gtk.Label(label="— t/s", xalign=1)
        self.speed_status.add_css_class("muted")
        status_row.append(self.speed_status)
        self.speed_status.set_margin_end(
            self.speed_status.create_pango_layout(" ").get_pixel_size()[0]
        )
        compose.append(status_row)
        row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
        compose.append(row)
        frame = Gtk.Frame()
        frame.add_css_class("composer-frame")
        frame.set_hexpand(True)
        row.append(frame)
        inner = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=3)
        inner.set_margin_top(8)
        inner.set_margin_bottom(8)
        inner.set_margin_start(10)
        inner.set_margin_end(10)
        frame.set_child(inner)
        self.prompt = Gtk.TextView()
        self.prompt.set_wrap_mode(Gtk.WrapMode.WORD_CHAR)
        self.prompt.set_accepts_tab(False)
        self.prompt.add_css_class("prompt-view")
        _, line_height = self.prompt.create_pango_layout("Ag").get_pixel_size()
        prompt_height = max(1, line_height) * 4
        self.prompt.set_size_request(-1, prompt_height)
        self.prompt.get_buffer().connect("changed", lambda *_: self._prompt_changed())
        key_controller = Gtk.EventControllerKey.new()
        key_controller.connect("key-pressed", self._key_pressed)
        self.prompt.add_controller(key_controller)
        prompt_scroll = Gtk.ScrolledWindow()
        prompt_scroll.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        prompt_scroll.set_child(self.prompt)
        prompt_scroll.set_min_content_height(prompt_height)
        prompt_scroll.set_max_content_height(1160)
        inner.append(prompt_scroll)
        self.send_button = Gtk.Button(icon_name="go-up-symbolic", tooltip_text="Send")
        self.send_button.add_css_class("suggested-action")
        self.send_button.set_valign(Gtk.Align.END)
        self.send_button.connect(
            "clicked", lambda *_: self.controller.stop() if self.busy else self._send()
        )
        actions = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        hint = Gtk.Label(label="Enter for a new line · Ctrl+Enter to send", xalign=0)
        hint.set_hexpand(True)
        hint.add_css_class("muted")
        actions.append(hint)
        self.thinking_label = Gtk.Label(label="Thinking budget")
        self.thinking_label.add_css_class("muted")
        actions.append(self.thinking_label)
        self.thinking_dropdown = Gtk.DropDown.new_from_strings(["auto"])
        self.thinking_dropdown.add_css_class("thinking-dropdown")
        self.thinking_dropdown.set_valign(Gtk.Align.CENTER)
        self.thinking_dropdown.set_margin_end(8)
        self._thinking_values = [None]
        self._thinking_handler = self.thinking_dropdown.connect(
            "notify::selected", self._thinking_selected
        )
        actions.append(self.thinking_dropdown)
        actions.append(self.send_button)
        inner.append(actions)
        main.append(compose)
        self.send_button.add_css_class("send-stop")
        self._prompt_changed()

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
        self.thinking_dropdown.set_tooltip_text(title + " · auto chooses effort for each prompt")

    def _thinking_selected(self, dropdown, _param):
        selected = dropdown.get_selected()
        if selected >= len(self._thinking_values):
            return
        try:
            self.controller.set_thinking(self._thinking_values[selected])
        except OSError as exc:
            self.update_thinking_options()
            self._show_alert("Could not save thinking setting", str(exc))

    def dispatch(self, callback, *args):
        GLib.idle_add(callback, *args)

    def prompt_text(self, strip=True):
        buffer = self.prompt.get_buffer()
        start, end = buffer.get_bounds()
        text = buffer.get_text(start, end, True)
        return text.strip() if strip else text

    def set_prompt(self, text):
        self.prompt.get_buffer().set_text(text)

    def set_status(self, text):
        self._activity_status = text
        if (
            self._notification_timer is not None
            and self._notification_model_busy
            and not self.controller.busy
            and text in ("Ready", "Disconnected", "Not Connected")
            and self._notification_deadline - GLib.get_monotonic_time() > 1_000_000
        ):
            GLib.source_remove(self._notification_timer)
            self._notification_deadline = GLib.get_monotonic_time() + 1_000_000
            self._notification_timer = GLib.timeout_add(1000, self._restore_status)
        if self._notification_timer is None:
            self._display_status(
                "Assistant processing documents" if self._processing_documents else text
            )

    def set_document_processing(self, active):
        self._processing_documents = active
        if active and self._notification_timer is not None:
            GLib.source_remove(self._notification_timer)
            self._notification_timer = None
        self.set_status(self._activity_status)

    def notify(self, text):
        if self._notification_timer is not None:
            GLib.source_remove(self._notification_timer)
        self._display_status(text)
        self._notification_model_busy = self.controller.busy
        self._notification_deadline = GLib.get_monotonic_time() + 3_000_000
        self._notification_timer = GLib.timeout_add(3000, self._restore_status)

    def _restore_status(self):
        if self.controller.busy and self._status_text == "Updating memories":
            self._notification_deadline = GLib.get_monotonic_time() + 3_000_000
            self._notification_timer = GLib.timeout_add(3000, self._restore_status)
            return False
        self._notification_timer = None
        self.set_status(self._activity_status)
        return False

    def _display_status(self, text):
        if text == self._status_text:
            return
        self._status_text = text
        self._status_dots = 0
        if self._status_timer is not None:
            GLib.source_remove(self._status_timer)
            self._status_timer = None
        self.status.set_text(text)
        if text in (
            "Waiting",
            "Thinking",
            "Responding",
            "Assistant processing documents",
            "Updating memories",
        ):
            self._status_timer = GLib.timeout_add(650, self._animate_status)

    def _animate_status(self):
        self._status_dots = (self._status_dots + 1) % 4
        self.status.set_text(self._status_text + "." * self._status_dots)
        return True

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

    def set_agent_mode(self, enabled):
        self.agent_mode = enabled

    def _prompt_changed(self):
        self.send_button.set_sensitive(self.busy or bool(self.prompt_text()))

    def _send(self):
        if self.busy:
            return
        text = self.prompt_text()
        if text:
            self.prompt.get_buffer().set_text("")
            self.controller.send(text)

    def _key_pressed(self, _controller, key, _code, state):
        if key in (Gdk.KEY_Return, Gdk.KEY_KP_Enter) and state & Gdk.ModifierType.CONTROL_MASK:
            self._send()
            return True
        return False

    def _is_near_bottom(self, adjustment):
        return adjustment.get_upper() - adjustment.get_page_size() - adjustment.get_value() <= 1

    def _prepare_native_scroll(self, _controller, _dx, _dy):
        # GTK ignores adjustment changes during overshoot deceleration. Cancel
        # that animation, then resync its unclamped position with the scrollbar.
        # Propagate the event so GTK supplies the native scrolling and edge glow.
        scroll = self.transcript_scroll
        kinetic = scroll.get_kinetic_scrolling()
        if kinetic:
            scroll.set_kinetic_scrolling(False)
        scroll.get_vadjustment().emit("value-changed")
        if kinetic:
            scroll.set_kinetic_scrolling(True)
        return False

    def _adjustment_value_changed(self, adjustment):
        if self._scroll_guard:
            return
        self.follow_output = self._is_near_bottom(adjustment)

    def _adjustment_changed(self, _adjustment):
        self.scroll_bottom()

    def show_chat(self, chat, folder):
        # Release the sidebar row before refresh removes it and GTK moves focus.
        self.set_focus(self.prompt)
        self.chat = chat
        self.set_chat_title(chat["title"])
        self.folder_label.set_text(f"Chat folder: {folder}")
        self.render_history(chat["messages"])
        if self.busy and chat["id"] == self._response_chat_id:
            self.transcript.append(self._response_row)
            self.scroll_bottom(True)
        stats = chat.get("inference_stats", {})
        self.context_used = stats.get("context_used")
        self.tokens_per_second = None
        self._refresh_inference_stats()

    def show_sidebar(self, chats, active_id):
        self._clear_drop_indicator()
        self._chat_rows = []
        self._dragged_row = None
        while (child := self.chat_list.get_first_child()) is not None:
            self.chat_list.remove(child)
        active = None
        for chat in chats:
            row = Gtk.ListBoxRow()
            row.chat_id = chat["id"]
            line = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=4)
            line.add_css_class("chat-row")
            pinned = chat.get("pinned", False)
            row.pinned = pinned
            if chat["id"] == active_id:
                row.add_css_class("active-chat")
                active = row
            label = Gtk.Label(label=chat.get("title", "New chat"), xalign=0)
            label.set_ellipsize(Pango.EllipsizeMode.END)
            label.set_hexpand(True)
            line.append(label)
            label.set_max_width_chars(28)
            options_button = Gtk.MenuButton(icon_name="view-more-symbolic")
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
                ("Add Documents", lambda chat_id=chat["id"]: self._choose_documents(chat_id)),
                ("Chat Settings", lambda chat_id=chat["id"]: self._show_chat_settings(chat_id)),
                ("Delete Chat", lambda chat_id=chat["id"]: self._confirm_delete_chat(chat_id)),
            )
            for index, (title, callback) in enumerate(choices):
                action = Gio.SimpleAction.new(f"option{index}", None)
                action.connect(
                    "activate", lambda _action, _parameter, callback=callback: callback()
                )
                actions.add_action(action)
                menu.append(title, f"chat.option{index}")
            options_button.insert_action_group("chat", actions)
            popover = Gtk.PopoverMenu.new_from_model(menu)
            popover.add_css_class("chat-options-menu")
            options_button.set_popover(popover)
            line.append(options_button)
            row.set_child(line)
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
            divider.set_margin_top(6)
            divider.set_margin_bottom(6)
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
        while (child := self.transcript.get_first_child()) is not None:
            self.transcript.remove(child)
        for message in messages:
            self.append_message(
                message.get("role"),
                message.get("content", ""),
                message.get("reasoning_content", ""),
            )
        self.scroll_bottom(True)

    def append_message(self, role, text, reasoning=""):
        row = Gtk.Box()
        row.set_layout_manager(BubbleLayout(role == "user"))
        row.add_css_class("message-row")
        row.set_hexpand(True)
        wrapper = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        wrapper.set_valign(Gtk.Align.START)
        wrapper.add_css_class("message")
        wrapper.add_css_class("user-message" if role == "user" else "assistant-message")
        row.append(wrapper)
        label = Gtk.Label(label="You" if role == "user" else "Assistant", xalign=0)
        label.add_css_class("message-role")
        wrapper.append(label)
        if reasoning:
            self._create_thinking_widget(wrapper, reasoning, True)
        if text:
            extractor = (
                self.controller.commands_in if role == "assistant" and self.agent_mode else None
            )
            append_markdown(wrapper, text, extractor, self._review_command)
        self.transcript.append(row)
        if role == "user":
            self.scroll_bottom(force=True)
        return wrapper

    def _create_thinking_widget(self, wrapper, text, completed):
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        box.add_css_class("thinking")
        header_button = Gtk.Button()
        header_button.set_has_frame(False)
        header_button.set_halign(Gtk.Align.START)
        content = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        arrow = Gtk.Label(label="▸" if completed else "▾")
        content.append(arrow)
        content.append(Gtk.Label(label="Thinking", xalign=0))
        header_button.set_child(content)
        body = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=0)
        body.thinking_arrow = arrow
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
            body.tail_label.set_text(remaining[:boundary])
            remaining = remaining[boundary:]
            body.tail_label = None
            body.tail_text = ""

    def _set_thinking_body(self, body, text):
        while (child := body.get_first_child()) is not None:
            body.remove(child)
        body.tail_label = None
        body.tail_text = ""
        self._append_thinking_body(body, text)

    def begin_response(self):
        self.busy = True
        self.send_button.set_sensitive(True)
        self.send_button.set_icon_name("window-close-symbolic")
        self.send_button.set_tooltip_text("Stop generation")
        self.send_button.remove_css_class("suggested-action")
        self.send_button.add_css_class("destructive-action")
        self.set_status("Waiting")
        self.current_wrapper = self.append_message("assistant", "")
        self.current_answer = MarkdownText()
        self.current_content = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        self.current_content.append(self.current_answer)
        self.current_wrapper.append(self.current_content)
        self._answer_committed = 0
        self._stream_code = None
        self._response_agent_mode = self.agent_mode
        self._thinking_collapsed = False
        self.current_thinking = None
        self.scroll_bottom(True)
        self._response_row = self.current_wrapper.get_parent()
        self._response_chat_id = self.chat["id"]

    def update_thinking(self, text):
        if text:
            self.set_status("Thinking")
        if self.current_thinking is None:
            self.current_thinking = self._create_thinking_widget(self.current_wrapper, "", False)
        self._thinking_pending.append(text)
        if self._thinking_flush_id is None:
            self._thinking_flush_id = GLib.timeout_add(100, self._flush_thinking)
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
        end = max((match.end() for match in _CODE_BLOCK_PATTERN.finditer(tail)), default=0)
        if end:
            if self._stream_code is not None:
                self.current_content.remove(self._stream_code[0])
                self._stream_code = None
            self.current_content.remove(self.current_answer)
            extractor = self.controller.commands_in if self._response_agent_mode else None
            append_markdown(self.current_content, tail[:end], extractor, self._review_command)
            self.current_content.append(self.current_answer)
            self._answer_committed += end
            tail = tail[end:]
        opening = re.search(r"```[^\n]*(?:\n(.*))?$", tail, re.DOTALL)
        prose = tail[: opening.start()] if opening else tail
        self.current_answer.set_visible(bool(prose))
        self.current_answer.set_text(prose)
        if opening:
            code = opening[1] or ""
            if self._stream_code is None:
                self._stream_code = append_code_block(self.current_content, code)
            else:
                self._stream_code[1].set_text(code)

    def _collapse_thinking(self):
        if self.current_thinking is None or self._thinking_collapsed:
            return
        self._thinking_collapsed = True
        self.current_thinking.set_visible(False)
        self.current_thinking.thinking_arrow.set_text("▸")

    def finish_response(self, answer, reasoning):
        visible = self._response_row.get_parent() == self.transcript
        if reasoning and self.current_thinking is None:
            self.current_thinking = self._create_thinking_widget(self.current_wrapper, "", True)
        self.current_wrapper.remove(self.current_content)
        append_markdown(
            self.current_wrapper,
            answer.strip(),
            self.controller.commands_in if self._response_agent_mode else None,
            self._review_command,
        )
        if self.current_thinking:
            self._collapse_thinking()
            self._set_thinking_body(self.current_thinking, reasoning.strip())
        if not visible:
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
        self._prompt_changed()
        self.set_status(self.controller.idle_status())
        return False

    def scroll_bottom(self, force=False):
        if force:
            self.follow_output = True
        if self.follow_output and self._scroll_idle_id is None:
            self._scroll_idle_id = GLib.idle_add(self._scroll_to_bottom)

    def _scroll_to_bottom(self):
        self._scroll_idle_id = None
        if not self.follow_output:
            return False
        adjustment = self.transcript_scroll.get_vadjustment()
        self._scroll_guard = True
        adjustment.set_value(adjustment.get_upper() - adjustment.get_page_size())
        self._scroll_guard = False
        return False

    def _show_alert(self, heading, body):
        alert = Adw.AlertDialog(heading=heading, body=body)
        alert.add_response("close", "Close")
        alert.set_default_response("close")
        alert.set_close_response("close")
        alert.present(self)

    def _rename_chat(self, chat_id):
        chat = self.controller.store.load(chat_id)
        dialog = Adw.Dialog(title="Rename Chat", content_width=380)
        toolbar = Adw.ToolbarView()
        header = Adw.HeaderBar()
        toolbar.add_top_bar(header)
        entry = Gtk.Entry(text=chat.get("title", ""))
        entry.set_margin_top(18)
        entry.set_margin_bottom(18)
        entry.set_margin_start(18)
        entry.set_margin_end(18)
        content = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        content.append(entry)
        buttons = Gtk.Box(spacing=8, halign=Gtk.Align.END)
        buttons.set_margin_bottom(18)
        buttons.set_margin_start(18)
        buttons.set_margin_end(18)
        cancel_button = Gtk.Button(label="Cancel")
        cancel_button.connect("clicked", lambda *_: dialog.close())
        buttons.append(cancel_button)
        content.append(buttons)
        toolbar.set_content(content)
        dialog.set_child(toolbar)
        save_button = Gtk.Button(label="Save")
        save_button.add_css_class("suggested-action")

        def save(*_):
            self.controller.rename_chat(chat_id, entry.get_text())
            dialog.close()

        save_button.connect("clicked", save)
        entry.connect("activate", save)
        buttons.append(save_button)
        dialog.present(self)
        entry.grab_focus()

    @staticmethod
    def _settings_dialog(title):
        dialog = Adw.PreferencesDialog(title=title, content_width=660, content_height=540)
        dialog.set_search_enabled(False)
        page = Adw.PreferencesPage()
        dialog.add(page)
        group = Adw.PreferencesGroup()
        page.add(group)
        return dialog, page, group

    def _show_app_settings(self):
        self._show_settings()

    def _show_chat_settings(self, chat_id):
        self._show_settings(chat_id)

    def _show_settings(self, chat_id=None):
        dialog, page, group = self._settings_dialog(
            "App Settings" if chat_id is None else "Chat Settings"
        )
        settings = self.controller.get_chat_settings(chat_id)
        group.set_title("Default Chat Settings" if chat_id is None else "Chat Settings")
        if chat_id is not None:
            page.set_description(
                "Changes override app defaults for this chat. 'X' denotes an override."
            )
        overrides = self.controller.get_chat_overrides(chat_id) if chat_id is not None else set()
        markers = []
        setting_controls = {}

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
                box = Gtk.Box(spacing=8)
                marker = Gtk.Button(tooltip_text="Delete override", valign=Gtk.Align.CENTER)
                marker.set_child(Gtk.Image(icon_name="window-close-symbolic", pixel_size=24))
                marker.add_css_class("flat")
                marker.connect("clicked", delete_override, key)
                box.append(marker)
                box.append(button)
                row.add_suffix(box)
                markers.append((marker, key, button))
                update_markers()
            row.set_activatable_widget(button)

        def add_switch(key, title, tooltip):
            row = Adw.ActionRow(title=title)
            button = Gtk.Switch(active=settings[key] is True, valign=Gtk.Align.CENTER)
            add_control(row, button, key)
            row.set_tooltip_text(tooltip)

            def changed(row, _param):
                try:
                    self.controller.set_setting(key, row.get_active(), chat_id)
                    settings[key] = row.get_active()
                    overrides.add(key)
                    update_markers()
                except OSError as exc:
                    with row.handler_block(handler):
                        row.set_active(settings[key] is True)
                    self._show_alert("Could not save settings", str(exc))

            handler = button.connect("notify::active", changed)
            setting_controls[key] = [(button, handler, None)]
            group.add(row)

        add_switch(
            "global_memories",
            "Global Memories",
            "Recall and learn personal memories across chats.",
        )
        if chat_id is None:
            memories_row = Adw.ActionRow(title="Manage Memories", activatable=True)
            memories_row.add_suffix(Gtk.Image(icon_name="go-next-symbolic"))
            memories_row.connect("activated", lambda *_: self._show_memories(dialog))
            group.add(memories_row)
        add_switch(
            "preserve_reasoning",
            "Preserve Reasoning",
            "Allow model to remember previous thinking blocks.",
        )
        add_switch(
            "rag_query_expansion",
            "Expand Document and Memory Searches",
            "Ask the current model for related search terms before retrieval. "
            "Adds slight prompt latency.",
        )
        add_switch(
            "rag_model_indexing",
            "Model-Assisted Document Indexing (SLOW)",
            "Modestly improves document info retrieval. CAN BE SLOW, best used with t/s > 80.",
        )

        def add_choices(title, options, key, setter):
            choice_group = Adw.PreferencesGroup(title=title)
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
                if radio_group is None:
                    radio_group = button
                else:
                    button.set_group(radio_group)
                button.set_active(value == settings[key])
                handler = button.connect("toggled", choice_changed, value)
                controls[value] = button, handler
                row = Adw.ActionRow(title=label)
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
        dialog.present(self)

    def _show_memories(self, settings_dialog):
        dialog = Adw.Dialog(title="Global Memories", content_width=660, content_height=540)
        toolbar = Adw.ToolbarView()
        header = Adw.HeaderBar()
        back = Gtk.Button(icon_name="go-previous-symbolic", tooltip_text="Back to App Settings")
        back.add_css_class("flat")
        header.pack_start(back)
        toolbar.add_top_bar(header)
        page = Adw.PreferencesPage()
        group = Adw.PreferencesGroup()
        page.add(group)
        toolbar.set_content(page)
        dialog.set_child(toolbar)
        dialog.return_to_settings = False

        def go_back(*_):
            dialog.return_to_settings = True
            dialog.close()

        back.connect("clicked", go_back)
        dialog.connect(
            "closed", lambda *_: None if dialog.return_to_settings else settings_dialog.close()
        )
        group.set_description(
            "Edit facts, select Core for always-present memories, or delete them. "
            "Click the pencil to edit, then the checkmark to save."
        )
        for memory_id, text, core in self.controller.memories.records():
            row = Adw.ActionRow(title=text)
            row.set_title_lines(0)
            entry = Gtk.Entry(text=text, hexpand=True, visible=False)
            row.add_prefix(entry)
            controls = Gtk.Box(spacing=6)
            core_button = Gtk.CheckButton(label="Core", active=bool(core))
            controls.append(core_button)
            edit = Gtk.Button(icon_name="document-edit-symbolic", tooltip_text="Edit memory")
            edit.add_css_class("flat")
            edit.set_valign(Gtk.Align.CENTER)
            delete = Gtk.Button(icon_name="window-close-symbolic", tooltip_text="Delete memory")
            delete.set_child(Gtk.Image(icon_name="window-close-symbolic", pixel_size=24))
            delete.add_css_class("flat")
            delete.set_valign(Gtk.Align.CENTER)

            def change(
                _button,
                *,
                memory_id=memory_id,
                entry=entry,
                core_button=core_button,
                row=row,
                edit=edit,
                delete=False,
            ):
                if not delete and not entry.get_visible():
                    row.set_title("")
                    row.set_subtitle("")
                    entry.set_visible(True)
                    edit.set_icon_name("object-select-symbolic")
                    edit.set_tooltip_text("Save memory")
                    entry.grab_focus()
                    return
                text = entry.get_text().strip()
                try:
                    self.controller.memories.apply(
                        [
                            {
                                "action": "delete" if delete else "update",
                                "id": memory_id,
                                "text": text,
                                "kind": "core" if core_button.get_active() else "casual",
                                "evidence": text or "Delete memory",
                            }
                        ],
                        text or "Delete memory",
                        "manual",
                        {memory_id},
                    )
                    if delete:
                        group.remove(row)
                    else:
                        saved = next(
                            (
                                record
                                for record in self.controller.memories.records()
                                if record[0] == memory_id
                            ),
                            None,
                        )
                        if saved and saved[1] == text:
                            with core_button.handler_block(core_button.memory_handler):
                                core_button.set_active(bool(saved[2]))
                            entry.set_visible(False)
                            row.set_title(text)
                            edit.set_icon_name("document-edit-symbolic")
                            edit.set_tooltip_text("Edit memory")
                        else:
                            self._show_alert(
                                "Memory was not saved", "Check for duplicate or empty text."
                            )
                except Exception as exc:
                    self._show_alert("Could not change memory", str(exc))

            def core_changed(button, memory_id=memory_id):
                try:
                    self.controller.memories.set_core(memory_id, button.get_active())
                except Exception as exc:
                    with button.handler_block(button.memory_handler):
                        button.set_active(not button.get_active())
                    self._show_alert("Could not change core memory", str(exc))

            core_button.memory_handler = core_button.connect("toggled", core_changed)
            edit.connect("clicked", change)
            delete.connect("clicked", lambda button, change=change: change(button, delete=True))
            controls.append(delete)
            row.add_suffix(edit)
            row.add_suffix(controls)
            group.add(row)
        dialog.present(self)

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

    def _choose_documents(self, chat_id):
        if self.chat and self.chat["id"] != chat_id:
            self.controller.open_chat(chat_id)
        chooser = Gtk.FileDialog.new()
        chooser.set_title("Add documents to this chat")
        chooser.open_multiple(self, None, self._documents_chosen)

    def _documents_chosen(self, chooser, result):
        try:
            files = chooser.open_multiple_finish(result)
        except GLib.Error:
            return
        paths = (files.get_item(i).get_path() for i in range(files.get_n_items()))
        self.controller.add_documents([path for path in paths if path])

    def show_document_warning(self, paths):
        detail = (
            "Only readable text documents are currently supported. "
            "These files are unsupported or could not be read and were not copied:\n\n"
            + "\n".join(str(path) for path in paths)
        )
        self._show_alert("Some documents could not be added", detail)

    def _review_command(self, _button, command):
        kind, payload, risks = self.controller.review_command(command)
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
        toolbar.add_top_bar(Adw.HeaderBar())
        dialog.set_content(toolbar)

        def close_preview(*_):
            self.set_sensitive(True)
            dialog.destroy()

        def close_requested(*_):
            self.set_sensitive(True)
            return False

        dialog.connect("close-request", close_requested)
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        box.set_margin_top(18)
        box.set_margin_bottom(18)
        box.set_margin_start(18)
        box.set_margin_end(18)
        toolbar.set_content(box)
        for text in (
            description,
            f"Working directory: {self.controller.store.folder(self.chat['id'])}",
        ):
            label = Gtk.Label(label=text, xalign=0)
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
            legend.set_wrap(True)
            box.append(legend)
        lines = command.count("\n") + 1
        if lines > 16:
            notice = Gtk.Label(
                label=f"↕ {lines} lines, scroll inside the preview to review the full command.",
                xalign=0,
            )
            notice.set_wrap(True)
            notice.add_css_class("heading")
            box.append(notice)
        preview = Gtk.TextView(editable=False, cursor_visible=False, monospace=True)
        preview.set_left_margin(4)
        preview.set_right_margin(4)
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
        frame.add_css_class("command-preview")
        frame.set_vexpand(True)
        frame.set_child(scroll)
        box.append(frame)
        buttons = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        buttons.set_halign(Gtk.Align.END)
        cancel_button = Gtk.Button(label="Cancel")
        cancel_button.connect("clicked", close_preview)
        buttons.append(cancel_button)
        approve_button = Gtk.Button(label=action_label)
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
        self.set_status(self.controller.idle_status())
        self.scroll_bottom()
        return False

    def show_command_error(self, error):
        self.set_status(self.controller.idle_status())
        self.notify(f"Command failed: {error}")
        self.append_message("assistant", f"Command could not be started: {error}")
        self.scroll_bottom()
        return False


class Lopaca(Adw.Application):
    def __init__(self, controller):
        super().__init__(application_id="app.lopaca.Lopaca")
        self.controller = controller
        GLib.set_application_name("Lopaca")

    def do_activate(self):
        window = self.props.active_window
        if not window:
            window = ChatWindow(self, self.controller)
        window.present()
