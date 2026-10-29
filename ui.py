"""GTK/libadwaita presentation layer. Application behavior lives in main.py."""

import html
import re

import gi

from settings import CHAT_SORT_OPTIONS

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


def append_markdown(parent, text, command_extractor=None, command_handler=None):
    """Append safe inline Markdown and fenced code blocks."""

    def add_label(value):
        label = Gtk.Label(xalign=0, yalign=0)
        label.set_wrap(True)
        label.set_wrap_mode(Pango.WrapMode.WORD_CHAR)
        label.set_selectable(True)
        value = re.sub(r"^### (.*)$", r"<b>\1</b>", html.escape(value), flags=re.MULTILINE)
        value = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", value)
        value = re.sub(r"`([^`]+)`", r"<tt>\1</tt>", value)
        label.set_markup(value)
        parent.append(label)

    last = 0
    for match in _CODE_BLOCK_PATTERN.finditer(text):
        if text[last : match.start()]:
            add_label(text[last : match.start()])
        commands = command_extractor(match.group(0)) if command_extractor else []
        if commands:
            for command in commands:
                button = Gtk.Button(label="Review Command")
                button.set_halign(Gtk.Align.START)
                button.connect("clicked", command_handler, command)
                parent.append(button)
        else:
            frame = Gtk.Frame()
            frame.add_css_class("code-block-frame")
            code_label = Gtk.Label(label=match.group(1).rstrip("\n"), xalign=0, yalign=0)
            code_label.set_wrap(True)
            code_label.set_wrap_mode(Pango.WrapMode.WORD_CHAR)
            code_label.set_selectable(True)
            code_label.add_css_class("code-block")
            frame.set_child(code_label)
            parent.append(frame)
        last = match.end()
    if text[last:] or not last:
        add_label(text[last:])


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
        self._notification_timer = None
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
        self.chat_scroll.set_margin_bottom(8)
        sidebar.append(self.chat_scroll)
        footer = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        footer.set_margin_start(10)
        footer.set_margin_end(10)
        footer.set_margin_bottom(12)
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
        actions.append(self.send_button)
        inner.append(actions)
        main.append(compose)
        self.send_button.add_css_class("send-stop")
        self._prompt_changed()

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
        if self._notification_timer is None:
            self._display_status(text)

    def notify(self, text):
        if self._notification_timer is not None:
            GLib.source_remove(self._notification_timer)
        self._display_status(text)
        self._notification_timer = GLib.timeout_add(3000, self._restore_status)

    def _restore_status(self):
        self._notification_timer = None
        self._display_status(self._activity_status)
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
        if text in ("Waiting", "Thinking", "Responding"):
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
        limit = max(adjustment.get_lower(), adjustment.get_upper() - adjustment.get_page_size())
        adjustment.set_value(max(adjustment.get_lower(), min(limit, row.get_allocation().y)))
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
        self.current_answer = Gtk.Label(label="", xalign=0, yalign=0)
        self.current_answer.set_wrap(True)
        self.current_answer.set_wrap_mode(Pango.WrapMode.WORD_CHAR)
        self.current_answer.set_selectable(True)
        self.current_content = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        self.current_content.append(self.current_answer)
        self.current_wrapper.append(self.current_content)
        self._answer_committed = 0
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
            self.current_content.remove(self.current_answer)
            extractor = self.controller.commands_in if self._response_agent_mode else None
            append_markdown(self.current_content, tail[:end], extractor, self._review_command)
            self.current_content.append(self.current_answer)
            self._answer_committed += end
            tail = tail[end:]
        self.current_answer.set_text(tail)

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
            self.controller.commands_in if self.agent_mode else None,
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

    def _show_alert(self, heading, body, parent=None):
        alert = Adw.AlertDialog(heading=heading, body=body)
        alert.add_response("close", "Close")
        alert.set_default_response("close")
        alert.set_close_response("close")
        alert.present(parent if parent is not None else self)

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
        dialog, page, group = self._settings_dialog("App Settings")
        preserve_reasoning_checkbox = Adw.SwitchRow(
            title="Preserve Reasoning", active=self.controller.preserve_thinking
        )
        preserve_reasoning_checkbox.set_tooltip_text(
            "Allow model to remember previous thinking blocks."
        )
        group.add(preserve_reasoning_checkbox)

        def changed(button, _param):
            try:
                self.controller.set_preserve_reasoning(button.get_active())
            except OSError as exc:
                with button.handler_block(preserve_reasoning_handler):
                    button.set_active(not button.get_active())
                self._show_alert("Could not save app settings", str(exc), self)

        preserve_reasoning_handler = preserve_reasoning_checkbox.connect("notify::active", changed)
        expansion_row = Adw.SwitchRow(
            title="Expand Document Searches", active=self.controller.rag_query_expansion
        )
        expansion_row.set_tooltip_text(
            "Ask the current model for related search terms before retrieval. Adds latency."
        )

        def expansion_changed(row, _param):
            try:
                self.controller.set_rag_query_expansion(row.get_active())
            except OSError as exc:
                with row.handler_block(expansion_handler):
                    row.set_active(not row.get_active())
                self._show_alert("Could not save app settings", str(exc), self)

        expansion_handler = expansion_row.connect("notify::active", expansion_changed)
        group.add(expansion_row)
        sort_group = Adw.PreferencesGroup(title="Unpinned Chat Sort")
        page.add(sort_group)
        sort_controls = {}

        def sort_changed(button, sort_mode):
            if not button.get_active():
                return
            try:
                self.controller.set_chat_sort(sort_mode)
            except OSError as exc:
                previous_button, previous_handler = sort_controls[self.controller.chat_sort]
                with previous_button.handler_block(previous_handler):
                    previous_button.set_active(True)
                self._show_alert("Could not save app settings", str(exc), self)

        group = None
        for sort_mode, label in CHAT_SORT_OPTIONS:
            button = Gtk.CheckButton(valign=Gtk.Align.CENTER)
            if group is None:
                group = button
            else:
                button.set_group(group)
            button.set_active(sort_mode == self.controller.chat_sort)
            sort_handler = button.connect("toggled", sort_changed, sort_mode)
            sort_controls[sort_mode] = button, sort_handler
            row = Adw.ActionRow(title=label)
            row.add_suffix(button)
            row.set_activatable_widget(button)
            sort_group.add(row)
        dialog.present(self)

    def _show_chat_settings(self, chat_id):
        chat_settings = self.controller.get_chat_settings(chat_id)
        dialog, _page, group = self._settings_dialog("Chat Settings")
        code_agent_checkbox = Adw.SwitchRow(title="Code Agent", active=chat_settings["code_agent"])
        code_agent_checkbox.set_tooltip_text(
            "Let the assistant suggest terminal commands for explicit approval"
        )
        code_agent_checkbox.set_sensitive(self.controller.settings.enable_command_approval)

        def changed(button, _param):
            try:
                self.controller.set_agent_mode(chat_id, button.get_active())
            except OSError as exc:
                with button.handler_block(handler):
                    button.set_active(not button.get_active())
                self._show_alert("Could not save chat settings", str(exc), self)

        handler = code_agent_checkbox.connect("notify::active", changed)
        group.add(code_agent_checkbox)
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
        self.notify(f"Command failed: {error}")
        self.append_message("assistant", f"Command could not be started: {error}")
        self.scroll_bottom()
        return False


class Lopaca(Adw.Application):
    def __init__(self, controller):
        super().__init__(application_id="io.github.lopaca.Client")
        self.controller = controller
        GLib.set_application_name("Lopaca")

    def do_activate(self):
        window = self.props.active_window
        if not window:
            window = ChatWindow(self, self.controller)
        window.present()
