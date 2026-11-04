"""GTK Markdown parsing, selectable rendering, code blocks and scrolling frames."""

import html
import re

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Gsk", "4.0")
gi.require_version("Graphene", "1.0")
from gi.repository import Graphene, Gtk, Pango  # noqa: E402

CODE_BLOCK_PATTERN = re.compile(
    r"^ {0,3}(`{3,}|~{3,})([^\n]*)\n(.*?)^ {0,3}\1[ \t]*(?=\n|$)",
    re.DOTALL | re.MULTILINE,
)


_INLINE = re.compile(
    r"\\([!\"#$%&'()*+,\-./:;<=>?@\[\]\\^_`{|}~])"
    r"|(`+)(.+?)\2"
    r'|!?\[([^\]]+)\]\(([^\s]+?)(?:\s+"[^"\n]*")?\)'
    r"|<(https?://[^<>\s]+|mailto:[^<>\s]+|[^<>\s@]+@[^<>\s@]+\.[^<>\s@]+)>"
    r"|((?:https?://|www\.)[^\s<>]+|[\w.+-]+@[\w-]+(?:\.[\w-]+)+)"
    r"|(\*{1,3}|(?<!\w)_{1,3}|~~|==|~|\^)(?=\S)(.+?)(?<=\S)\8(?:(?![\w*_])|(?<=~)|(?<=\^))"
    r"|\[\^([^\]\n]+)\]"
    r"|<(sub|sup|ins)>(.*?)</\11>",
    re.DOTALL,
)


def inline_markdown(text, depth=0):
    """Escape model text and allow only explicitly generated Pango markup."""
    if depth > 12:
        return html.escape(text)
    parts, start = [], 0
    for match in _INLINE.finditer(text):
        parts.append(html.escape(html.unescape(text[start : match.start()])))
        escaped, ticks, code, title, url, autolink, bare, emphasis, body = match.groups()[:9]
        if match[0].startswith("!"):
            value = html.escape(match[0])
        elif escaped:
            value = html.escape(escaped)
        elif ticks:
            value = "<tt>" + html.escape(code.replace("\n", " ")) + "</tt>"
        elif title or autolink or bare:
            target = url or autolink or bare
            caption_target = target
            if "@" in target and ":" not in target:
                target = "mailto:" + target
            elif target.startswith("www."):
                target = "https://" + target
            caption = inline_markdown(title, depth + 1) if title else html.escape(caption_target)
            tooltip = re.search(r'\s+"([^"\n]*)"\)$', match[0])
            hover = f"{tooltip[1]} — {target}" if tooltip else target
            attributes = f' title="{html.escape(hover, quote=True)}"'
            value = (
                f'<a href="{html.escape(target, quote=True)}"{attributes}>{caption}</a>'
                if re.match(r"^(?:https?://|mailto:|file:///|#)", target, re.IGNORECASE)
                else caption
            )
        elif match[10]:
            number = html.escape(match[10])
            value = (
                f'<a href="#fn-{number}" title="Go to footnote {number}"><sup>{number}</sup></a>'
            )
        elif match[11]:
            tag = "u" if match[11] == "ins" else match[11]
            value = f"<{tag}>" + inline_markdown(match[12], depth + 1) + f"</{tag}>"
        else:
            tags = {
                "*": ("i",),
                "_": ("i",),
                "**": ("b",),
                "__": ("b",),
                "***": ("b", "i"),
                "___": ("b", "i"),
                "~~": ("s",),
                "~": ("sub",),
                "^": ("sup",),
                "==": (),
            }[emphasis]
            value = "".join(f"<{tag}>" for tag in tags) + inline_markdown(body, depth + 1)
            value += "".join(f"</{tag}>" for tag in reversed(tags))
            if emphasis == "==":
                value = f'<span background="#e5c34b" foreground="#241f31">{value}</span>'
        parts.append(value)
        start = match.end()
    parts.append(html.escape(html.unescape(text[start:])))
    return "".join(parts)


def set_markdown_text(label, text):
    label.set_markup(inline_markdown(text))


def reference_links(text):
    """Resolve reference-style links while leaving inline code untouched."""
    references = {}

    def definition(match):
        references[match[1].casefold()] = match[2].strip("<>")
        return ""

    text = re.sub(
        r'(?m)^ {0,3}\[(?!\^)([^\]\n]+)\]:[ \t]*(\S+)(?:[ \t]+"[^"\n]*")?[ \t]*$',
        definition,
        text,
    )
    if not references:
        return text

    def link(match):
        if match[1] or match[0].startswith("!"):
            return match[0]
        target = references.get((match[4] or match[3]).casefold())
        return f"[{match[3]}]({target})" if target else match[0]

    return re.sub(r"(`+)(.*?)\1|(?<!\\)!?\[([^\]\n]+)\](?:\[([^\]\n]*)\])?(?!\()", link, text)


def markdown_footnotes(text):
    definitions, order = {}, []
    fences = [(match.start(), match.end()) for match in CODE_BLOCK_PATTERN.finditer(text)]

    def definition(match):
        if any(start <= match.start() < end for start, end in fences):
            return match[0]
        definitions[match[1]] = re.sub(r"\n[ \t]{4}", "\n", match[2]).strip()
        return ""

    text = re.sub(r"(?m)^ {0,3}\[\^([^\]\n]+)\]:[ \t]*(.*(?:\n[ \t]{4}.*)*)$", definition, text)
    fences = [(match.start(), match.end()) for match in CODE_BLOCK_PATTERN.finditer(text)]

    def reference(match):
        key = match[3]
        if key not in definitions or any(start <= match.start() < end for start, end in fences):
            return match[0]
        if key not in order:
            order.append(key)
        return f"[^{order.index(key) + 1}]"

    text = re.sub(r"(`+)(.*?)\1|\[\^([^\]\n]+)\]", reference, text)
    return text, [("footnote", (index, definitions[key])) for index, key in enumerate(order, 1)]


def prose_blocks(text):
    """Recognize headings, quotations, rules and indented list items."""
    lines = text.splitlines(keepends=True)
    index, paragraph = 0, []
    while index < len(lines):
        line = lines[index].rstrip("\n")
        heading = re.match(r"^ {0,3}(#{1,6})\s+(.+?)(?:\s+#+)?\s*$", line)
        item = re.match(r"^(\s*)([-+*]|\d+[.)])\s+(.*)$", line)
        quote = re.match(r"^ {0,3}>\s?(.*)$", line)
        rule = re.fullmatch(r" {0,3}(?:(?:\*\s*){3,}|(?:-\s*){3,}|(?:_\s*){3,})", line)
        underline = index + 1 < len(lines) and re.fullmatch(r" {0,3}(=+|-+)\s*", lines[index + 1])
        kind, value = None, None
        if heading:
            kind, value = "heading", (len(heading[1]), heading[2])
        elif underline and line.strip():
            kind, value = "heading", (1 if "=" in lines[index + 1] else 2, line)
            index += 1
        elif rule:
            kind, value = "rule", ""
        elif quote:
            content = [quote[1]]
            while index + 1 < len(lines) and (
                following := re.match(r"^ {0,3}>\s?(.*)", lines[index + 1])
            ):
                index += 1
                content.append(following[1])
            kind, value = "quote", "\n".join(content)
        elif item:
            indent, marker, content = item.groups()
            task = re.match(r"\[([ xX])\]\s*(.*)", content)
            marker = (
                ("☑" if task[1].lower() == "x" else "☐")
                if task
                else ("•" if marker in "-+*" else marker)
            )
            content = task[2] if task else content
            while (
                index + 1 < len(lines)
                and lines[index + 1].strip()
                and (
                    len(lines[index + 1]) - len(lines[index + 1].lstrip()) > len(indent)
                    and not re.match(r"^\s*(?:[-+*]|\d+[.)])\s", lines[index + 1])
                )
            ):
                index += 1
                content += "\n" + lines[index].strip()
            kind, value = "item", (len(indent.expandtabs(4)), marker, content)
        elif line.startswith(("    ", "\t")) and (not paragraph or not paragraph[-1].strip()):
            code = [line[4:] if line.startswith("    ") else line[1:]]
            while index + 1 < len(lines) and lines[index + 1].startswith(("    ", "\t")):
                index += 1
                following = lines[index].rstrip("\n")
                code.append(following[4:] if following.startswith("    ") else following[1:])
            kind, value = "code", "\n".join(code)
        if kind:
            if paragraph:
                yield "prose", "".join(paragraph).strip("\n")
                paragraph.clear()
            yield kind, value
        else:
            paragraph.append(lines[index])
        index += 1
    if paragraph:
        yield "prose", "".join(paragraph).strip("\n")


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
    label.connect("activate-link", activate_markdown_link)
    set_markdown_text(label, text)
    return label


def activate_markdown_link(label, uri):
    if not uri.startswith("#"):
        return False
    parent = label.get_parent()
    message = None
    while parent is not None:
        if parent.has_css_class("message"):
            message = parent
            break
        if isinstance(parent, MarkdownText):
            target = parent.anchors.get(uri[1:], parent if uri == "#" else None)
            if target is not None:
                target, offset = target if isinstance(target, tuple) else (target, 0)
                outer = target.get_ancestor(Gtk.ScrolledWindow)
                while outer is not None and outer.get_parent() is not None:
                    ancestor = outer.get_parent().get_ancestor(Gtk.ScrolledWindow)
                    if ancestor is None:
                        break
                    outer = ancestor
                if outer is not None:
                    content = outer.get_child()
                    if isinstance(content, Gtk.Viewport):
                        content = content.get_child()
                    valid, point = target.compute_point(content, Graphene.Point().init(0, 0))
                    if valid:
                        line = (
                            target.get_layout().index_to_pos(offset).y / Pango.SCALE
                            if offset
                            else 0
                        )
                        outer.get_vadjustment().set_value(point.y + content.get_margin_top() + line)
                return True
        parent = parent.get_parent()
    if message is not None:
        pending = [message]
        while pending:
            widget = pending.pop()
            if isinstance(widget, MarkdownText) and uri[1:] in widget.anchors:
                # Reuse the same navigation path when the target is across a code fence.
                target = widget.anchors[uri[1:]]
                return activate_markdown_link(
                    target[0] if isinstance(target, tuple) else target, uri
                )
            child = widget.get_first_child()
            while child is not None:
                pending.append(child)
                child = child.get_next_sibling()
    return True


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
        self._line_widths = {}
        self._width_signature = None
        self._content_line_width = None
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
        if isinstance(child, CodeLabel):
            context = child.get_pango_context()
            signature = (context.get_serial(), context.get_font_description().to_string())
            previous = self._line_widths if signature == self._width_signature else {}
            widths = {}
            for line in set(child.get_text().split("\n")):
                widths[line] = (
                    previous[line]
                    if line in previous
                    else child.create_pango_layout(line).get_pixel_size()[0]
                )
            widest = max(widths.values(), default=0)
            unchanged = signature == self._width_signature and widest == self._content_line_width
            self._line_widths, self._width_signature = widths, signature
            self._content_line_width = widest
            if unchanged:
                return
        # GTK still determines the exact width/cap whenever the widest line changes.
        child.set_size_request(-1, -1)
        child.set_size_request(child.measure(Gtk.Orientation.HORIZONTAL, -1)[1], -1)

    def _connect_scrollbar(self, _widget):
        self._outer_scroll = self.get_ancestor(Gtk.ScrolledWindow)
        while self._outer_scroll is not None:
            parent = self._outer_scroll.get_parent()
            outer = parent.get_ancestor(Gtk.ScrolledWindow) if parent else None
            if outer is None:
                break
            self._outer_scroll = outer
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

    def __init__(self, text="", boxed=False):
        super().__init__(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        self.blocks = []
        self.boxed = boxed
        self.set_text(text)

    def set_text(self, text):
        self.anchors = {}
        text, notes = markdown_footnotes(text)
        text = reference_links(text)
        blocks = [
            block
            for kind, value in markdown_blocks(text)
            for block in (prose_blocks(value) if kind == "prose" else [(kind, value)])
        ]
        blocks.extend(notes)
        grouped = []
        for kind, value in blocks:
            if kind in ("prose", "heading", "item", "footnote"):
                if not grouped or grouped[-1][0] != "text":
                    grouped.append(("text", []))
                grouped[-1][1].append((kind, value))
            else:
                grouped.append((kind, value))
        blocks = grouped
        for index, (kind, value) in enumerate(blocks):
            if index >= len(self.blocks) or self.blocks[index][0] != kind:
                while len(self.blocks) > index:
                    self.remove(self.blocks.pop()[1])
                child = render_markdown_block(kind, value)
                self.append(child)
            else:
                _, child, previous = self.blocks[index]
                if value != previous:
                    render_markdown_block(kind, value, child)
            block = (kind, child, value)
            if index == len(self.blocks):
                self.blocks.append(block)
            else:
                self.blocks[index] = block
            if self.boxed and isinstance(child, Gtk.Label):
                child.set_max_width_chars(500)
                child.set_size_request(-1, -1)
                child.set_size_request(child.measure(Gtk.Orientation.HORIZONTAL, -1)[1], -1)
            if kind == "text":
                for name, offset in child.markdown_anchors:
                    anchor, suffix = name, 1
                    while anchor in self.anchors:
                        anchor, suffix = f"{name}-{suffix}", suffix + 1
                    self.anchors[anchor] = (child, offset)
        while len(self.blocks) > len(blocks):
            self.remove(self.blocks.pop()[1])


def selectable_markdown(label, blocks):
    """Use one label for adjacent headings, paragraphs and lists so selection spans them."""
    parts, plain, previous = [], "", None
    label.markdown_anchors = []
    for kind, value in blocks:
        separator = "\n" if kind == previous == "item" else "\n\n"
        if parts:
            parts.append(separator)
            plain += separator
        if kind == "heading":
            level, text = value
            size = ("xx-large", "x-large", "large", "medium", "medium", "small")[level - 1]
            markup = f'<span size="{size}" weight="bold">{inline_markdown(text)}</span>'
            anchor = re.sub(r"[^\w\s-]", "", text.casefold()).replace(" ", "-")
            label.markdown_anchors.append((anchor, len(plain.encode("utf-8"))))
        elif kind in ("item", "footnote"):
            indent, marker, text = value if kind == "item" else (0, f"{value[0]}.", value[1])
            markup = (
                "\u2003" * min(indent // 2, 20) + html.escape(marker) + " " + inline_markdown(text)
            )
            if kind == "footnote":
                label.markdown_anchors.append((f"fn-{value[0]}", len(plain.encode("utf-8"))))
        else:
            markup = inline_markdown(value)
        parts.append(markup)
        plain += html.unescape(re.sub(r"<[^>]*>", "", markup))
        previous = kind
    label.set_markup("".join(parts))


def render_markdown_block(kind, value, widget=None):
    if kind == "table":
        if widget is None:
            return MarkdownTable(value)
        widget.set_data(value)
    elif kind == "rule":
        return (
            widget if widget is not None else Gtk.Separator(orientation=Gtk.Orientation.HORIZONTAL)
        )
    elif kind == "code":
        if widget is None:
            widget = append_code_block(None, value)[0]
        else:
            widget.scroll.get_child().get_child().set_text(value)
    elif kind == "quote":
        if widget is None:
            widget = HorizontalScrollFrame(MarkdownText(boxed=True))
            widget.add_css_class("markdown-quote")
        widget.scroll.get_child().get_child().set_text(value)
        widget.update_content_width()
    else:
        if widget is None:
            widget = markdown_label()
        selectable_markdown(widget, value if kind == "text" else [(kind, value)])
    return widget


_HIGHLIGHT = re.compile(
    r"(?P<comment>\#[^\n]*|//[^\n]*|/\*.*?\*/)"
    r"""|(?P<string>"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*')"""
    r"|(?P<number>\b\d+(?:\.\d+)?\b)"
    r"|(?P<keyword>\b(?:def|class|return|import|from|as|if|else|elif|for|while|try|except|"
    r"finally|with|yield|lambda|async|await|in|is|not|and|or|True|False|None|function|const|"
    r"let|var|new|throw|public|private|static|void|int|bool|struct|enum|fn|func|package|"
    r"use|pub|match|case|switch|break|continue|true|false|null|nil|SELECT|FROM|WHERE)\b)",
    re.DOTALL,
)


class CodeLabel(Gtk.Label):
    """Keep source text selectable while applying lightweight syntax colors."""

    def __init__(self, text, language):
        super().__init__(xalign=0, yalign=0)
        self.language = language
        self.set_text(text)

    def set_text(self, text):
        if self.language.casefold() not in {
            "python",
            "py",
            "javascript",
            "js",
            "typescript",
            "ts",
            "json",
            "c",
            "cpp",
            "c++",
            "java",
            "rust",
            "rs",
            "go",
            "bash",
            "sh",
            "shell",
            "sql",
            "css",
        }:
            Gtk.Label.set_text(self, text)
            return
        parts, start = [], 0
        colors = {
            "comment": "#77767b",
            "string": "#26a269",
            "number": "#c061cb",
            "keyword": "#3584e4",
        }
        for match in _HIGHLIGHT.finditer(text):
            parts.append(html.escape(text[start : match.start()]))
            parts.append(
                f'<span foreground="{colors[match.lastgroup]}">{html.escape(match[0])}</span>'
            )
            start = match.end()
        parts.append(html.escape(text[start:]))
        self.set_markup("".join(parts))


def append_code_block(parent, text, language=""):
    label = CodeLabel(text, language)
    label.set_wrap(True)
    label.set_wrap_mode(Pango.WrapMode.WORD_CHAR)
    label.set_max_width_chars(500)
    label.set_selectable(True)
    label.add_css_class("code-block")
    frame = HorizontalScrollFrame(label)
    frame.add_css_class("code-block-frame")
    if parent is not None:
        parent.append(frame)
    return frame, label


def append_markdown(parent, text, command_extractor=None, command_handler=None):
    """Append safe Markdown, tables, and fenced code or command blocks."""

    text, notes = markdown_footnotes(text)
    last = 0
    for match in CODE_BLOCK_PATTERN.finditer(text):
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
            append_code_block(parent, match.group(3).rstrip("\n"), match.group(2).strip())
        last = match.end()
    if text[last:] or not last:
        parent.append(MarkdownText(text[last:]))
    if notes:
        footer = MarkdownText()
        for _kind, value in notes:
            child = render_markdown_block("footnote", value)
            footer.append(child)
            footer.anchors[f"fn-{value[0]}"] = child
        parent.append(footer)
