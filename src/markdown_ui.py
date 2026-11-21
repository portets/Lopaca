"""GTK Markdown parsing, selectable rendering, code blocks and scrolling frames."""

import html
import re
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from urllib.parse import unquote

import gi

from images import decode_image, fit_image_size, image_mime_type
from web import fetch_web_image, is_web_image_url

gi.require_version("Gtk", "4.0")
gi.require_version("Gdk", "4.0")
gi.require_version("Gsk", "4.0")
gi.require_version("Graphene", "1.0")
gi.require_version("GdkPixbuf", "2.0")
from gi.repository import Gdk, GdkPixbuf, GLib, Graphene, Gtk, Pango  # noqa: E402

_image_loader = ThreadPoolExecutor(max_workers=2, thread_name_prefix="chat-image")
_web_image_loader = ThreadPoolExecutor(max_workers=2, thread_name_prefix="web-image")
_REPLY_IMAGE_SIZE = (256, 512)
_EXPANDED_IMAGE_SIZE = (512, 1024)

CODE_BLOCK_PATTERN = re.compile(
    r"^ {0,3}(`{3,}|~{3,})([^\n]*)\n(.*?)^ {0,3}\1[ \t]*(?=\n|$)",
    re.DOTALL | re.MULTILINE,
)


_IMAGE_SPAN = re.compile(
    r"\\[^\n]|(?P<ticks>`+)[^\n]+?(?P=ticks)"
    r"|!\[(?P<alt>(?:\\.|[^\]\\\n])*)\]\([ \t]*"
    r"(?P<source><[^<>\n]+>|(?:[^\s()<>]|\([^\s()<>]*\))+)"
    r'(?:[ \t]+"[^"\n]*")?[ \t]*\)'
    r"|(?P<unfinished>!\[(?:\\.|[^\]\\\n])*(?:\](?:\([^\n]*)?)?)\Z"
)


def image_spans(text, *, streaming=False):
    """Recognize Markdown images from the chat's documents folder or the web."""
    if "![" not in text:
        return
    for match in _IMAGE_SPAN.finditer(text):
        if streaming and match["unfinished"]:
            yield match.start(), match.end(), None
        if match["source"] is None:
            continue
        source = html.unescape(match["source"].removeprefix("<").removesuffix(">"))
        from_web = is_web_image_url(source)
        if not from_web:
            source = unquote(source)
        if from_web or (source.startswith("documents/") and image_mime_type(source)):
            alt = html.unescape(re.sub(r"\\([\\\[\]])", r"\1", match["alt"]))
            yield match.start(), match.end(), (source, alt)


def _decode_picture(read_image, bounds):
    mime, data = read_image()
    return decode_image(mime, data, bounds)


def _show_picture_when_mapped(widget, request, loaded, failed):
    """Subscribe on map and apply the result on the GTK thread."""

    def display(result):
        try:
            pixbuf = result.result()
        except OSError, ValueError, GLib.Error:
            failed()
        else:
            loaded(pixbuf)

    def start():
        result = request()
        if result.done():
            display(result)
        else:
            result.add_done_callback(lambda future: GLib.idle_add(display, future))

    def mapped(widget):
        widget.disconnect(handler)
        start()

    if widget.get_mapped():
        start()
    else:
        handler = widget.connect("map", mapped)


def load_picture(widget, read_image, bounds, loaded, failed):
    """Read and decode a thumbnail off the GTK thread when its widget maps."""
    _show_picture_when_mapped(
        widget, lambda: _image_loader.submit(_decode_picture, read_image, bounds), loaded, failed
    )


class ReplyImages:
    """Share image loads across all renders of one reply."""

    def __init__(self, documents):
        self.documents = documents
        self.requests = {}

    def request(self, source):
        if source not in self.requests:
            if is_web_image_url(source):
                executor = _web_image_loader
                read_image = partial(fetch_web_image, source)
            else:
                executor = _image_loader
                read_image = partial(self.documents.image, source.removeprefix("documents/"))
            self.requests[source] = executor.submit(
                _decode_picture, read_image, _EXPANDED_IMAGE_SIZE
            )
        return self.requests[source]


class _ImageLayout(Gtk.BoxLayout):
    def do_measure(self, widget, orientation, for_size):
        minimum, natural, baseline, natural_baseline = Gtk.BoxLayout.do_measure(
            self, widget, orientation, for_size
        )
        # Reserve height for the scaled image in the scrolling transcript.
        if orientation == Gtk.Orientation.VERTICAL and for_size >= 0:
            minimum = natural
        return minimum, natural, baseline, natural_baseline


class MarkdownImage(Gtk.Button):
    """A reply image that can expand inline."""

    def __init__(self, images, value, alignment=Gtk.Align.START):
        super().__init__(has_frame=False, halign=alignment, css_classes=["markdown-image"])
        self.set_layout_manager(_ImageLayout(orientation=Gtk.Orientation.VERTICAL))
        self.set_image(images, value)

    def set_image(self, images, value):
        source, alt = value
        from_web = is_web_image_url(source)
        description = alt or ("Web image" if from_web else source.rsplit("/", 1)[-1])
        status = Gtk.Label(label=f"Loading image: {description}", ellipsize=Pango.EllipsizeMode.END)
        self.set_child(status)
        self.set_sensitive(False)
        self.set_tooltip_text(description)
        if handler := getattr(self, "_zoom_handler", None):
            self.disconnect(handler)
            self._zoom_handler = None

        def loaded(pixbuf):
            if self.get_child() is not status:
                return
            full = Gdk.Texture.new_for_pixbuf(pixbuf)
            original_size = full.get_width(), full.get_height()
            size = fit_image_size(*original_size, _REPLY_IMAGE_SIZE)
            can_zoom = size != original_size
            preview = (
                Gdk.Texture.new_for_pixbuf(
                    pixbuf.scale_simple(*size, GdkPixbuf.InterpType.BILINEAR)
                )
                if can_zoom
                else full
            )
            picture = Gtk.Picture(
                paintable=preview,
                content_fit=Gtk.ContentFit.SCALE_DOWN,
                overflow=Gtk.Overflow.HIDDEN,
                alternative_text=description,
            )
            self.set_child(picture)
            self.set_sensitive(True)
            self.set_can_focus(can_zoom)
            if not can_zoom:
                return

            def zoom(_button):
                expand = picture.get_paintable() == preview
                picture.set_paintable(full if expand else preview)
                self.set_tooltip_text("Click to shrink" if expand else "Click to enlarge")

            self.set_tooltip_text("Click to enlarge")
            self._zoom_handler = self.connect("clicked", zoom)

        def failed():
            if self.get_child() is status:
                status.set_text(f"Image unavailable: {description}")

        _show_picture_when_mapped(self, lambda: images.request(source), loaded, failed)


_INLINE = re.compile(
    r"(?P<math>\\\(.{1,500}?\\\)|\\\[.{1,500}?\\\]"
    r"|\$\$(?!\s).{1,500}?\$\$|(?<!\$)\$(?![\s$])[^$\n]{1,500}?(?<!\s)\$(?!\$))"
    r"|\\([!\"#$%&'()*+,\-./:;<=>?@\[\]\\^_`{|}~])"
    r"|(`+)(.+?)\3"
    r'|!?\[([^\]]+)\]\(([^\s]+?)(?:\s+"[^"\n]*")?\)'
    r"|<(https?://[^<>\s]+|mailto:[^<>\s]+|[^<>\s@]+@[^<>\s@]+\.[^<>\s@]+)>"
    r"|((?:https?://|www\.)[^\s<>]+|(?<![\w.+-])[\w.+-]+@[\w-]+(?:\.[\w-]+)+)"
    r"|(\*{1,3}|(?<!\w)_{1,3}|~~|==|~|\^)(?=\S)(.+?)(?<=\S)\9(?:(?![\w*_])|(?<=~)|(?<=\^))"
    r"|\[\^([^\]\n]+)\]"
    r"|<(sub|sup|ins)>(.*?)</\12>",
    re.DOTALL,
)


_MATH_SYMBOLS = {
    "alpha": "α",  # noqa: RUF001 - intentional Greek math symbol
    "beta": "β",
    "gamma": "γ",  # noqa: RUF001
    "delta": "δ",
    "epsilon": "ε",
    "theta": "θ",
    "lambda": "λ",
    "mu": "μ",
    "pi": "π",
    "rho": "ρ",  # noqa: RUF001
    "sigma": "σ",  # noqa: RUF001
    "tau": "τ",
    "phi": "φ",
    "omega": "ω",
    "Delta": "Δ",
    "Gamma": "Γ",
    "Lambda": "Λ",
    "Pi": "Π",
    "Sigma": "Σ",
    "Omega": "Ω",
    "propto": "∝",
    "times": "×",  # noqa: RUF001
    "cdot": "·",
    "pm": "±",
    "leq": "≤",
    "geq": "≥",
    "neq": "≠",
    "approx": "≈",
    "infty": "∞",
    "to": "→",
    "rightarrow": "→",
    "partial": "∂",
    "sum": "∑",
    "int": "∫",
}


def math_markup(source):
    """Render a small, unambiguous TeX subset; leave other formulas untouched."""
    expression = source.strip()
    if source.startswith(("$$", r"\(", r"\[")):
        expression = source[2:-2].strip()
    elif source.startswith("$"):
        expression = source[1:-1].strip()
    if not expression or len(expression) > 500:
        return html.escape(source)

    def parse(text):
        parts = []
        index = 0
        while index < len(text):
            char = text[index]
            if char == "\\":
                command = re.match(r"\\([A-Za-z]+)", text[index:])
                if not command:
                    raise ValueError("unsupported math escape")
                name = command[1]
                index += len(command[0])
                if name == "frac":
                    arguments = []
                    for _ in range(2):
                        if index >= len(text) or text[index] != "{":
                            raise ValueError("incomplete fraction")
                        end = text.find("}", index + 1)
                        if end < 0 or "{" in text[index + 1 : end]:
                            raise ValueError("nested fraction")
                        raw = text[index + 1 : end]
                        value = parse(raw)
                        if re.search(r"[+\-*/=\s]", raw):
                            value = f"({value})"
                        arguments.append(value)
                        index = end + 1
                    parts.append(f"{arguments[0]}/{arguments[1]}")
                elif name in _MATH_SYMBOLS:
                    parts.append(_MATH_SYMBOLS[name])
                else:
                    raise ValueError("unsupported math command")
            elif char in "^_":
                index += 1
                if index >= len(text):
                    raise ValueError("incomplete script")
                if text[index] == "{":
                    end = text.find("}", index + 1)
                    if end < 0:
                        raise ValueError("incomplete script")
                    value = parse(text[index + 1 : end])
                    index = end + 1
                else:
                    value = parse(text[index])
                    index += 1
                tag = "sup" if char == "^" else "sub"
                parts.append(f"<{tag}>{value}</{tag}>")
            elif char in "{}$&":
                raise ValueError("unsupported math syntax")
            else:
                parts.append(html.escape(char))
                index += 1
        return "".join(parts)

    try:
        return parse(expression)
    except ValueError:
        return html.escape(source)


def inline_markdown(text, depth=0):
    """Escape model text and allow only explicitly generated Pango markup."""
    if depth > 12:
        return html.escape(text)
    parts, start = [], 0
    for match in _INLINE.finditer(text):
        parts.append(html.escape(html.unescape(text[start : match.start()])))
        math, escaped, ticks, code, title, url, autolink, bare, emphasis, body = match.groups()[:10]
        if match[0].startswith("!"):
            value = html.escape(match[0])
        elif math:
            value = math_markup(math)
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
        elif match[11]:
            number = html.escape(match[11])
            value = (
                f'<a href="#fn-{number}" title="Go to footnote {number}"><sup>{number}</sup></a>'
            )
        elif match[12]:
            tag = "u" if match[12] == "ins" else match[12]
            value = f"<{tag}>" + inline_markdown(match[13], depth + 1) + f"</{tag}>"
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


def model_references(text):
    """Render numbered model reference blocks as ordinary clickable Markdown."""
    blocks = list(re.finditer(r"<reference>\s*(.*?)\s*</reference>", text, re.DOTALL))
    targets = {
        number: url
        for block in blocks
        for number, url in re.findall(r"(?m)^\s*\[(\d+)\]\s+(https?://\S+)\s*$", block[1])
    }

    def render(match):
        lines = []
        for line in match[1].splitlines():
            entry = re.match(r"\s*\[(\d+)\]\s+(.+)", line)
            if entry and entry[1] in targets:
                # A separate URL entry is redundant once the source label links to it.
                if entry[2].strip() == targets[entry[1]]:
                    continue
                url = targets[entry[1]]
                label = entry[2]
                if re.match(r"https?://(?:[a-z-]+\.)?wikipedia\.org/", url):
                    title = re.split(r',\s*["“]', label, maxsplit=1)[0]
                    title = re.sub(r"\s*\(lines?\s+[^)]*\)\s*$", "", title, flags=re.IGNORECASE)
                    line = f"Source: [Wikipedia/{title}]({url})"
                else:
                    line = f"[{label}]({url})"
            lines.append(line)
        return "\n\n" + "\n\n".join(lines) + "\n\n"

    return re.sub(r"<reference>\s*(.*?)\s*</reference>", render, text, flags=re.DOTALL)


def reference_links(text):
    """Resolve reference-style links while leaving inline code untouched."""
    references = {}

    def definition(match):
        references[match[1].casefold()] = match[2].strip("<>")
        return ""

    text = re.sub(
        r'(?m)^ {0,3}\[(?!\^)([^\]\n]+)\]:[ \t]*(<[^<>\n]+>|\S+)(?:[ \t]+"[^"\n]*")?[ \t]*$',
        definition,
        text,
    )
    if not references:
        return text

    def link(match):
        if match[1]:
            return match[0]
        target = references.get((match[4] or match[3]).casefold())
        if target and match[0].startswith("!"):
            return f"![{match[3]}](<{target}>)"
        return f"[{match[3]}]({target})" if target else match[0]

    return re.sub(r"(`+)(.*?)\1|(?<!\\)!?\[([^\]\n]+)\](?:\[([^\]\n]*)\])?(?!\()", link, text)


def markdown_footnotes(text):
    if "[^" not in text:
        return text, []
    definitions, order = {}, []
    fences = [match.span() for match in CODE_BLOCK_PATTERN.finditer(text)]

    def definition(match):
        if any(start <= match.start() < end for start, end in fences):
            return match[0]
        definitions[match[1]] = re.sub(r"\n[ \t]{4}", "\n", match[2]).strip()
        return ""

    text = re.sub(r"(?m)^ {0,3}\[\^([^\]\n]+)\]:[ \t]*(.*(?:\n[ \t]{4}.*)*)$", definition, text)
    fences = [match.span() for match in CODE_BLOCK_PATTERN.finditer(text)]

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
            kind = "note" if content[0] == "[!NOTE]" else "quote"
            value = "\n".join(content[1:] if kind == "note" else content)
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
    if prose := "".join(paragraph).strip("\n"):
        yield "prose", prose


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

    def __init__(self, data, *, images=None):
        self.grid = Gtk.Grid()
        super().__init__(self.grid)
        self.add_css_class("markdown-table")
        self.cells = []
        self.data = ([], [])
        self.images = images
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
            for cell in self.cells.pop():
                self.grid.remove(cell)
        for row_index, row in enumerate(rows):
            if row_index == len(self.cells):
                self.cells.append([None] * len(alignment))
            for column, value in enumerate(row):
                if row_index < len(previous_rows) and previous_rows[row_index][column] == value:
                    continue
                cell = self.cells[row_index][column]
                has_image = self.images is not None and any(image_spans(value))
                if cell is None or has_image != isinstance(cell, MarkdownText):
                    if cell is not None:
                        self.grid.remove(cell)
                    cell = self._add_cell(row_index, column, alignment[column], has_image)
                    self.cells[row_index][column] = cell
                if has_image:
                    cell.set_text(value)
                else:
                    set_markdown_text(cell, value)
                    self.update_content_width(cell)
        self.data = data

    def _add_cell(self, row, column, xalign, has_image):
        if has_image:
            align = {0.0: Gtk.Align.START, 0.5: Gtk.Align.CENTER, 1.0: Gtk.Align.END}
            cell = MarkdownText(images=self.images, image_alignment=align[xalign])
        else:
            cell = markdown_label()
            cell.set_max_width_chars(500)
            cell.set_xalign(xalign)
        cell.set_hexpand(True)
        cell.add_css_class("table-cell")
        if row == 0:
            cell.add_css_class("table-heading")
        elif row % 2 == 0:
            cell.add_css_class("table-alternate")
        self.grid.attach(cell, column, row, 1, 1)
        return cell


_INLINE_PROSE_MARKUP = re.compile(r"[\\`*_~=^$<&@\[]|https?://|www\.")


def _wrapped_fragments(text, wrap_boundary):
    """Yield source offsets and display text at natural line breaks."""
    start = 0
    while len(text) - start > 2048:
        end = start + wrap_boundary(text[start : start + 2048])
        yield start, text[start:end].removesuffix("\n").rstrip(" ")
        start = end
    yield start, text[start:]


def _stream_text_blocks(blocks, wrap_boundary):
    """Bound plain-text layouts while streaming. Keep inline Markdown spans intact."""
    for kind, value in blocks:
        if kind != "prose" or len(value) < 2048:
            yield kind, value
            continue
        # Inline Markdown is scoped to its paragraph. One emphasis marker in
        # another paragraph must not force GTK to reshape an entire response.
        for i, paragraph in enumerate(value.split("\n\n")):
            if len(paragraph) < 2048 or _INLINE_PROSE_MARKUP.search(paragraph):
                yield "prose", paragraph
                continue
            for start, fragment in _wrapped_fragments(paragraph, wrap_boundary):
                yield "paragraph-start" if i and not start else "fragment", fragment


def _with_images(blocks, *, streaming=False):
    for kind, value in blocks:
        if kind not in ("prose", "item"):
            yield kind, value
            continue
        text = value if kind == "prose" else value[2]
        start = 0
        for left, right, image in image_spans(text, streaming=streaming):
            lead = text[start:left].strip("\n")
            if kind == "item" and not start:
                yield kind, (*value[:2], lead)
            elif lead.strip():
                yield "prose", lead
            if image is not None:
                yield "image", image
            start = right
        if not start:
            yield kind, value
        elif tail := text[start:].strip():
            yield "prose", tail


class _WrappedBoxLayout(Gtk.BoxLayout):
    def do_allocate(self, widget, width, height, baseline):
        Gtk.BoxLayout.do_allocate(self, widget, width, height, baseline)
        if width > 0 and width != widget._wrap_width and widget._wrap_breaks:
            widget._wrap_width = width
            if not widget._rewrap_id:
                widget._rewrap_id = GLib.idle_add(widget._rewrap)


class MarkdownText(Gtk.Box):
    """Share prose/table rendering between stored messages and the streaming tail."""

    def __init__(self, text="", *, streaming=False, images=None, image_alignment=Gtk.Align.START):
        super().__init__(orientation=Gtk.Orientation.VERTICAL)
        self.add_css_class("markdown-text")
        self._wrap_width = 680
        self._wrap_breaks = {}
        self._rewrap_id = 0
        self.streaming = streaming
        self.images = images
        self.image_alignment = image_alignment
        self.blocks = []
        self._source_text = None
        self.set_text(text)

    def _rewrap(self):
        self._rewrap_id = 0
        self._wrap_breaks.clear()
        text, self._source_text = self._source_text, None
        self.set_text(text)
        return False

    def _wrap_boundary(self, sample):
        if sample not in self._wrap_breaks:
            if not isinstance(self.get_layout_manager(), _WrappedBoxLayout):
                self.set_layout_manager(_WrappedBoxLayout(orientation=Gtk.Orientation.VERTICAL))
            layout = self.create_pango_layout(sample)
            layout.set_width(self._wrap_width * Pango.SCALE)
            layout.set_wrap(Pango.WrapMode.WORD_CHAR)
            # Leave the incomplete last display line for the following label.
            line = layout.get_line_readonly(layout.get_line_count() - 1)
            end = len(sample.encode("utf-8")[: line.start_index].decode("utf-8"))
            self._wrap_breaks[sample] = end or len(sample)
        return self._wrap_breaks[sample]

    def prepend_prose(self, text, width):
        """Yield between preparing and revealing the pieces of one archived paragraph."""
        if (
            len(text) < 8192
            or width <= 0
            or _INLINE_PROSE_MARKUP.search(text)
            or list(prose_blocks(text)) != [("prose", text)]
            or "|" in text
        ):
            self.set_text(text)
            yield True
            return

        # Find real wrap boundaries from the start, then reveal from the end.
        # Keep the same MarkdownText so resize rewraps the visible text together.
        self._wrap_width = width
        fragments = []
        for piece in _wrapped_fragments(text, self._wrap_boundary):
            fragments.append(piece)
            yield False

        self.add_css_class("markdown-fragmented")
        for start, value in reversed(fragments):
            if self._wrap_width != width:
                # A resize invalidated the old line starts.
                self.set_text(text)
                yield True
                return
            label = render_markdown_block("fragment", value)
            self.prepend(label)
            self.blocks.insert(0, ("fragment", label, value))
            self._source_text = text[start:]
            yield True

    def set_text(self, text, *, finished=False):
        if finished and self.streaming:
            self.streaming = False
            self._source_text = None
        # While a fenced block streams, its preceding prose often stays identical.
        if text == self._source_text:
            return
        self._source_text = text
        self.anchors = {}
        text = model_references(text)
        # Treat the model's nonstandard note wrapper as a Markdown blockquote.
        text = re.sub(
            r"<note>\s*(.*?)(?:\s*</note>|$)",
            lambda match: (
                "\n\n> [!NOTE]\n"
                + "\n".join("> " + line for line in match[1].strip().splitlines())
                + "\n\n"
            ),
            text,
            flags=re.DOTALL | re.IGNORECASE,
        )
        # Hide an unfinished wrapper while its contents are streaming.
        text = re.sub(r"(?im)^\s*</?reference>\s*$", "", text)
        text, notes = markdown_footnotes(text)
        text = reference_links(text)
        blocks = [
            block
            for kind, value in markdown_blocks(text)
            for block in (prose_blocks(value) if kind == "prose" else [(kind, value)])
        ]
        blocks.extend(notes)
        if self.images is not None and "![" in text:
            blocks = list(_with_images(blocks, streaming=self.streaming))
        grouped = []
        fragmented = self.streaming or len(text) > 8192
        if fragmented:
            self.add_css_class("markdown-fragmented")
        else:
            self.remove_css_class("markdown-fragmented")
        for kind, value in (
            _stream_text_blocks(blocks, self._wrap_boundary) if fragmented else blocks
        ):
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
                child = render_markdown_block(
                    kind, value, images=self.images, image_alignment=self.image_alignment
                )
                self.append(child)
            else:
                _, child, previous = self.blocks[index]
                if value != previous:
                    render_markdown_block(kind, value, child, images=self.images)
            continuation = (
                index
                and kind == "fragment"
                and blocks[index - 1][0] in ("fragment", "paragraph-start")
            )
            if fragmented and index and not continuation:
                child.add_css_class("markdown-block-gap")
            else:
                child.remove_css_class("markdown-block-gap")
            block = (kind, child, value)
            if index == len(self.blocks):
                self.blocks.append(block)
            else:
                self.blocks[index] = block
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
    parts, offset, previous = [], 0, None
    has_anchors = any(kind in ("heading", "footnote") for kind, _ in blocks)
    label.markdown_anchors = []
    for kind, value in blocks:
        if kind == "prose" and not value:
            # Blank lines separate adjacent blocks once, including loose list items.
            previous = kind
            continue
        separator = "\n" if kind == previous == "item" else "\n\n"
        if parts:
            parts.append(separator)
            offset += len(separator)
        if kind == "heading":
            level, text = value
            size = ("xx-large", "x-large", "large", "medium", "medium", "small")[level - 1]
            markup = f'<span size="{size}" weight="bold">{inline_markdown(text)}</span>'
            anchor = re.sub(r"[^\w\s-]", "", text.casefold()).replace(" ", "-")
            label.markdown_anchors.append((anchor, offset))
        elif kind in ("item", "footnote"):
            indent, marker, text = value if kind == "item" else (0, f"{value[0]}.", value[1])
            markup = (
                "\u2003" * min(indent // 2, 20) + html.escape(marker) + " " + inline_markdown(text)
            )
            if kind == "footnote":
                label.markdown_anchors.append((f"fn-{value[0]}", offset))
        else:
            markup = inline_markdown(value)
        parts.append(markup)
        if has_anchors:
            offset += len(html.unescape(re.sub(r"<[^>]*>", "", markup)).encode("utf-8"))
        previous = kind
    if any(kind == "heading" for kind, _ in blocks):
        label.remove_css_class("chat-prose")
    else:
        label.add_css_class("chat-prose")
    label.set_markup("".join(parts))


def render_markdown_block(
    kind, value, widget=None, *, images=None, image_alignment=Gtk.Align.START
):
    if kind == "table":
        if widget is None:
            return MarkdownTable(value, images=images)
        widget.set_data(value)
    elif kind == "image":
        if widget is None:
            return MarkdownImage(images, value, image_alignment)
        widget.set_image(images, value)
    elif kind == "rule":
        return (
            widget if widget is not None else Gtk.Separator(orientation=Gtk.Orientation.HORIZONTAL)
        )
    elif kind == "code":
        if widget is None:
            widget = append_code_block(None, value)[0]
        else:
            widget.scroll.get_child().get_child().set_text(value)
    elif kind in ("quote", "note"):
        if widget is None:
            widget = Gtk.Frame()
            content = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
            content.add_css_class("markdown-quote-content")
            title = Gtk.Label(label="Note:" if kind == "note" else "Quote:", xalign=0)
            title.add_css_class("dim-label")
            content.append(title)
            content.append(MarkdownText(images=images))
            widget.set_child(content)
            widget.add_css_class("markdown-quote")
        widget.get_child().get_last_child().set_text(value)
    else:
        if widget is None:
            widget = markdown_label()
        if kind in ("fragment", "paragraph-start"):
            widget.add_css_class("chat-prose")
            widget.set_text(value)
        else:
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


def append_markdown(parent, text, command_extractor=None, command_handler=None, *, images=None):
    """Append safe Markdown, tables, and fenced code or command blocks."""

    text, notes = markdown_footnotes(text)
    last = 0
    for match in CODE_BLOCK_PATTERN.finditer(text):
        if text[last : match.start()]:
            parent.append(MarkdownText(text[last : match.start()], images=images))
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
        parent.append(MarkdownText(text[last:], images=images))
    if notes:
        footer = MarkdownText()
        for _kind, value in notes:
            child = render_markdown_block("footnote", value)
            footer.append(child)
            footer.anchors[f"fn-{value[0]}"] = child
        parent.append(footer)
