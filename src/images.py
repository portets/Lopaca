"""Image formats and conversion using the installed GdkPixbuf loaders."""

from functools import cache
from pathlib import Path

IMAGE_SIZE_LIMIT = 16 * 1024 * 1024
_NATIVE_IMAGE_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".bmp": "image/bmp",
    ".gif": "image/gif",
}
NATIVE_IMAGE_MIMES = frozenset(_NATIVE_IMAGE_TYPES.values())
_NATIVE_SIGNATURES = {
    "image/png": (b"\x89PNG\r\n\x1a\n",),
    "image/jpeg": (b"\xff\xd8\xff",),
    "image/bmp": (b"BM",),
    "image/gif": (b"GIF87a", b"GIF89a"),
}


def _pixbuf_libraries():
    import gi

    gi.require_version("GdkPixbuf", "2.0")
    from gi.repository import GdkPixbuf, GLib

    return GdkPixbuf, GLib


@cache
def _image_formats():
    extensions = dict(_NATIVE_IMAGE_TYPES)
    mime_types = {mime: mime for mime in NATIVE_IMAGE_MIMES}
    try:
        pixbuf, _ = _pixbuf_libraries()
    except ImportError, ValueError:
        return extensions, mime_types
    for image_format in pixbuf.Pixbuf.get_formats():
        if image_format.is_disabled() or not (types := image_format.get_mime_types()):
            continue
        mime_types.update((mime.lower(), mime) for mime in types)
        for extension in image_format.get_extensions():
            mime = f"image/{extension}"
            extensions.setdefault(f".{extension.lower()}", mime if mime in types else types[0])
    return extensions, mime_types


def image_mime_type(path):
    name = str(path).lower()
    if name.endswith("/"):
        return None
    extensions, _ = _image_formats()
    if name.endswith(".svg.gz"):
        return extensions.get(".svg.gz")
    return extensions.get(Path(name).suffix)


def supported_image_mimes():
    return _image_formats()[1]


def read_image(stream, mime):
    data = stream.read(IMAGE_SIZE_LIMIT + 1)
    if len(data) > IMAGE_SIZE_LIMIT:
        raise ValueError(f"Image exceeds the {IMAGE_SIZE_LIMIT // (1024 * 1024)} MiB limit.")
    if mime in _NATIVE_SIGNATURES and not data.startswith(_NATIVE_SIGNATURES[mime]):
        raise ValueError("Image data does not match its extension.")
    return data


def decode_image(mime, data, bounds=None):
    pixbuf, glib = _pixbuf_libraries()
    try:
        loader = pixbuf.PixbufLoader.new_with_mime_type(mime) if mime else pixbuf.PixbufLoader.new()
        if bounds is not None:
            loader.connect(
                "size-prepared",
                lambda decoder, width, height: decoder.set_size(
                    *fit_image_size(width, height, bounds)
                ),
            )
        try:
            loader.write(data)
        finally:
            loader.close()
        if result := loader.get_pixbuf():
            return result
        raise ValueError("Image has no decodable frame.")
    except glib.Error as exc:
        raise ValueError(f"Could not decode image: {exc}") from exc


def fit_image_size(width, height, bounds):
    scale = min(1, bounds[0] / width, bounds[1] / height)
    return max(1, round(width * scale)), max(1, round(height * scale))


def image_to_png(mime, data):
    image = decode_image(mime, data)
    _, glib = _pixbuf_libraries()
    try:
        _, encoded = image.save_to_bufferv("png", ["compression"], ["1"])
    except glib.Error as exc:
        raise ValueError(f"Could not encode PNG: {exc}") from exc
    if len(encoded) > IMAGE_SIZE_LIMIT:
        raise ValueError(
            f"Converted PNG exceeds the {IMAGE_SIZE_LIMIT // (1024 * 1024)} MiB limit."
        )
    return encoded
