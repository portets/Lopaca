"""Download images from public HTTP(S) URLs for the chat renderer."""

import http.client
import socket
import ssl
from contextlib import closing
from ipaddress import ip_address
from time import monotonic
from urllib.parse import quote, urljoin, urlsplit

from images import IMAGE_SIZE_LIMIT, supported_image_mimes

_REDIRECTS = {301, 302, 303, 307, 308}
_SOCKET_TIMEOUT = 5
_DOWNLOAD_TIMEOUT = 15
_HEADERS = {
    "Accept": "image/*",
    "User-Agent": "Lopaca (https://github.com/portets/Lopaca)",
}


def is_web_image_url(source):
    return source.lower().startswith(("http://", "https://"))


def _public_address(address):
    ip = ip_address(address)
    ip = getattr(ip, "ipv4_mapped", None) or ip
    return ip.is_global and not ip.is_multicast


class _ImageConnection(http.client.HTTPConnection):
    def __init__(self, host, port, tls, deadline):
        self.default_port = 443 if tls else 80
        super().__init__(host, port, timeout=_SOCKET_TIMEOUT)
        self.tls = tls
        self.deadline = deadline

    def connect(self):
        # Connect to a vetted IP directly. Keep the original hostname for Host and TLS.
        addresses = socket.getaddrinfo(self.host, self.port, type=socket.SOCK_STREAM)
        for family, kind, protocol, _, address in addresses:
            if not _public_address(address[0]):
                continue
            remaining = self.deadline - monotonic()
            if remaining <= 0:
                raise TimeoutError("Image download timed out.")
            sock = socket.socket(family, kind, protocol)
            try:
                sock.settimeout(min(self.timeout, remaining))
                sock.connect(address)
                if self.tls:
                    sock = ssl.create_default_context().wrap_socket(sock, server_hostname=self.host)
            except OSError:
                sock.close()
                continue
            self.sock = sock
            return
        raise OSError("Could not connect to a public image host.")


def fetch_web_image(url):
    """Return the MIME type and bounded image bytes, rechecking each redirect."""
    try:
        return _download_image(url)
    except http.client.HTTPException as exc:
        raise OSError("Invalid response from the image host.") from exc


def _download_image(url):
    deadline = monotonic() + _DOWNLOAD_TIMEOUT
    for _ in range(5):
        parts = urlsplit(url)
        if not is_web_image_url(url) or not parts.hostname or parts.username is not None:
            raise ValueError("Use a public HTTP(S) image URL.")
        tls = parts.scheme == "https"
        port = parts.port if parts.port is not None else (443 if tls else 80)
        path = quote(parts.path or "/", safe="/%:@!$&'()*+,;=")
        if parts.query:
            path += "?" + quote(parts.query, safe="/%?:@!$&'()*+,;=")

        with closing(_ImageConnection(parts.hostname, port, tls, deadline)) as connection:
            connection.request("GET", path, headers=_HEADERS)
            with connection.getresponse() as response:
                if response.status in _REDIRECTS and (location := response.getheader("Location")):
                    url = urljoin(url, location)
                    continue
                content_type = response.headers.get_content_type()
                mime = supported_image_mimes().get(content_type)
                if response.status != 200 or (
                    mime is None and content_type != "application/octet-stream"
                ):
                    raise ValueError("URL did not return a supported image.")
                if response.length is not None and response.length > IMAGE_SIZE_LIMIT:
                    raise ValueError("Web image is too large.")
                data = bytearray()
                while chunk := response.read1(min(65536, IMAGE_SIZE_LIMIT + 1 - len(data))):
                    data.extend(chunk)
                    if len(data) > IMAGE_SIZE_LIMIT:
                        raise ValueError("Web image is too large.")
                    if monotonic() >= deadline:
                        raise TimeoutError("Image download timed out.")
                return mime, bytes(data)
    raise ValueError("Too many image redirects.")
