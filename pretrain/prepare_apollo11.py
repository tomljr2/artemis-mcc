"""Download the Apollo 11 air-to-ground transcript and turn it into plain text.

Source: NASA Apollo Lunar Surface Journal, Technical Air-to-Ground Voice Transcription
(GOSS NET 1). Released by NASA with no copyright asserted.

Run:  python pretrain/prepare_apollo11.py
"""

import html
import re
import ssl
import urllib.request
from collections import Counter
from pathlib import Path

import certifi

URL = "https://www.nasa.gov/wp-content/uploads/static/history/alsj/a11/a11transcript_tec.html"
RAW_PATH = Path("data/raw/apollo11_tec.html")
TEXT_PATH = Path("data/processed/apollo11_tec.txt")


def download() -> str:
    """Fetch the HTML once and cache it, so reruns don't hit NASA's server."""
    if not RAW_PATH.exists():
        RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
        # Verify HTTPS against Mozilla's current CA list (certifi) rather than the OS store,
        # which can hold expired root certificates that make valid sites fail to verify.
        context = ssl.create_default_context(cafile=certifi.where())
        with urllib.request.urlopen(URL, context=context) as response:
            RAW_PATH.write_bytes(response.read())
    raw = RAW_PATH.read_bytes()
    # The OCR misread one "o" as the byte 0xA2, which is not valid UTF-8. Fix it here, then
    # decode strictly so any other bad byte raises an error instead of slipping through.
    raw = raw.replace(b"my way \xa2ut", b"my way out")
    return raw.decode("utf-8")


def to_plain_text(page: str) -> str:
    """Strip the HTML markup, keeping timestamps, speakers, and dialogue."""
    body = page.split("<body>", 1)[1]
    body = re.sub(r"<br\s*/?>\n?", "\n", body)  # each <br> (and the newline after it) -> one newline
    body = re.sub(r"<[^>]+>", "", body)  # drop every other tag
    body = html.unescape(body)  # &nbsp; -> space, &amp; -> &, ...
    body = re.sub(r"[ \t\xa0]+", " ", body)  # collapse runs of spaces
    lines = [line.strip() for line in body.splitlines()]
    # Page headers like "(GOSS NET 1) Tape 4/4 Page 28" come from the paper layout, not the
    # conversation, so leave them out.
    lines = [line for line in lines if not line.startswith("(GOSS NET 1)")]
    text = "\n".join(lines)
    return re.sub(r"\n{3,}", "\n\n", text).strip() + "\n"  # at most one blank line


def main() -> None:
    text = to_plain_text(download())
    TEXT_PATH.parent.mkdir(parents=True, exist_ok=True)
    TEXT_PATH.write_text(text, encoding="utf-8")

    chars = sorted(set(text))
    print(f"saved {TEXT_PATH}")
    print(f"length: {len(text):,} characters")
    print(f"unique characters (vocabulary): {len(chars)}")
    print(ascii("".join(chars)))  # ascii() escapes anything the terminal can't display
    print("rarest characters:", ascii(Counter(text).most_common()[-10:]))


if __name__ == "__main__":
    main()
