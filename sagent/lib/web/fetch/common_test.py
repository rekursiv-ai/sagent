"""Tests for sagent.lib.web.fetch."""

from __future__ import annotations

import gzip
import io
import zlib

import brotli
import pytest
import zstandard

from sagent.lib.web.fetch.common import (
    apply_redirect,
    bracket_ipv6,
    decompress,
    rewrite_origin,
)


class TestRewriteOrigin:
    def test_cross_origin_rewrite(self) -> None:
        out = rewrite_origin({"Origin": "https://a.com"}, "https://b.com/land")
        assert out["Origin"] == "https://b.com"

    def test_no_origin_header_unchanged(self) -> None:
        h = {"Accept": "*/*"}
        assert rewrite_origin(h, "https://b.com/x") is h

    def test_ipv6_target_is_bracketed(self) -> None:
        # REV2061-003: a v6 redirect target must yield a BRACKETED Origin;
        # "https://2606:...::1" is an invalid Origin (colons unbracketed).
        out = rewrite_origin({"Origin": "https://a.com"}, "https://[2606:4700::1]/x")
        assert out["Origin"] == "https://[2606:4700::1]"

    def test_case_variant_origin_is_rewritten_not_leaked(self) -> None:
        # REVE559-003: HTTP field names are case-insensitive. A caller-supplied
        # "origin" (lowercase) must still be rewritten, not leaked verbatim.
        out = rewrite_origin({"origin": "https://a.com"}, "https://b.com/x")
        assert not any(
            v == "https://a.com" for k, v in out.items() if k.lower() == "origin"
        )
        assert any(
            v == "https://b.com" for k, v in out.items() if k.lower() == "origin"
        )


class TestApplyRedirect:
    def test_303_drops_case_variant_content_type(self) -> None:
        # REVE559-002: a 303 POST->GET must drop Content-Type regardless of case.
        headers, method, body = apply_redirect(
            "https://x/submit",
            {"content-type": "application/json", "Accept": "*/*"},
            "POST",
            b"{}",
            303,
            "https://x/result",
        )
        assert method == "GET"
        assert body is None
        assert not any(k.lower() == "content-type" for k in headers)

    def test_302_downgrades_post_to_get(self) -> None:
        _headers, method, body = apply_redirect(
            "https://x/submit", {}, "POST", b"{}", 302, "https://x/land"
        )
        assert method == "GET"
        assert body is None

    def test_307_preserves_method_and_body(self) -> None:
        _headers, method, body = apply_redirect(
            "https://x/submit", {}, "POST", b"{}", 307, "https://x/land"
        )
        assert method == "POST"
        assert body == b"{}"

    def test_cross_origin_drops_cookie_and_hints(self) -> None:
        headers, _m, _b = apply_redirect(
            "https://a.com/1",
            {"Cookie": "SID=x", "sec-ch-ua-arch": '"x86"', "Accept": "*/*"},
            "GET",
            None,
            302,
            "https://b.com/2",
        )
        assert "Cookie" not in headers
        assert "sec-ch-ua-arch" not in headers
        assert headers.get("Accept") == "*/*"  # non-origin-bound survives

    def test_same_origin_keeps_cookie_and_hints(self) -> None:
        headers, _m, _b = apply_redirect(
            "https://a.com/1",
            {"Cookie": "SID=x", "sec-ch-ua-arch": '"x86"'},
            "GET",
            None,
            302,
            "https://a.com/2",
        )
        assert headers.get("Cookie") == "SID=x"
        assert headers.get("sec-ch-ua-arch") == '"x86"'


class TestDecompress:
    def test_gzip(self) -> None:
        data = b"hello world"
        assert decompress(gzip.compress(data), "gzip") == data

    def test_deflate(self) -> None:
        data = b"hello world"
        assert decompress(zlib.compress(data), "deflate") == data

    def test_brotli(self) -> None:
        data = b"hello world"
        assert decompress(brotli.compress(data), "br") == data

    def test_zstd(self) -> None:
        data = b"hello world"
        compressed = zstandard.ZstdCompressor().compress(data)
        assert decompress(compressed, "zstd") == data

    def test_zstd_streaming_frame_no_size(self) -> None:
        # Streaming-mode frames omit decompressed size from the header;
        # `ZstdDecompressor.decompress()` rejects them. Real servers
        # (e.g. Cloudflare) emit such frames -- we must handle them.
        data = b"hello world " * 1000
        buf = io.BytesIO()
        with zstandard.ZstdCompressor().stream_writer(buf, closefd=False) as w:
            _ = w.write(data)
        assert decompress(buf.getvalue(), "zstd") == data

    def test_identity(self) -> None:
        assert decompress(b"raw", "identity") == b"raw"

    def test_empty_encoding(self) -> None:
        assert decompress(b"raw", "") == b"raw"

    def test_unknown_encoding_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown Content-Encoding"):
            decompress(b"raw", "unknown")

    def testdecompression_failure_raises(self) -> None:
        with pytest.raises(ValueError, match="Decompression failed"):
            decompress(b"not gzip", "gzip")

    def test_raw_deflate_without_zlib_header(self) -> None:
        # REV2A-002: some servers emit raw DEFLATE (no zlib wrapper); a browser
        # falls back to wbits=-MAX_WBITS. We must decode it, not raise.
        data = b"hello world"
        raw = zlib.compress(data)[2:-4]  # strip zlib header + adler checksum
        assert decompress(raw, "deflate") == data

    def test_chained_content_encoding(self) -> None:
        # REV2A-003: chained "gzip, br" is RFC-legal; apply right-to-left.
        data = b"hello world"
        chained = gzip.compress(brotli.compress(data))
        assert decompress(chained, "br, gzip") == data


class TestIPv6Bracketing:
    def test_ipv6_address_bracketed(self) -> None:
        assert bracket_ipv6("2606:4700::6810:7c60") == "[2606:4700::6810:7c60]"

    def test_already_bracketed_unchanged(self) -> None:
        assert bracket_ipv6("[::1]") == "[::1]"

    def test_ipv4_unchanged(self) -> None:
        assert bracket_ipv6("93.184.216.34") == "93.184.216.34"

    def test_hostname_unchanged(self) -> None:
        assert bracket_ipv6("example.com") == "example.com"
