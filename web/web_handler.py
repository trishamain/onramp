"""Serves the demo page.

A zip-packaged Lambda with no dependencies, deliberately separate from the
1.87 GB query container. Serving static HTML from that image would mean a 12.5 s
cold start to render a page, which is absurd; this one cold-starts in well under
a second because it imports nothing but the standard library.

WHY API GATEWAY RATHER THAN S3 STATIC WEBSITE HOSTING
-----------------------------------------------------
Fewer moving parts, and the count is not close:

  S3 static website  needs a second bucket, a block-public-access EXCEPTION on
                     it, a public bucket policy, a website configuration, and
                     CloudFront on top of all that to get HTTPS at all. It also
                     puts the page on a different origin from /ask, so every
                     query pays a CORS preflight.

  API Gateway route  needs this file and one GET method. The page is served from
                     the same origin as /ask, so CORS is not a consideration --
                     no preflight, no headers to get wrong. HTTPS is automatic.

The S3 route would also mean the project's only publicly-readable bucket exists
purely to host a demo page, which undercuts the block-all-public-access posture
the corpus bucket is configured with.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

# Read once at import, cached for the life of the execution environment. The
# page is ~14 KB; re-reading it per request would be pointless I/O.
_PAGE = (Path(__file__).parent / "index.html").read_text(encoding="utf-8")


def handler(_event: dict[str, Any], _context: Any) -> dict[str, Any]:
    return {
        "statusCode": 200,
        "headers": {
            "Content-Type": "text/html; charset=utf-8",
            # No caching: the page is tiny and a stale demo UI during a live
            # presentation is a worse outcome than one extra request.
            "Cache-Control": "no-store",
            # The page loads no third-party anything, so the strictest sensible
            # CSP costs nothing and documents that fact. 'unsafe-inline' is
            # required because the CSS and JS are inline by design -- the
            # requirement is one self-contained file.
            "Content-Security-Policy": (
                "default-src 'none'; "
                "style-src 'unsafe-inline'; "
                "script-src 'unsafe-inline'; "
                "connect-src 'self'; "
                "form-action 'none'; "
                "frame-ancestors 'none'"
            ),
            "X-Content-Type-Options": "nosniff",
            "Referrer-Policy": "no-referrer",
        },
        "body": _PAGE,
    }
