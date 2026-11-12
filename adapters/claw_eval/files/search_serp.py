"""Web search for Claw-Eval's web_real service, replacing upstream's search_serp.py.

Same module name, function signature and return shape as the file it displaces; only the engine behind it changes:

* Vendor. Upstream queries Novada (scraperapi.novada.com) with a Novada key. This version sends the request through
  ScraperAPI (``SERP_API_URL``); inside a task that is the local stand-in which the host relay answers with the
  host's ScraperAPI key, so no key enters the container.
* Engine. ScraperAPI's structured Google endpoint returns ``/goto?url=<protobuf>`` redirects that expire within
  minutes, so most results were dead by the time the agent fetched them. DuckDuckGo's HTML endpoint puts the plain
  URL in ``uddg=`` and answers in about 2 s instead of about 35 s.

Input:  query (str), timeout (int), num (int), start (int)
Output: {"status": <int>, "output": <list[dict]>}
"""

import html
import itertools
import os
import re
import sys
import threading
import urllib.parse

import requests

SERP_API_URL = os.getenv("SERP_API_URL", "https://api.scraperapi.com/")
DDG_URL = os.getenv("DDG_URL", "https://html.duckduckgo.com/html/")

# When this module talks to ScraperAPI directly (outside a task), every SERP_DEV_KEY* in the environment is used
# round-robin: ScraperAPI caps concurrency per key. Inside a task there is no key; the stand-in drops the parameter.
_KEYS = [v for k, v in sorted(os.environ.items())
         if k == "SERP_DEV_KEY" or k.startswith("SERP_DEV_KEY_")] or ["YOUR_API_KEY"]
_next = itertools.cycle(_KEYS)
_key_lock = threading.Lock()


def api_key() -> str:
    """One key per call, cycling — the service is threaded, so guard the cycle."""
    with _key_lock:
        return next(_next)


SERP_DEV_KEY = _KEYS[0]  # kept for callers that read the module attribute

_RESULT_RE = re.compile(
    r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>', re.S)
_SNIPPET_RE = re.compile(r'class="result__snippet"[^>]*>(.*?)</a>', re.S)
_TAG_RE = re.compile(r"<[^>]+>")


def _detect_language(query: str) -> tuple[str, str]:
    if re.search(r"[一-鿿]", query):
        return "zh", "cn"
    return "en", "us"


def _text(raw: str) -> str:
    return html.unescape(_TAG_RE.sub("", raw)).strip()


def search_serp(
    query: str,
    timeout: int = 90,
    num: int = 10,
    start: int = 1,
    raw_save_path: str | None = None,
) -> dict:
    """Search the live web.

    Args:
        query: Search query string.
        timeout: Request timeout in seconds.
        num: Number of results (1-10).
        start: 1-based result offset.
        raw_save_path: If set, the raw response body is written here.

    Returns:
        dict with keys:
            status (int): HTTP status code, or -1 on error.
            output (list[dict]): List of result dicts with keys:
                title, link, snippet, source, date, query.
    """
    _hl, gl = _detect_language(query)
    target = (f"{DDG_URL}?q={urllib.parse.quote(query)}"
              f"&kl={'cn-zh' if gl == 'cn' else 'us-en'}")
    want = max(1, min(num, 10))
    skip = max(start - 1, 0)
    try:
        resp = requests.get(SERP_API_URL,
                            params={"api_key": api_key(), "url": target},
                            timeout=timeout)
        if raw_save_path and resp.status_code == 200:
            os.makedirs(os.path.dirname(raw_save_path) or ".", exist_ok=True)
            with open(raw_save_path, "w", encoding="utf-8") as fh:
                fh.write(resp.text)
        if resp.status_code != 200:
            print(f"SERP HTTP {resp.status_code} for {query!r}",
                  file=sys.stderr, flush=True)
            return {"status": resp.status_code, "output": []}

        results = []
        for match in _RESULT_RE.finditer(resp.text):
            href = html.unescape(match.group(1))
            link = urllib.parse.parse_qs(
                urllib.parse.urlparse(href).query).get("uddg", [href])[0]
            if link.startswith("//"):
                link = "https:" + link
            snippet = ""
            after = _SNIPPET_RE.search(resp.text, match.end(), match.end() + 4000)
            if after:
                snippet = _text(after.group(1))
            results.append({
                "title": _text(match.group(2)),
                "link": link,
                "snippet": snippet,
                "source": urllib.parse.urlparse(link).netloc,
                "date": "",
                "query": query,
            })
        results = results[skip:skip + want]
        if not results:
            print(f"SERP empty: status={resp.status_code} query={query!r}",
                  file=sys.stderr, flush=True)
        return {"status": resp.status_code, "output": results}
    except Exception as exc:  # noqa: BLE001
        print(f"SERP error for {query!r}: {exc}", file=sys.stderr, flush=True)
        return {"status": -1, "output": []}


if __name__ == "__main__":
    import json

    result = search_serp("Python web scraping", num=3)
    print(f"status={result['status']}  count={len(result['output'])}")
    print(json.dumps(result["output"], indent=2, ensure_ascii=False)[:1000])
