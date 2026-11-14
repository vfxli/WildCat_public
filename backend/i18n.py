# -*- coding: utf-8 -*-
"""Dynamic JSON-Driven Internationalization (i18n) Registry & Translator.

Adding a new language requires ZERO code changes: drop `frontend/locales/<lang>.json`
(and optionally `frontend/locales/private/<lang>.json` in the private edition) with a
`_meta` header (`code`, `label`, `short`, `htmlLang`, `order`).
"""
import contextvars
import json
import os
import re
import threading

from . import db

LOCALES_DIR = os.path.join(db.ROOT, "frontend", "locales")
PRIVATE_LOCALES_DIR = os.path.join(LOCALES_DIR, "private")
DEFAULT_LOCALE = os.environ.get("WILDCAT_DEFAULT_LANG", "zh-CN")

_CURRENT_LANG: contextvars.ContextVar[str] = contextvars.ContextVar("wc_lang", default=DEFAULT_LOCALE)
_LOCK = threading.RLock()
_FILE_CACHE: dict[str, tuple[tuple[int, int], dict]] = {}
_BUNDLE_CACHE: dict[str, tuple[tuple, dict]] = {}
_PLACEHOLDER_RE = re.compile(r"\{([a-zA-Z0-9_]+)\}")


def get_current_lang() -> str:
    return _CURRENT_LANG.get() or DEFAULT_LOCALE


def set_current_lang(lang: str | None) -> contextvars.Token:
    resolved = resolve_locale_code(lang)
    return _CURRENT_LANG.set(resolved)


def reset_current_lang(token: contextvars.Token) -> None:
    _CURRENT_LANG.reset(token)


def _file_sig(path: str) -> tuple[int, int] | None:
    try:
        st = os.stat(path)
        return (st.st_mtime_ns, st.st_size)
    except OSError:
        return None


def _load_json_cached(path: str) -> dict:
    sig = _file_sig(path)
    if sig is None:
        return {}
    with _LOCK:
        hit = _FILE_CACHE.get(path)
        if hit is not None and hit[0] == sig:
            return hit[1]
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, dict):
            data = {}
    except Exception:
        data = {}
    with _LOCK:
        _FILE_CACHE[path] = (sig, data)
    return data


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def list_locales() -> list[dict]:
    """Auto-discover all locale JSON files in `frontend/locales/` sorted by `_meta.order`."""
    if not os.path.isdir(LOCALES_DIR):
        return [
            {
                "code": DEFAULT_LOCALE,
                "label": DEFAULT_LOCALE,
                "short": DEFAULT_LOCALE.upper(),
                "htmlLang": DEFAULT_LOCALE,
                "order": 10,
            }
        ]
    items = []
    for fn in sorted(os.listdir(LOCALES_DIR)):
        if not fn.endswith(".json") or fn.startswith("."):
            continue
        fpath = os.path.join(LOCALES_DIR, fn)
        if not os.path.isfile(fpath):
            continue
        code_stem = fn[:-5]
        data = _load_json_cached(fpath)
        meta = data.get("_meta") if isinstance(data.get("_meta"), dict) else {}
        code = str(meta.get("code") or code_stem)
        items.append(
            {
                "code": code,
                "file": fn,
                "label": str(meta.get("label") or code),
                "short": str(meta.get("short") or code.upper()),
                "htmlLang": str(meta.get("htmlLang") or code),
                "order": int(meta.get("order", 100)),
            }
        )
    items.sort(key=lambda x: (x["order"], x["code"]))
    return [{k: v for k, v in it.items() if k != "file"} for it in items]


def resolve_locale_code(raw: str | None) -> str:
    """Resolve an arbitrary language header/tag to an installed locale code (defaulting to DEFAULT_LOCALE)."""
    catalog = list_locales()
    if not catalog:
        return DEFAULT_LOCALE
    codes = [c["code"] for c in catalog]
    if not raw:
        return DEFAULT_LOCALE if DEFAULT_LOCALE in codes else codes[0]
    cleaned = raw.strip().split(",")[0].split(";")[0].strip()
    if not cleaned:
        return DEFAULT_LOCALE if DEFAULT_LOCALE in codes else codes[0]
    lower_map = {c.lower(): c for c in codes}
    if cleaned.lower() in lower_map:
        return lower_map[cleaned.lower()]
    prefix = cleaned.lower().split("-")[0].split("_")[0]
    for c in codes:
        if c.lower().split("-")[0].split("_")[0] == prefix:
            return c
    return DEFAULT_LOCALE if DEFAULT_LOCALE in codes else codes[0]


def _single_locale_layers(code: str) -> list[str]:
    paths = [os.path.join(LOCALES_DIR, f"{code}.json")]
    priv = os.path.join(PRIVATE_LOCALES_DIR, f"{code}.json")
    if os.path.isfile(priv):
        paths.append(priv)
    return paths


def get_locale_bundle(raw_code: str | None = None) -> dict:
    """Return the deep-merged translation bundle for `raw_code` with per-key fallback."""
    code = resolve_locale_code(raw_code)
    chain_codes = [DEFAULT_LOCALE]
    if code != DEFAULT_LOCALE:
        if code != "en" and os.path.isfile(os.path.join(LOCALES_DIR, "en.json")):
            chain_codes.append("en")
        chain_codes.append(code)

    layer_paths = []
    for c in chain_codes:
        for p in _single_locale_layers(c):
            if p not in layer_paths:
                layer_paths.append(p)

    bundle_sig = tuple((p, _file_sig(p)) for p in layer_paths)
    with _LOCK:
        hit = _BUNDLE_CACHE.get(code)
        if hit is not None and hit[0] == bundle_sig:
            return hit[1]

    merged: dict = {}
    for p in layer_paths:
        merged = _deep_merge(merged, _load_json_cached(p))
    # Ensure `_meta` reflects the active target language rather than a fallback layer
    target_main = _load_json_cached(os.path.join(LOCALES_DIR, f"{code}.json"))
    if isinstance(target_main.get("_meta"), dict):
        merged["_meta"] = dict(target_main["_meta"])

    with _LOCK:
        _BUNDLE_CACHE[code] = (bundle_sig, merged)
    return merged


def t(key: str, lang: str | None = None, **kwargs) -> str:
    """Translate dot-separated `key` in the current request's language with `{var}` interpolation."""
    bundle = get_locale_bundle(lang or get_current_lang())
    cur = bundle
    for part in (key or "").split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return key
    if not isinstance(cur, str):
        return key
    if not kwargs:
        return cur
    return _PLACEHOLDER_RE.sub(lambda m: str(kwargs[m.group(1)]) if m.group(1) in kwargs else m.group(0), cur)
