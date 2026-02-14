import os
import re
import json
import time
import hashlib
from typing import Any, Dict, List, Optional, Tuple

import requests
import pandas as pd
import streamlit as st

try:
    from google import genai
except Exception:
    genai = None


# -----------------------------
# Constants / Config
# -----------------------------
S2_BASE = "https://api.semanticscholar.org/graph/v1"
S2_FIELDS_BASE = [
    "paperId",
    "title",
    "abstract",
    "year",
    "authors",
    "venue",
    "citationCount",
    "influentialCitationCount",
    "externalIds",
    "url",
    "fieldsOfStudy",
    "publicationTypes",
    "openAccessPdf",
]
S2_FIELDS_DETAILS = S2_FIELDS_BASE + ["tldr"]
S2_RECO_BASE = "https://api.semanticscholar.org/recommendations/v1"
FINAL_STATUSES = {"include", "maybe", "exclude"}
DEFAULT_LIMIT = 25
MAX_LIMIT = 100  # S2 will cap; still keep reasonable
USER_AGENT = "lit-review-streamlit/1.0"


# -----------------------------
# Utilities
# -----------------------------
def now_ms() -> int:
    return int(time.time() * 1000)


def safe_get(d: Dict[str, Any], path: List[str], default=None):
    cur = d
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


def normalize_title(t: str) -> str:
    t = (t or "").strip().lower()
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"[^a-z0-9\s]", "", t)
    return t


def short(s: str, n: int = 240) -> str:
    s = s or ""
    return s if len(s) <= n else s[: n - 1] + "…"


def authors_to_str(authors: List[Dict[str, Any]]) -> str:
    if not authors:
        return ""
    names = [a.get("name", "").strip() for a in authors if a.get("name")]
    return ", ".join(names[:8]) + ("" if len(names) <= 8 else " et al.")


def get_doi(external_ids: Dict[str, Any]) -> Optional[str]:
    if not external_ids:
        return None
    # S2 can return DOI in externalIds
    return external_ids.get("DOI") or external_ids.get("doi")


def paper_key(p: Dict[str, Any]) -> str:
    # Prefer stable unique identifiers
    pid = p.get("paperId")
    doi = get_doi(p.get("externalIds") or {}) or ""
    title = normalize_title(p.get("title") or "")
    base = pid or doi or title
    return hashlib.sha1(base.encode("utf-8")).hexdigest()


def json_download_button(label: str, obj: Any, filename: str):
    data = json.dumps(obj, indent=2, ensure_ascii=False).encode("utf-8")
    st.download_button(label, data, file_name=filename, mime="application/json")


def status_badge(k: str) -> str:
    d = st.session_state.decisions.get(k)
    if not d or not d.get("status"):
        return "🟦 Unscreened"
    s = d["status"]
    return {"include": "🟩 Included", "maybe": "🟨 Maybe", "exclude": "🟥 Excluded"}.get(s, "🟦 Unscreened")

def parse_year_range(s: str) -> Optional[Tuple[int, int]]:
    s = (s or "").strip()
    if not s:
        return None
    m = re.match(r"^\s*(\d{4})\s*-\s*(\d{4})\s*$", s)
    if m:
        return int(m.group(1)), int(m.group(2))
    if s.isdigit():
        y = int(s)
        return y, y
    return None


def record_snowballed(new_keys: List[str], snow_type: str, from_pid: str, from_title: str):
    ts = now_ms()
    for k in new_keys:
        if k not in st.session_state.snowballed["meta"]:
            st.session_state.snowballed["keys"].append(k)
            st.session_state.snowballed["meta"][k] = {
                "type": snow_type,               # refs | cits | sim
                "from_paperId": from_pid,
                "from_title": from_title,
                "ts_ms": ts,
            }
    st.session_state.snowballed["runs"].append({
        "ts_ms": ts,
        "type": snow_type,
        "from_paperId": from_pid,
        "from_title": from_title,
        "added": len(new_keys),
    })


def ensure_decision(k: str) -> Dict[str, Any]:
    if k not in st.session_state.decisions:
        st.session_state.decisions[k] = {"status": None, "reason": "", "notes": "", "tags": []}
    return st.session_state.decisions[k]


def set_status(k: str, status: str):
    d = ensure_decision(k)
    d["status"] = status
    st.session_state.decisions[k] = d


def is_screened(k: str) -> bool:
    d = st.session_state.decisions.get(k, {})
    return d.get("status") in FINAL_STATUSES


def is_unscreened(k: str) -> bool:
    return not is_screened(k)


def should_keep_paper(p: Dict[str, Any]) -> bool:
    """
    Applies the SAME constraints used by the app:
    - If AI mode ON: use st.session_state.scope (year_range, must/exclude keywords)
    - Always enforce citation/open-access constraints if present in a unified session_state key
      (you can keep using your snowball_filters too, or migrate everything into one)
    """
    # ---- 1) "Global" constraints (optional but recommended) ----
    # If you keep snowball_filters from before:
    sb = st.session_state.get("snowball_filters", {})
    min_cites = int(sb.get("min_cites") or 0)
    open_only = bool(sb.get("open_only"))
    yr_range_manual = sb.get("year_range") or ""

    if int(p.get("citationCount") or 0) < min_cites:
        return False
    if open_only and not safe_get(p, ["openAccessPdf", "url"], None):
        return False

    # ---- 2) Year constraint: prefer scope.year_range in AI mode, else manual year_range ----
    year_val = int(p.get("year") or 0)
    if st.session_state.get("ai_on"):
        yr_range = (st.session_state.get("scope", {}).get("year_range") or "").strip()
    else:
        yr_range = (yr_range_manual or "").strip()

    yr = parse_year_range(yr_range)
    if yr and year_val:
        lo, hi = yr
        if not (lo <= year_val <= hi):
            return False

    # ---- 3) Keyword constraints from scope (AI mode only; optional) ----
    # Snowball results don't include full text; we can only do title/abstract keyword filtering.
    if st.session_state.get("ai_on"):
        scope = st.session_state.get("scope", {})
        must = [x.lower() for x in (scope.get("must_include") or [])]
        exc = [x.lower() for x in (scope.get("exclude") or [])]

        hay = " ".join([
            (p.get("title") or ""),
            (p.get("abstract") or ""),
        ]).lower()

        # Must include: require at least ONE must term present (if any are defined)
        if must:
            if not any(m in hay for m in must):
                return False

        # Exclude: reject if ANY exclude term appears
        if exc:
            if any(x in hay for x in exc):
                return False

    return True

@st.cache_data(show_spinner=False, ttl=60 * 20)
def s2_search_cached(api_key: str, timeout: int, params_key: str) -> Dict[str, Any]:
    params = json.loads(params_key)
    headers = {"User-Agent": USER_AGENT}
    if api_key:
        headers["x-api-key"] = api_key

    url = f"{S2_BASE}/paper/search"
    r = requests.get(url, headers=headers, params=params, timeout=timeout)
    if r.status_code >= 400:
        raise RuntimeError(f"S2 error {r.status_code}: {r.text[:400]}")
    return r.json()


@st.cache_data(show_spinner=False, ttl=60 * 60)
def s2_paper_details_cached(api_key: str, timeout: int, paper_id: str, fields_key: str) -> Dict[str, Any]:
    headers = {"User-Agent": USER_AGENT}
    if api_key:
        headers["x-api-key"] = api_key

    url = f"{S2_BASE}/paper/{paper_id}"
    params = {"fields": fields_key}
    r = requests.get(url, headers=headers, params=params, timeout=timeout)
    if r.status_code >= 400:
        raise RuntimeError(f"S2 error {r.status_code}: {r.text[:400]}")
    return r.json()


@st.cache_data(show_spinner=False, ttl=60 * 60)
def s2_references_cached(api_key: str, timeout: int, paper_id: str, limit: int, fields_key: str) -> Dict[str, Any]:
    headers = {"User-Agent": USER_AGENT}
    if api_key:
        headers["x-api-key"] = api_key

    url = f"{S2_BASE}/paper/{paper_id}/references"
    params = {"fields": fields_key, "limit": min(int(limit), 500)}
    r = requests.get(url, headers=headers, params=params, timeout=timeout)
    if r.status_code >= 400:
        raise RuntimeError(f"S2 error {r.status_code}: {r.text[:400]}")
    return r.json()


@st.cache_data(show_spinner=False, ttl=60 * 60)
def s2_citations_cached(api_key: str, timeout: int, paper_id: str, limit: int, fields_key: str) -> Dict[str, Any]:
    headers = {"User-Agent": USER_AGENT}
    if api_key:
        headers["x-api-key"] = api_key

    url = f"{S2_BASE}/paper/{paper_id}/citations"
    params = {"fields": fields_key, "limit": min(int(limit), 500)}
    r = requests.get(url, headers=headers, params=params, timeout=timeout)
    if r.status_code >= 400:
        raise RuntimeError(f"S2 error {r.status_code}: {r.text[:400]}")
    return r.json()


@st.cache_data(show_spinner=False, ttl=60 * 60)
def s2_recommended_cached(api_key: str, timeout: int, paper_id: str, limit: int, fields_key: str) -> Dict[str, Any]:
    headers = {"User-Agent": USER_AGENT}
    if api_key:
        headers["x-api-key"] = api_key

    url = f"{S2_BASE}/paper/{paper_id}/recommended"
    params = {"fields": fields_key, "limit": min(int(limit), 500)}
    r = requests.get(url, headers=headers, params=params, timeout=timeout)
    if r.status_code >= 400:
        raise RuntimeError(f"S2 error {r.status_code}: {r.text[:400]}")
    return r.json()

@st.cache_data(show_spinner=False, ttl=60 * 60)
def s2_recommended_v1_cached(api_key: str, timeout: int, paper_id: str, limit: int) -> Dict[str, Any]:
    headers = {"User-Agent": USER_AGENT}
    if api_key:
        headers["x-api-key"] = api_key

    url = f"{S2_RECO_BASE}/papers/forpaper/{paper_id}"
    params = {"limit": min(int(limit), 500)}
    r = requests.get(url, headers=headers, params=params, timeout=timeout)
    if r.status_code >= 400:
        raise RuntimeError(f"S2 reco error {r.status_code}: {r.text[:400]}")
    return r.json()

# -----------------------------
# Semantic Scholar Client
# -----------------------------
class SemanticScholarClient:
    def __init__(self, api_key: Optional[str] = None, timeout: int = 30):
        self.api_key = api_key or ""
        self.timeout = int(timeout)

    def _headers(self) -> Dict[str, str]:
        h = {"User-Agent": USER_AGENT}
        if self.api_key:
            h["x-api-key"] = self.api_key
        return h

    def _get(self, url: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        r = requests.get(url, headers=self._headers(), params=params, timeout=self.timeout)
        if r.status_code >= 400:
            raise RuntimeError(f"S2 error {r.status_code}: {r.text[:400]}")
        return r.json()

    def search_papers(
        self,
        query: str,
        limit: int = DEFAULT_LIMIT,
        offset: int = 0,
        fields: Optional[List[str]] = None,
        year: Optional[str] = None,
        venue: Optional[str] = None,
        fields_of_study: Optional[str] = None,
        publication_types: Optional[str] = None,
        min_citations: Optional[int] = None,
        open_access_only: bool = False,
        sort: str = "relevance",
    ) -> Dict[str, Any]:
        fields = fields or S2_FIELDS_BASE
        limit = max(1, min(int(limit), MAX_LIMIT))
        offset = max(0, int(offset))

        params: Dict[str, Any] = {
            "query": query,
            "limit": limit,
            "offset": offset,
            "fields": ",".join(fields),
        }
        if year:
            params["year"] = year

        params_key = json.dumps(params, sort_keys=True)
        data = s2_search_cached(self.api_key, self.timeout, params_key)

        papers = data.get("data", []) or []

        # Client-side filters
        if venue:
            v = venue.strip().lower()
            papers = [p for p in papers if (p.get("venue") or "").strip().lower() == v]
        if fields_of_study:
            fos = fields_of_study.strip().lower()
            papers = [
                p for p in papers
                if any((f or "").strip().lower() == fos for f in (p.get("fieldsOfStudy") or []))
            ]
        if publication_types:
            pt = publication_types.strip().lower()
            papers = [
                p for p in papers
                if any((t or "").strip().lower() == pt for t in (p.get("publicationTypes") or []))
            ]
        if min_citations is not None:
            papers = [p for p in papers if int(p.get("citationCount") or 0) >= int(min_citations)]
        if open_access_only:
            papers = [p for p in papers if safe_get(p, ["openAccessPdf", "url"], None)]

        # Local sort
        if sort == "citations":
            papers.sort(key=lambda x: int(x.get("citationCount") or 0), reverse=True)
        elif sort == "year":
            papers.sort(key=lambda x: int(x.get("year") or 0), reverse=True)

        data["data"] = papers
        return data

    def paper_details(self, paper_id: str, fields: Optional[List[str]] = None) -> Dict[str, Any]:
        fields = fields or S2_FIELDS_DETAILS
        fields_key = ",".join(fields)
        try:
            return s2_paper_details_cached(self.api_key, self.timeout, paper_id, fields_key)
        except RuntimeError:
            # fallback in case 'tldr' isn't supported for your account/region
            return s2_paper_details_cached(self.api_key, self.timeout, paper_id, ",".join(S2_FIELDS_BASE))


    def references(self, paper_id: str, limit: int = 100) -> Dict[str, Any]:
        fields_key = ",".join(S2_FIELDS_BASE)
        return s2_references_cached(self.api_key, self.timeout, paper_id, int(limit), fields_key)


    def citations(self, paper_id: str, limit: int = 100) -> Dict[str, Any]:
        fields_key = ",".join(S2_FIELDS_BASE)
        return s2_citations_cached(self.api_key, self.timeout, paper_id, int(limit), fields_key)


    def recommended(self, paper_id: str, limit: int = 50) -> Dict[str, Any]:
        """
        Uses Semantic Scholar Recommendations API (separate service).
        Returns dict with a 'data' list of paper objects (normalized to match our expectations).
        """
        reco = s2_recommended_v1_cached(self.api_key, self.timeout, paper_id, int(limit))

        # The reco API returns something like: {"recommendedPapers":[{"paperId":...}, ...]}
        recs = reco.get("recommendedPapers") or reco.get("data") or []

        # If it only returns IDs, we can fetch details in batch-like manner (simple loop).
        # We'll try to treat each entry as a paper object; if missing title, pull details.
        out_papers = []
        for item in recs:
            if isinstance(item, dict) and item.get("title"):
                out_papers.append(item)
            else:
                pid = item.get("paperId") if isinstance(item, dict) else None
                if pid:
                    try:
                        out_papers.append(self.paper_details(pid, fields=S2_FIELDS_BASE))
                    except Exception:
                        continue

        return {"data": out_papers}


# -----------------------------
# Gemini Helper
# -----------------------------
class GeminiHelper:
    def __init__(self, api_key: str, model: str):
        if not genai:
            raise RuntimeError("google-genai is not installed. `pip install google-genai`")
        self.client = genai.Client(api_key=api_key)
        self.model = model

    def list_models_text(self) -> str:
        # Best-effort; depends on SDK response structure
        try:
            models = self.client.models.list()
            names = []
            for m in models:
                name = getattr(m, "name", None) or str(m)
                names.append(name)
            return "\n".join(names[:200])
        except Exception as e:
            return f"Could not list models: {e}"

    def generate_json(self, system: str, user: str, schema_hint: str, temperature: float = 0.2) -> Dict[str, Any]:
        """
        Enforces JSON output by instruction. (SDK has JSON mode in newer docs, but we do prompt-enforced JSON.)
        """
        prompt = (
            f"{system}\n\n"
            f"Return ONLY valid JSON, no markdown, no extra text.\n"
            f"JSON schema (hint): {schema_hint}\n\n"
            f"USER:\n{user}\n"
        )
        resp = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
        )
        text = (getattr(resp, "text", None) or "").strip()
        # Robust parse
        try:
            return json.loads(text)
        except Exception:
            # Attempt to extract a JSON object substring
            m = re.search(r"\{.*\}", text, flags=re.DOTALL)
            if m:
                return json.loads(m.group(0))
            raise RuntimeError(f"Gemini returned non-JSON:\n{text[:800]}")

    def generate_text(self, system: str, user: str) -> str:
        prompt = f"{system}\n\nUSER:\n{user}\n"
        resp = self.client.models.generate_content(model=self.model, contents=prompt)
        return (getattr(resp, "text", None) or "").strip()


# -----------------------------
# State
# -----------------------------
def init_state():
    if "user_s2_key" not in st.session_state:
        st.session_state.user_s2_key = ""
    if "user_gem_key" not in st.session_state:
        st.session_state.user_gem_key = ""
    if "_loaded_session_once" not in st.session_state:
        st.session_state._loaded_session_once = False
    if "ai_on" not in st.session_state:
        st.session_state.ai_on = False
    if "papers" not in st.session_state:
        st.session_state.papers = {}  # key -> paper dict (merged/deduped)
    if "decisions" not in st.session_state:
        st.session_state.decisions = {}  # key -> {status, reason, notes, tags}
    if "search_runs" not in st.session_state:
        st.session_state.search_runs = []  # list of search run metadata
    if "gemini_meta" not in st.session_state:
        st.session_state.gemini_meta = {}  # key -> triage metadata (score, summary, cluster, etc.)
    if "snowballed" not in st.session_state:
        st.session_state.snowballed = {
            "keys": [],          # ordered list of paperKeys that were added via snowballing
            "meta": {},          # paperKey -> {type, from_paperId, from_title, ts_ms}
            "runs": [],          # log of snowball actions
        }
    if "clusters" not in st.session_state:
        st.session_state.clusters = {}  # cluster -> list of keys
    if "scope" not in st.session_state:
        st.session_state.scope = {
            "research_question": "",
            "must_include": [],
            "exclude": [],
            "year_range": "",
            "rubric": "",
            "notes": "",
        }


def upsert_papers(papers: List[Dict[str, Any]]) -> Tuple[int, int, List[str]]:
    added = 0
    updated = 0
    newly_added_keys = []

    for p in papers:
        k = paper_key(p)
        if k not in st.session_state.papers:
            st.session_state.papers[k] = p
            added += 1
            newly_added_keys.append(k)
        else:
            cur = st.session_state.papers[k]
            for field in ["abstract", "tldr", "openAccessPdf", "externalIds", "url"]:
                if not cur.get(field) and p.get(field):
                    cur[field] = p.get(field)
            cur["citationCount"] = max(int(cur.get("citationCount") or 0), int(p.get("citationCount") or 0))
            st.session_state.papers[k] = cur
            updated += 1

    return added, updated, newly_added_keys


def prisma_counts() -> Dict[str, int]:
    total = len(st.session_state.papers)
    decisions = st.session_state.decisions

    def has_final_status(v: Dict[str, Any]) -> bool:
        return v.get("status") in {"include", "maybe", "exclude"}

    screened = sum(1 for v in decisions.values() if has_final_status(v))
    include = sum(1 for v in decisions.values() if v.get("status") == "include")
    maybe = sum(1 for v in decisions.values() if v.get("status") == "maybe")
    exclude = sum(1 for v in decisions.values() if v.get("status") == "exclude")

    unscreened = total - screened
    return {
        "retrieved": total,
        "screened": screened,
        "unscreened": unscreened,
        "included": include,
        "maybe": maybe,
        "excluded": exclude,
    }


# -----------------------------
# Export
# -----------------------------
def export_df(keys: List[str]) -> pd.DataFrame:
    rows = []
    for k in keys:
        p = st.session_state.papers.get(k, {})
        d = st.session_state.decisions.get(k, {})
        g = st.session_state.gemini_meta.get(k, {})
        ext = p.get("externalIds") or {}
        rows.append(
            {
                "paperKey": k,
                "paperId": p.get("paperId"),
                "title": p.get("title"),
                "year": p.get("year"),
                "venue": p.get("venue"),
                "authors": authors_to_str(p.get("authors") or []),
                "citations": p.get("citationCount"),
                "doi": get_doi(ext),
                "url": p.get("url"),
                "openAccessPdf": safe_get(p, ["openAccessPdf", "url"], None),
                "tldr": safe_get(p, ["tldr", "text"], None),
                "abstract": p.get("abstract"),
                "fieldsOfStudy": ", ".join(p.get("fieldsOfStudy") or []),
                "publicationTypes": ", ".join(p.get("publicationTypes") or []),
                "status": d.get("status"),
                "reason": d.get("reason"),
                "notes": d.get("notes"),
                "tags": ", ".join(d.get("tags") or []),
                # Gemini V2 columns
                "ai_relevance_score": g.get("relevance_score"),
                "ai_decision": g.get("decision"),
                "ai_why": g.get("why"),
                "ai_summary": g.get("summary"),
                "ai_cluster": g.get("cluster"),
                "ai_flags": ", ".join(g.get("flags") or []),
            }
        )
    return pd.DataFrame(rows)


def export_bibtex(df: pd.DataFrame) -> str:
    # Simple BibTeX; good enough for MVP/V2.
    # Key: first author + year + first meaningful word from title
    entries = []
    for _, r in df.iterrows():
        title = (r.get("title") or "").strip()
        year = str(r.get("year") or "")
        authors = (r.get("authors") or "").strip()
        venue = (r.get("venue") or "").strip()
        doi = (r.get("doi") or "").strip()
        url = (r.get("url") or "").strip()

        first_author = (authors.split(",")[0] if authors else "unknown").split()[-1].lower()
        first_word = re.sub(r"[^a-zA-Z0-9]", "", (title.split()[:1] or ["paper"])[0]).lower() or "paper"
        key = f"{first_author}{year}{first_word}"

        def bib_escape(s: str) -> str:
            return s.replace("{", "\\{").replace("}", "\\}").replace("\n", " ").strip()

        entry = f"@article{{{key},\n"
        entry += f"  title={{ {bib_escape(title)} }},\n"
        if authors:
            entry += f"  author={{ {bib_escape(authors)} }},\n"
        if venue:
            entry += f"  journal={{ {bib_escape(venue)} }},\n"
        if year:
            entry += f"  year={{ {bib_escape(year)} }},\n"
        if doi:
            entry += f"  doi={{ {bib_escape(doi)} }},\n"
        if url:
            entry += f"  url={{ {bib_escape(url)} }},\n"
        entry += "}\n"
        entries.append(entry)
    return "\n".join(entries)


def export_ris(df: pd.DataFrame) -> str:
    # Minimal RIS
    lines = []
    for _, r in df.iterrows():
        lines.append("TY  - JOUR")
        if r.get("title"):
            lines.append(f"TI  - {r['title']}")
        if r.get("authors"):
            for a in str(r["authors"]).split(","):
                a = a.strip()
                if a:
                    lines.append(f"AU  - {a}")
        if r.get("year"):
            lines.append(f"PY  - {r['year']}")
        if r.get("venue"):
            lines.append(f"JO  - {r['venue']}")
        if r.get("doi"):
            lines.append(f"DO  - {r['doi']}")
        if r.get("url"):
            lines.append(f"UR  - {r['url']}")
        lines.append("ER  - ")
        lines.append("")
    return "\n".join(lines)


# -----------------------------
# AI V2: prompts
# -----------------------------
def gemini_search_plan(g: GeminiHelper, scope: Dict[str, Any], n_queries: int = 8) -> Dict[str, Any]:
    system = (
        "You are a research assistant that designs search strategies for literature reviews.\n"
        "Optimize for recall + precision. Keep queries distinct.\n"
        "You must not fabricate citations; you only propose search queries."
    )
    user = (
        f"Research question:\n{scope.get('research_question','')}\n\n"
        f"Must include concepts/keywords:\n{scope.get('must_include', [])}\n\n"
        f"Exclude concepts/keywords:\n{scope.get('exclude', [])}\n\n"
        f"Year range constraint (if any):\n{scope.get('year_range','')}\n\n"
        f"Rubric / inclusion criteria:\n{scope.get('rubric','')}\n\n"
        f"Return {n_queries} queries and a few synonyms per major concept."
    )
    schema = """{
      "queries":[{"query":"string","why":"string","must_include":["string"],"exclude":["string"]}],
      "synonyms":{"concept":"[syn1,syn2]"},
      "recommended_filters":{"year_range":"string","fields_of_study":["string"],"publication_types":["string"]}
    }"""
    return g.generate_json(system, user, schema_hint=schema)


def gemini_triage_batch(g: GeminiHelper, scope: Dict[str, Any], papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    system = (
        "You help screen papers for a literature review.\n"
        "Decide based only on title+abstract+metadata provided.\n"
        "Be conservative: prefer 'maybe' when uncertain.\n"
        "Return JSON only."
    )
    compact = []
    for p in papers:
        compact.append(
            {
                "paperId": p.get("paperId"),
                "title": p.get("title"),
                "year": p.get("year"),
                "venue": p.get("venue"),
                "citationCount": p.get("citationCount"),
                "tldr": safe_get(p, ["tldr", "text"], None),
                "abstract": p.get("abstract"),
                "fieldsOfStudy": p.get("fieldsOfStudy"),
            }
        )

    user = (
        f"Research question:\n{scope.get('research_question','')}\n\n"
        f"Rubric / inclusion criteria:\n{scope.get('rubric','')}\n\n"
        f"Must include:\n{scope.get('must_include', [])}\n\n"
        f"Exclude:\n{scope.get('exclude', [])}\n\n"
        "PAPERS:\n"
        + json.dumps(compact, ensure_ascii=False)
        + "\n\nFor each paper, provide: relevance_score(0-100), decision(include/maybe/exclude), why (1-2 sentences), "
          "summary (1-2 sentences), flags (like survey/dataset/methods/application), tags (<=5), cluster label."
    )

    schema = """[
      {"paperId":"string","relevance_score":0,"decision":"include|maybe|exclude",
       "why":"string","summary":"string","flags":["string"],"tags":["string"],"cluster":"string"}
    ]"""
    out = g.generate_json(system, user, schema_hint=schema)
    if not isinstance(out, list):
        raise RuntimeError("Gemini triage did not return a JSON list.")
    return out


def gemini_gap_and_outline(g: GeminiHelper, scope: Dict[str, Any], included: List[Dict[str, Any]]) -> Dict[str, Any]:
    system = (
        "You help synthesize a literature review plan.\n"
        "Use only the provided shortlisted papers (title, year, venue, tldr/abstract). "
        "Do not invent citations. If something is missing, say so."
    )
    compact = []
    for p in included:
        compact.append(
            {
                "paperId": p.get("paperId"),
                "title": p.get("title"),
                "year": p.get("year"),
                "venue": p.get("venue"),
                "tldr": safe_get(p, ["tldr", "text"], None),
                "abstract": p.get("abstract"),
            }
        )
    user = (
        f"Research question:\n{scope.get('research_question','')}\n\n"
        f"Rubric:\n{scope.get('rubric','')}\n\n"
        "SHORTLIST:\n"
        + json.dumps(compact, ensure_ascii=False)
        + "\n\nReturn: (1) gaps_to_search (bullets), (2) recommended_next_queries (5), "
          "(3) a lit_review_outline with sections and which paperIds map to each section."
    )
    schema = """{
      "gaps_to_search":["string"],
      "recommended_next_queries":["string"],
      "lit_review_outline":[{"section":"string","notes":"string","paperIds":["string"]}]
    }"""
    return g.generate_json(system, user, schema_hint=schema)


# -----------------------------
# UI
# -----------------------------
def header_bar():
    c1, c2 = st.columns([0.78, 0.22])
    with c1:
        st.title("LitReview Finder")
        st.caption("Search → Screen → Snowball → Export. Toggle AI on/off (top right).")
    with c2:
        # "Top right" approximation: it sits on the same row as the title area.
        st.session_state.ai_on = st.toggle("AI ON", value=st.session_state.ai_on, help="Switch between Manual and Gemini-assisted mode.")


def get_secret(name: str) -> str:
    if hasattr(st, "secrets") and name in st.secrets:
        return str(st.secrets.get(name) or "")
    return str(os.getenv(name) or "")


def sidebar_config() -> Tuple[SemanticScholarClient, Optional[GeminiHelper]]:
    # defaults from secrets/env
    default_s2 = get_secret("SEMANTIC_SCHOLAR_API_KEY")
    default_gem = get_secret("GEMINI_API_KEY")
    model = get_secret("GEMINI_MODEL") or "gemini-3-flash-preview"

    # optional user override (if you implement the expander)
    user_s2 = (st.session_state.get("user_s2_key") or "").strip()
    user_gem = (st.session_state.get("user_gem_key") or "").strip()

    s2_key = user_s2 or default_s2
    gem_key = user_gem or default_gem

    s2 = SemanticScholarClient(api_key=s2_key)

    gem = None
    if st.session_state.ai_on:
        if gem_key:
            gem = GeminiHelper(api_key=gem_key, model=model)
        else:
            st.warning("AI mode is ON but Gemini key is not configured.")
    return s2, gem


def tab_prisma():
    counts = prisma_counts()
    c = st.columns(5)
    c[0].metric("Retrieved", counts["retrieved"])
    c[1].metric("Screened", counts["screened"])
    c[2].metric("Unscreened", counts["unscreened"])
    c[3].metric("Included", counts["included"])
    c[4].metric("Maybe", counts["maybe"])


def decision_controls(k: str, key_prefix: str = ""):
    prefix = f"{key_prefix}__" if key_prefix else ""

    d = ensure_decision(k)

    cols = st.columns([0.16, 0.16, 0.16, 0.52])

    with cols[0]:
        st.button("✅ Include", key=f"{prefix}inc_{k}", on_click=set_status, args=(k, "include"))
    with cols[1]:
        st.button("🤔 Maybe", key=f"{prefix}may_{k}", on_click=set_status, args=(k, "maybe"))
    with cols[2]:
        st.button("❌ Exclude", key=f"{prefix}exc_{k}", on_click=set_status, args=(k, "exclude"))

    with cols[3]:
        options = ["", "Out of scope", "Wrong population", "Wrong outcome", "Not peer-reviewed",
                   "Too old", "Insufficient detail", "Other"]
        current_reason = d.get("reason") or ""
        idx = options.index(current_reason) if current_reason in options else 0

        reason = st.selectbox("Reason (optional)", options=options, index=idx, key=f"{prefix}reason_{k}")
        if reason != d.get("reason"):
            d["reason"] = reason

    notes = st.text_area("Notes", value=d.get("notes", ""), key=f"{prefix}notes_{k}", height=80)
    if notes != d.get("notes"):
        d["notes"] = notes

    tag_str = st.text_input("Tags (comma-separated)", value=", ".join(d.get("tags") or []), key=f"{prefix}tags_{k}")
    new_tags = [t.strip() for t in tag_str.split(",") if t.strip()]
    if new_tags != (d.get("tags") or []):
        d["tags"] = new_tags

    st.session_state.decisions[k] = d


def render_paper_card(k: str, p: Dict[str, Any], show_ai: bool, key_prefix: str = ""):
    title = p.get("title") or "(untitled)"
    st.markdown(f"**{status_badge(k)}**")
    year = p.get("year") or ""
    venue = p.get("venue") or ""
    cites = p.get("citationCount") or 0
    auth = authors_to_str(p.get("authors") or [])
    tldr = safe_get(p, ["tldr", "text"], None)
    abstract = p.get("abstract") or ""
    url = p.get("url") or ""
    oa_pdf = safe_get(p, ["openAccessPdf", "url"], None)

    st.markdown(f"### {title}")
    st.caption(f"{auth} • {venue} • {year} • citations: {cites}")
    link_line = []
    if url:
        link_line.append(f"[Semantic Scholar]({url})")
    if oa_pdf:
        link_line.append(f"[Open PDF]({oa_pdf})")
    if link_line:
        st.markdown(" • ".join(link_line))

    if show_ai:
        g = st.session_state.gemini_meta.get(k, {})
        if g:
            st.info(
                f"**AI triage:** score={g.get('relevance_score')} • decision={g.get('decision')} • cluster={g.get('cluster')}\n\n"
                f"**Why:** {g.get('why')}\n\n"
                f"**Summary:** {g.get('summary')}\n\n"
                f"**Flags:** {', '.join(g.get('flags') or [])} • **Tags:** {', '.join(g.get('tags') or [])}"
            )

    with st.expander("TLDR / Abstract"):
        if tldr:
            st.markdown(f"**TLDR:** {tldr}")
        st.write(abstract if abstract else "(No abstract)")
        prefix = f"{key_prefix}__" if key_prefix else ""
        if not tldr and p.get("paperId"):
            if st.button("Fetch details (may include TLDR)", key=f"{prefix}details_{k}"):
                try:
                    details = st.session_state._s2.paper_details(p["paperId"])
                    cur = st.session_state.papers[k]
                    for field in ["abstract", "tldr", "openAccessPdf", "externalIds", "url"]:
                        if details.get(field):
                            cur[field] = details.get(field)
                    st.session_state.papers[k] = cur
                    st.rerun()
                except Exception as e:
                    st.error(f"Details fetch error: {e}")

    with st.expander("Screening"):
        decision_controls(k, key_prefix=key_prefix)

    with st.expander("Snowball this paper (references / citations / similar)"):
        st.caption(
            "Snowballing adds new papers to your workspace.\n"
            "References refer to the papers this one cites\n"
            "Citations refer to the papers that cite this one\n"
            "Similar refers to recommendations / fallback title search\n"
            "New papers are added as **Unscreened** until you mark Include/Maybe/Exclude."
        )
        cols = st.columns(3)
        pid = p.get("paperId")
        if not pid:
            st.warning("No paperId available for snowballing.")
            return

        prefix = f"{key_prefix}__" if key_prefix else ""

        with cols[0]:
            if st.button("➕ Add References", key=f"{prefix}refs_{k}"):
                try:
                    data = st.session_state._s2.references(pid, limit=200)
                    refs = [x.get("citedPaper") for x in (data.get("data") or []) if x.get("citedPaper")]
                    refs = [pp for pp in refs if should_keep_paper(pp)]
                    a, u, new_keys = upsert_papers(refs)
                    record_snowballed(new_keys, "refs", pid, p.get("title") or "")
                    st.success(f"Added {a} references ({u} updated).")
                    st.rerun()
                except Exception as e:
                    st.error(f"References error: {e}")

        with cols[1]:
            if st.button("➕ Add Citations", key=f"{prefix}cits_{k}"):
                try:
                    data = st.session_state._s2.citations(pid, limit=200)
                    cits = [x.get("citingPaper") for x in (data.get("data") or []) if x.get("citingPaper")]
                    cits = [pp for pp in cits if should_keep_paper(pp)]
                    a, u, new_keys = upsert_papers(cits)
                    record_snowballed(new_keys, "cits", pid, p.get("title") or "")
                    st.success(f"Added {a} citations ({u} updated).")
                    st.rerun()
                except Exception as e:
                    st.error(f"Citations error: {e}")

        with cols[2]:
            if st.button("➕ Add Similar", key=f"{prefix}sim_{k}"):
                try:
                    data = st.session_state._s2.recommended(pid, limit=100)
                    recs = data.get("data") or []
                    recs = [pp for pp in recs if should_keep_paper(pp)]
                    a, u, new_keys = upsert_papers(recs)
                    record_snowballed(new_keys, "sim", pid, p.get("title") or "")
                    st.success(f"Added {a} similar papers ({u} updated).")
                    st.rerun()
                except Exception as e:
                    # Fallback: title-based search
                    try:
                        title = (p.get("title") or "").strip()
                        if not title:
                            raise RuntimeError("No title for fallback search.")
                        fallback = st.session_state._s2.search_papers(
                            query=title,
                            limit=25,
                            sort="relevance",
                        )
                        papers = fallback.get("data") or []
                        papers = [pp for pp in papers if pp.get("paperId") and pp.get("paperId") != pid]
                        papers = [pp for pp in papers if should_keep_paper(pp)]
                        a, u, new_keys = upsert_papers(papers)
                        st.warning(f"Recommendations API failed ({e}). Used title-search fallback: added {a} ({u} updated).")
                    except Exception as e2:
                        st.error(f"Similar/recommended not available: {e}\nFallback also failed: {e2}")

def tab_search_manual(s2: SemanticScholarClient):
    st.subheader("Search (Manual)")
    tab_prisma()

    with st.form("search_form"):
        q = st.text_area("Query", placeholder="e.g., retrieval augmented generation evaluation in HCI", height=90)
        cols = st.columns(4)
        with cols[0]:
            year_range = st.text_input("Year (optional)", placeholder="e.g., 2020-2026 or 2023")
        with cols[1]:
            sort = st.selectbox("Sort", ["relevance", "citations", "year"])
        with cols[2]:
            limit = st.slider("Limit", 10, 100, 25, 5)
        with cols[3]:
            min_cit = st.number_input("Min citations", min_value=0, value=0, step=10)

        cols2 = st.columns(4)
        with cols2[0]:
            open_access_only = st.checkbox("Open-access PDF only", value=False)
        with cols2[1]:
            venue = st.text_input("Venue (exact match)", placeholder="e.g., CHI")
        with cols2[2]:
            fos = st.text_input("Field of Study (exact)", placeholder="e.g., Computer Science")
        with cols2[3]:
            pub_type = st.text_input("Publication type (exact)", placeholder="e.g., JournalArticle")

        submitted = st.form_submit_button("Search")

    if submitted and q.strip():
        try:
            data = s2.search_papers(
                query=q.strip(),
                limit=int(limit),
                offset=0,
                year=year_range.strip() or None,
                venue=venue.strip() or None,
                fields_of_study=fos.strip() or None,
                publication_types=pub_type.strip() or None,
                min_citations=int(min_cit) if min_cit else None,
                open_access_only=open_access_only,
                sort=sort,
            )
            papers = data.get("data", []) or []
            a, u, new_keys = upsert_papers(papers)
            st.session_state.search_runs.append({"ts": now_ms(), "mode": "manual", "query": q.strip(), "count": len(papers)})
            st.success(f"Fetched {len(papers)} results. Added {a}, updated {u}.")
        except Exception as e:
            st.error(f"Search error: {e}")

    st.divider()
    st.subheader("Results workspace")
    keys = list(st.session_state.papers.keys())
    if not keys:
        st.info("No papers yet. Run a search or import a session JSON.")
        return

    # Quick filters
    fcols = st.columns(4)
    with fcols[0]:
        status_filter = st.selectbox("Status filter", ["All", "Unscreened", "Include", "Maybe", "Exclude"])
    with fcols[1]:
        text_filter = st.text_input("Title contains", "")
    with fcols[2]:
        year_min = st.number_input("Year >= ", min_value=0, value=0, step=1)
    with fcols[3]:
        min_c = st.number_input("Citations >= ", min_value=0, value=0, step=10)

    def passes(k: str) -> bool:
        p = st.session_state.papers[k]
        d = st.session_state.decisions.get(k, {})
        if status_filter == "Unscreened":
            d = st.session_state.decisions.get(k, {})
            if d.get("status") in {"include", "maybe", "exclude"}:
                return False
        if status_filter in ["Include", "Maybe", "Exclude"]:
            want = status_filter.lower()
            if d.get("status") != want:
                return False
        if text_filter.strip() and text_filter.strip().lower() not in (p.get("title") or "").lower():
            return False
        if year_min and int(p.get("year") or 0) < int(year_min):
            return False
        if min_c and int(p.get("citationCount") or 0) < int(min_c):
            return False
        return True

    filtered = [k for k in keys if passes(k)]
    st.caption(f"Showing {len(filtered)} / {len(keys)} papers")

    # Paginate
    per_page = st.selectbox("Per page", [5, 10, 20], index=1)
    page = st.number_input("Page", min_value=1, value=1, step=1)
    start = (page - 1) * per_page
    end = start + per_page
    for k in filtered[start:end]:
        render_paper_card(k, st.session_state.papers[k], show_ai=False, key_prefix="manual_search")
        st.divider()


def tab_scope_ai(g: GeminiHelper):
    st.subheader("Scope (AI Mode)")
    s = st.session_state.scope

    s["research_question"] = st.text_area("Research question / objective", value=s.get("research_question", ""), height=90)
    s["year_range"] = st.text_input("Year range constraint (optional)", value=s.get("year_range", ""), placeholder="e.g., 2019-2026")
    s["rubric"] = st.text_area(
        "Inclusion rubric / screening criteria",
        value=s.get("rubric", ""),
        height=120,
        placeholder="Example: include peer-reviewed studies on RAG evaluation in interactive systems; exclude pure engineering blogs; require methodology details...",
    )

    c = st.columns(2)
    with c[0]:
        must = st.text_area("Must-include concepts/keywords (one per line)", value="\n".join(s.get("must_include", [])), height=120)
        s["must_include"] = [x.strip() for x in must.splitlines() if x.strip()]
    with c[1]:
        exc = st.text_area("Exclude concepts/keywords (one per line)", value="\n".join(s.get("exclude", [])), height=120)
        s["exclude"] = [x.strip() for x in exc.splitlines() if x.strip()]

    s["notes"] = st.text_area("Notes (optional)", value=s.get("notes", ""), height=80)
    st.session_state.scope = s

    with st.expander("Expand to ask AI for a search plan that incorporates the above requirements"):
        n = st.slider("Number of queries", 3, 15, 8)
        if st.button("Generate plan with Gemini"):
            if not s["research_question"].strip():
                st.warning("Add a research question first.")
            else:
                try:
                    plan = gemini_search_plan(g, s, n_queries=n)
                    st.session_state.ai_plan = plan
                    st.success("Generated plan.")
                except Exception as e:
                    st.error(f"Plan error: {e}")

        plan = st.session_state.get("ai_plan")
        if plan:
            st.json(plan)


def tab_search_ai(s2: SemanticScholarClient, g: GeminiHelper):
    st.subheader("Search (Gemini-assisted)")
    tab_prisma()

    plan = st.session_state.get("ai_plan")
    if not plan:
        st.info("Go to **Scope** tab and generate a search plan first (or do a manual search).")
        return

    queries = plan.get("queries") or []
    if not queries:
        st.warning("Plan has no queries.")
        return

    st.write("Edit which queries to run:")
    selected_queries = []
    for i, qobj in enumerate(queries):
        q = qobj.get("query", "")
        if st.checkbox(f"{i+1}. {q}", value=True):
            selected_queries.append(q)

    cols = st.columns(4)
    with cols[0]:
        limit = st.slider("Limit per query", 10, 100, 25, 5, key="ai_limit")
    with cols[1]:
        sort = st.selectbox("Sort", ["relevance", "citations", "year"], key="ai_sort")
    with cols[2]:
        open_access_only = st.checkbox("Open-access PDF only", value=False, key="ai_oa")
    with cols[3]:
        min_cit = st.number_input("Min citations", min_value=0, value=0, step=10, key="ai_minc")

    year_range = (st.session_state.scope.get("year_range") or "").strip() or None

    if st.button("Run selected queries on Semantic Scholar"):
        all_papers = []
        for q in selected_queries:
            try:
                data = s2.search_papers(
                    query=q,
                    limit=int(limit),
                    year=year_range,
                    sort=sort,
                    min_citations=int(min_cit) if min_cit else None,
                    open_access_only=open_access_only,
                )
                papers = data.get("data", []) or []
                all_papers.extend(papers)
                st.session_state.search_runs.append({"ts": now_ms(), "mode": "ai_retrieval", "query": q, "count": len(papers)})
            except Exception as e:
                st.error(f"Query failed: {q}\n{e}")
        a, u, new_keys = upsert_papers(all_papers)
        st.success(f"Total retrieved (merged): {len(all_papers)}. Added {a}, updated {u}.")

    st.divider()
    st.subheader("AI triage (score/decision/cluster)")

    keys = list(st.session_state.papers.keys())
    if not keys:
        st.info("No papers yet. Run retrieval first.")
        return

    # Choose candidates to triage
    c = st.columns(3)
    with c[0]:
        triage_only_unscreened = st.checkbox("Only triage unscreened", value=True)
    with c[1]:
        top_n = st.slider("Triage top N (by citations)", 10, 200, 50, 10)
    with c[2]:
        batch_size = st.selectbox("Batch size", [5, 8, 10], index=1)

    # rank candidates by citation count for triage priority
    candidates = keys[:]
    candidates.sort(key=lambda k: int(st.session_state.papers[k].get("citationCount") or 0), reverse=True)
    if triage_only_unscreened:
        candidates = [k for k in candidates if is_unscreened(k)]
    candidates = candidates[:top_n]

    st.caption(f"Ready to triage {len(candidates)} papers.")
    if st.button("Run Gemini triage"):
        scope = st.session_state.scope
        for i in range(0, len(candidates), batch_size):
            batch_keys = candidates[i : i + batch_size]
            batch_papers = [st.session_state.papers[k] for k in batch_keys]
            try:
                triaged = gemini_triage_batch(g, scope, batch_papers)
                # Map paperId -> paperKey
                pid_to_key = {st.session_state.papers[k].get("paperId"): k for k in batch_keys}
                for t in triaged:
                    k = pid_to_key.get(t.get("paperId"))
                    if not k:
                        continue
                    st.session_state.gemini_meta[k] = t
                    # Optional auto-fill decisions (V2): only if unscreened
                    if is_unscreened(k):
                        st.session_state.decisions[k] = {
                            "status": t.get("decision"),
                            "reason": "",
                            "notes": "",
                            "tags": t.get("tags") or [],
                        }
                st.success(f"Triage done for batch {i//batch_size + 1}.")
            except Exception as e:
                st.error(f"Triage error: {e}")
                break

        # Build clusters
        clusters: Dict[str, List[str]] = {}
        for k, meta in st.session_state.gemini_meta.items():
            cl = (meta.get("cluster") or "Unclustered").strip()
            clusters.setdefault(cl, []).append(k)
        st.session_state.clusters = clusters

    # Show ranked list by AI score
    scored = [(k, st.session_state.gemini_meta.get(k, {}).get("relevance_score")) for k in keys if k in st.session_state.gemini_meta]
    scored = [(k, int(s or 0)) for k, s in scored]
    scored.sort(key=lambda x: x[1], reverse=True)

    st.write("Top triaged papers (click to review):")
    for k, score in scored[:25]:
        p = st.session_state.papers[k]
        with st.expander(f"[{score}] {p.get('title')} ({p.get('year')})"):
            render_paper_card(k, p, show_ai=True, key_prefix="screening")


def tab_clusters_ai(g: GeminiHelper):
    st.subheader("Clusters (AI Mode)")
    clusters = st.session_state.get("clusters") or {}
    if not clusters:
        st.info("No clusters yet. Run AI triage first.")
        return

    cluster_names = sorted(clusters.keys(), key=lambda x: (-len(clusters[x]), x.lower()))
    pick = st.selectbox("Pick a cluster", cluster_names)
    keys = clusters.get(pick) or []

    st.caption(f"{pick}: {len(keys)} papers")
    for k in keys[:30]:
        p = st.session_state.papers[k]
        render_paper_card(k, p, show_ai=True, key_prefix="ai_ranked")
        st.divider()

    st.subheader("Gap analysis + outline (from Included papers)")
    included_keys = [k for k, d in st.session_state.decisions.items() if d.get("status") == "include"]
    if not included_keys:
        st.info("No included papers yet. Mark some as Included first.")
        return

    if st.button("Generate gap notes + outline using Gemini"):
        scope = st.session_state.scope
        included_papers = [st.session_state.papers[k] for k in included_keys[:60]]
        try:
            out = gemini_gap_and_outline(g, scope, included_papers)
            st.session_state.ai_synthesis = out
            st.success("Generated synthesis.")
        except Exception as e:
            st.error(f"Synthesis error: {e}")

    syn = st.session_state.get("ai_synthesis")
    if syn:
        st.json(syn)


def tab_screening():
    st.subheader("Screening Queue")
    tab_prisma()

    keys = list(st.session_state.papers.keys())
    if not keys:
        st.info("No papers yet.")
        return

    # Show unscreened first, then maybe, then include, then exclude
    def rank(k: str) -> Tuple[int, int]:
        d = st.session_state.decisions.get(k)
        status = d.get("status") if d else None
        order = {"include": 2, "maybe": 1, "exclude": 3, None: 0}
        cites = int(st.session_state.papers[k].get("citationCount") or 0)
        return (order.get(status, 9), -cites)

    keys.sort(key=rank)

    per_page = st.selectbox("Per page", [5, 10, 20], index=1, key="screen_pp")
    page = st.number_input("Page", min_value=1, value=1, step=1, key="screen_page")
    start = (page - 1) * per_page
    end = start + per_page

    for k in keys[start:end]:
        p = st.session_state.papers[k]
        show_ai = bool(st.session_state.ai_on and st.session_state.gemini_meta.get(k))
        render_paper_card(k, p, show_ai=show_ai, key_prefix="clusters")
        st.divider()


def tab_snowballed():
    st.subheader("Snowballed (Refs / Cites / Similar)")
    tab_prisma()

    sb = st.session_state.get("snowballed", {})
    keys = sb.get("keys", [])
    meta = sb.get("meta", {})

    if not keys:
        st.info("No snowballed papers yet. Use the snowball buttons on any paper card.")
        return

    # Filters
    c = st.columns(4)
    with c[0]:
        typ = st.selectbox("Source", ["All", "refs", "cits", "sim"])
    with c[1]:
        status_filter = st.selectbox("Status", ["All", "Unscreened", "Include", "Maybe", "Exclude"])
    with c[2]:
        text_filter = st.text_input("Title contains", "", key="snow_title_filter")
    with c[3]:
        newest_first = st.checkbox("Newest first", value=True)

    def passes(k: str) -> bool:
        p = st.session_state.papers.get(k, {})
        m = meta.get(k, {})
        d = st.session_state.decisions.get(k, {})
        if typ != "All" and m.get("type") != typ:
            return False
        if status_filter == "Unscreened":
            if d.get("status") in {"include", "maybe", "exclude"}:
                return False
        elif status_filter in ["Include", "Maybe", "Exclude"]:
            if d.get("status") != status_filter.lower():
                return False
        if text_filter.strip() and text_filter.strip().lower() not in (p.get("title") or "").lower():
            return False
        return True

    filtered = [k for k in keys if passes(k)]
    if newest_first:
        filtered.sort(key=lambda k: int(meta.get(k, {}).get("ts_ms") or 0), reverse=True)

    st.caption(f"Showing {len(filtered)} / {len(keys)} snowballed papers")

    # Show provenance header + cards
    for k in filtered[:50]:
        m = meta.get(k, {})
        st.caption(f"Added via **{m.get('type')}** from: {m.get('from_title')}")
        show_ai = bool(st.session_state.ai_on and st.session_state.gemini_meta.get(k))
        render_paper_card(k, st.session_state.papers[k], show_ai=show_ai, key_prefix="snowballed")
        st.divider()


def tab_export():
    st.subheader("Export + Session")
    tab_prisma()

    keys_all = list(st.session_state.papers.keys())
    inc = [k for k, d in st.session_state.decisions.items() if d.get("status") == "include"]
    may = [k for k, d in st.session_state.decisions.items() if d.get("status") == "maybe"]
    exc = [k for k, d in st.session_state.decisions.items() if d.get("status") == "exclude"]
    uns = [k for k in keys_all if is_unscreened(k)]

    choice = st.selectbox("Which set to export?", ["Included", "Maybe", "Excluded", "Unscreened", "All"], index=0)
    if choice == "Included":
        keys = inc
    elif choice == "Maybe":
        keys = may
    elif choice == "Excluded":
        keys = exc
    elif choice == "Unscreened":
        keys = uns
    else:
        keys = keys_all

    df = export_df(keys)
    st.dataframe(df, width=True, hide_index=True)

    # CSV
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", csv_bytes, file_name=f"litreview_{choice.lower()}.csv", mime="text/csv")

    # BibTeX
    bib = export_bibtex(df)
    st.download_button("Download BibTeX", bib.encode("utf-8"), file_name=f"litreview_{choice.lower()}.bib", mime="application/x-bibtex")

    # RIS
    ris = export_ris(df)
    st.download_button("Download RIS", ris.encode("utf-8"), file_name=f"litreview_{choice.lower()}.ris", mime="application/x-research-info-systems")

    st.divider()
    st.subheader("Save / Load session")
    session_obj = {
        "version": "1.0",
        "ai_on": st.session_state.ai_on,
        "scope": st.session_state.scope,
        "papers": st.session_state.papers,
        "decisions": st.session_state.decisions,
        "gemini_meta": st.session_state.gemini_meta,
        "clusters": st.session_state.clusters,
        "search_runs": st.session_state.search_runs,
        "saved_at_ms": now_ms(),
    }
    json_download_button("Download session JSON", session_obj, "litreview_session.json")

    up = st.file_uploader("Load session JSON", type=["json"])

    # If user removes the file, allow loading again later
    if up is None:
        st.session_state._loaded_session_once = False

    if up and not st.session_state._loaded_session_once:
        try:
            loaded = json.loads(up.read().decode("utf-8"))

            st.session_state.papers = loaded.get("papers") or {}
            st.session_state.decisions = loaded.get("decisions") or {}
            st.session_state.gemini_meta = loaded.get("gemini_meta") or {}
            st.session_state.clusters = loaded.get("clusters") or {}
            st.session_state.search_runs = loaded.get("search_runs") or []
            st.session_state.scope = loaded.get("scope") or st.session_state.scope
            st.session_state.ai_on = bool(loaded.get("ai_on", st.session_state.ai_on))

            st.session_state._loaded_session_once = True
            st.success("Session loaded.")
            st.rerun()

        except Exception as e:
            st.session_state._loaded_session_once = False
            st.error(f"Load error: {e}")

    st.divider()
    st.subheader("Danger zone")
    if st.button("Reset everything"):
        st.session_state.papers = {}
        st.session_state.decisions = {}
        st.session_state.gemini_meta = {}
        st.session_state.clusters = {}
        st.session_state.search_runs = []
        st.success("Reset complete.")


# -----------------------------
# Main
# -----------------------------
def main():
    st.set_page_config(page_title="LitReview Finder", layout="wide")
    init_state()
    header_bar()

    with st.expander("Use your own API keys (optional)"):
        st.session_state.user_s2_key = st.text_input(
            "Semantic Scholar key",
            value=st.session_state.get("user_s2_key", ""),
            type="password",
        )
        st.session_state.user_gem_key = st.text_input(
            "Gemini key",
            value=st.session_state.get("user_gem_key", ""),
            type="password",
        )

    s2, gem = sidebar_config()
    # store s2 in session for snowball callbacks
    st.session_state._s2 = s2

    if st.session_state.ai_on and not gem:
        st.warning("AI mode is ON but Gemini API key is not configured in environment.")

    if st.session_state.ai_on:
        tabs = st.tabs(["Scope", "Search", "Clusters", "Screening", "Snowballed", "Export"])
        with tabs[0]:
            if gem:
                tab_scope_ai(gem)
            else:
                st.info("Configure Gemini key in the sidebar.")
        with tabs[1]:
            if gem:
                tab_search_ai(s2, gem)
            else:
                st.info("Configure Gemini key in the sidebar.")
        with tabs[2]:
            if gem:
                tab_clusters_ai(gem)
            else:
                st.info("Configure Gemini key in the sidebar.")
        with tabs[3]:
            tab_screening()
        with tabs[4]:
            tab_snowballed()
        with tabs[5]:
            tab_export()
    else:
        tabs = st.tabs(["Search", "Screening", "Snowballed", "Export"])
        with tabs[0]:
            tab_search_manual(s2)
        with tabs[1]:
            tab_screening()
        with tabs[2]:
            tab_snowballed()
        with tabs[3]:
            tab_export()


if __name__ == "__main__":
    main()