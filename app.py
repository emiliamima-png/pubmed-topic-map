"""
Literature Topic Map — Shiny for Python + BERTopic
==================================================

1. Runs a search strategy in PubMed (NCBI E-utilities) or OpenAlex and downloads the
   records (title, keywords, optional MeSH terms, abstract, journal, year, DOI, PMID).
2. Builds one document per article ("title. keywords. abstract") and embeds it with a
   sentence-transformers model.
3. Fits BERTopic (UMAP -> HDBSCAN -> c-TF-IDF) to find topics, and fits a separate 2D UMAP
   on the same embeddings to draw the map.
4. Overlays a comparison list (PMIDs, DOIs or OpenAlex IDs, optionally grouped). Articles
   from the list that are in the search results are drawn at their own position. Articles
   that are not can be fetched, embedded and *projected* onto the same map (and assigned a
   topic). The app also reports how many list articles the search strategy retrieves
   (a recall check).

Run locally:   pip install -r requirements.txt
               shiny run --launch-browser app.py

Optional environment variables (e.g. as secrets on a hosting service), used when the
corresponding box in the app is left empty:
    NCBI_EMAIL, NCBI_API_KEY, OPENALEX_API_KEY
"""

from __future__ import annotations

import html
import os
import re
import textwrap
import time
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
from shiny import App, reactive, render, req, ui
from shinywidgets import output_widget, render_plotly

MAX_RECORDS = 9999
DB_NAMES = {"pubmed": "PubMed", "openalex": "OpenAlex"}


def _batches(seq: list, size: int):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def _empty_record() -> dict:
    return {"key": "", "pmid": "", "openalex_id": "", "doi": "", "title": "", "abstract": "",
            "keywords": [], "mesh": [], "journal": "", "year": None}


# ----------------------------------------------------------------------------------------
# PubMed client (NCBI E-utilities)
# ----------------------------------------------------------------------------------------

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"


class PubMedClient:
    name = "PubMed"

    def __init__(self, email: str, api_key: str | None = None, tool: str = "literature_topic_map"):
        self.base = {"tool": tool, "email": email}
        if api_key:
            self.base["api_key"] = api_key
        # NCBI limits: 3 requests/s without a key, 10/s with one
        self.delay = 0.11 if api_key else 0.34
        self._last = 0.0
        self.session = requests.Session()

    def _get(self, endpoint: str, **params) -> requests.Response:
        for attempt in range(5):
            wait = self.delay - (time.time() - self._last)
            if wait > 0:
                time.sleep(wait)
            r = self.session.post(EUTILS + endpoint, data={**self.base, **params}, timeout=90)
            self._last = time.time()
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(1.5 * (attempt + 1))
                continue
            r.raise_for_status()
            return r
        r.raise_for_status()
        return r

    @staticmethod
    def _esr(r: requests.Response) -> dict:
        j = r.json()
        if "esearchresult" not in j:
            raise RuntimeError(f"Unexpected PubMed response: {str(j)[:300]}")
        e = j["esearchresult"]
        if "ERROR" in e:
            raise RuntimeError(f"PubMed error: {e['ERROR']}")
        return e

    # -- search ---------------------------------------------------------------------------
    def search_records(self, query: str, n: int, sort: str, progress=None) -> tuple[int, list[dict]]:
        e = self._esr(self._get("esearch.fcgi", db="pubmed", term=query, usehistory="y",
                                retmax=0, retmode="json", sort=sort))
        count = int(e["count"])
        if count == 0:
            return 0, []
        n = min(count, n, MAX_RECORDS)
        recs: list[dict] = []
        for start in range(0, n, 200):
            r = self._get("efetch.fcgi", db="pubmed", WebEnv=e["webenv"],
                          query_key=e["querykey"], retstart=start,
                          retmax=min(200, n - start), retmode="xml")
            recs.extend(parse_pubmed_xml(r.content))
            if progress:
                progress(min(start + 200, n), n)
        return count, recs

    # -- comparison list --------------------------------------------------------------------
    def fetch_pmids(self, pmids: list[str]) -> list[dict]:
        recs: list[dict] = []
        for chunk in _batches(pmids, 200):
            r = self._get("efetch.fcgi", db="pubmed", id=",".join(chunk), retmode="xml")
            recs.extend(parse_pubmed_xml(r.content))
        return recs

    def doi_to_pmid(self, doi: str) -> str | None:
        e = self._esr(self._get("esearch.fcgi", db="pubmed", term=f'"{doi}"[AID]',
                                retmode="json", retmax=2))
        ids = e.get("idlist", [])
        return ids[0] if len(ids) == 1 else None

    def resolve(self, pairs: list[tuple[str, str]]) -> dict[tuple[str, str], dict]:
        """Map (kind, value) identifiers to full records. OpenAlex IDs are not supported."""
        to_pmid: dict[tuple[str, str], str] = {}
        for kind, val in pairs:
            if kind == "pmid":
                to_pmid[(kind, val)] = val
            elif kind == "doi":
                pid = self.doi_to_pmid(val)
                if pid:
                    to_pmid[(kind, val)] = pid
        recs = {r["pmid"]: r for r in self.fetch_pmids(sorted(set(to_pmid.values())))}
        return {kv: recs[p] for kv, p in to_pmid.items() if p in recs}

    def retrieved(self, query: str, records: list[dict]) -> set[str]:
        """Keys of `records` that the search strategy retrieves (recall check)."""
        hits: set[str] = set()
        for chunk in _batches([r["key"] for r in records], 200):
            term = f"({query}) AND ({' OR '.join(f'{p}[uid]' for p in chunk)})"
            e = self._esr(self._get("esearch.fcgi", db="pubmed", term=term,
                                    retmax=len(chunk), retmode="json"))
            hits.update(e.get("idlist", []))
        return hits


def _text(el) -> str:
    return re.sub(r"\s+", " ", "".join(el.itertext())).strip() if el is not None else ""


def _year(article) -> int | None:
    for path in ("Journal/JournalIssue/PubDate/Year", "ArticleDate/Year"):
        el = article.find(path)
        if el is not None and el.text and el.text[:4].isdigit():
            return int(el.text[:4])
    md = article.find("Journal/JournalIssue/PubDate/MedlineDate")
    if md is not None and md.text:
        m = re.search(r"\d{4}", md.text)
        if m:
            return int(m.group())
    return None


def parse_pubmed_xml(content: bytes) -> list[dict]:
    root = ET.fromstring(content)
    out = []
    for art in root.iter("PubmedArticle"):
        mc = art.find("MedlineCitation")
        a = mc.find("Article") if mc is not None else None
        if a is None:
            continue
        abstract = " ".join(t for t in (_text(x) for x in a.findall("Abstract/AbstractText")) if t)
        doi = ""
        for aid in art.findall("PubmedData/ArticleIdList/ArticleId"):
            if aid.get("IdType") == "doi" and aid.text:
                doi = aid.text.strip().lower()
        pmid = _text(mc.find("PMID"))
        out.append({
            **_empty_record(),
            "key": pmid, "pmid": pmid, "doi": doi,
            "title": _text(a.find("ArticleTitle")),
            "abstract": abstract,
            "keywords": [k for k in (_text(x) for x in mc.findall("KeywordList/Keyword")) if k],
            "mesh": [m for m in (_text(x) for x in
                                 mc.findall("MeshHeadingList/MeshHeading/DescriptorName")) if m],
            "journal": _text(a.find("Journal/Title")),
            "year": _year(a),
        })
    return out


# ----------------------------------------------------------------------------------------
# OpenAlex client
# ----------------------------------------------------------------------------------------

OPENALEX = "https://api.openalex.org"
OA_SELECT = ("id,doi,ids,title,display_name,publication_year,primary_location,keywords,"
             "mesh,abstract_inverted_index")
OA_PAGE = 100


class OpenAlexError(RuntimeError):
    def __init__(self, status: int, message: str):
        super().__init__(f"OpenAlex error {status}: {message}")
        self.status = status


class OpenAlexClient:
    name = "OpenAlex"

    def __init__(self, api_key: str | None, scope: str = "tiab", extra_filters: str = ""):
        self.api_key = api_key or None
        self.scope = scope
        self.extra = (extra_filters or "").strip().strip(",")
        self.session = requests.Session()

    def _get(self, params: dict) -> dict:
        p = dict(params)
        if self.api_key:
            p["api_key"] = self.api_key
        for attempt in range(5):
            r = self.session.get(OPENALEX + "/works", params=p, timeout=90)
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(1.5 * (attempt + 1))
                continue
            if r.status_code in (401, 402, 403, 409):
                raise OpenAlexError(r.status_code, (
                    "the request was refused, most likely because the daily free allowance "
                    "is used up or the API key is wrong. Get a free API key at openalex.org "
                    "(account settings) and enter it in the app."))
            if r.status_code >= 400:
                try:
                    j = r.json()
                    msg = j.get("message") or j.get("error") or r.text[:300]
                except ValueError:
                    msg = r.text[:300]
                raise OpenAlexError(r.status_code, str(msg))
            return r.json()
        raise RuntimeError("OpenAlex is not responding (too many requests). Try again shortly.")

    def _query_params(self, query: str) -> dict:
        filters = []
        if self.scope == "tiab":
            # commas separate filters in OpenAlex, so they can't appear inside the search text
            filters.append("title_and_abstract.search:" + query.replace(",", " "))
        if self.extra:
            filters.append(self.extra)
        p = {"filter": ",".join(filters)} if filters else {}
        if self.scope != "tiab":
            p["search"] = query
        return p

    # -- search ---------------------------------------------------------------------------
    def search_records(self, query: str, n: int, sort: str, progress=None) -> tuple[int, list[dict]]:
        params = {**self._query_params(query), "per_page": OA_PAGE, "cursor": "*",
                  "select": OA_SELECT,
                  "sort": {"relevance": "relevance_score:desc",
                           "pub_date": "publication_date:desc"}.get(sort, "relevance_score:desc")}
        try:
            j = self._get(params)
        except OpenAlexError as e:
            if e.status != 400:
                raise
            params.pop("sort")  # some query types can't be sorted by relevance
            j = self._get(params)
        count = int(j["meta"]["count"])
        target = min(count, n, MAX_RECORDS)
        recs = [parse_openalex(w) for w in j["results"]]
        while len(recs) < target and j["results"] and j["meta"].get("next_cursor"):
            if progress:
                progress(len(recs), target)
            params["cursor"] = j["meta"]["next_cursor"]
            j = self._get(params)
            recs.extend(parse_openalex(w) for w in j["results"])
        if progress:
            progress(target, target)
        return count, recs[:target]

    # -- comparison list --------------------------------------------------------------------
    def resolve(self, pairs: list[tuple[str, str]]) -> dict[tuple[str, str], dict]:
        out: dict[tuple[str, str], dict] = {}
        specs = {"doi": ("doi", lambda v: "https://doi.org/" + v, lambda r: r["doi"]),
                 "pmid": ("ids.pmid", str, lambda r: r["pmid"]),
                 "openalex": ("ids.openalex", str, lambda r: r["key"])}
        for kind, (field, fmt, getter) in specs.items():
            vals = sorted({v for k, v in pairs if k == kind and not re.search(r"[|,]", v)})
            for chunk in _batches(vals, 50):
                wanted = set(chunk)
                j = self._get({"filter": f"{field}:" + "|".join(fmt(v) for v in chunk),
                               "per_page": OA_PAGE, "select": OA_SELECT})
                for w in j["results"]:
                    r = parse_openalex(w)
                    v = getter(r)
                    if v in wanted:
                        out.setdefault((kind, v), r)
        return out

    def retrieved(self, query: str, records: list[dict]) -> set[str]:
        hits: set[str] = set()
        base = self._query_params(query)
        for chunk in _batches([r["key"] for r in records], 50):
            p = dict(base)
            idf = "ids.openalex:" + "|".join(chunk)
            p["filter"] = f"{p['filter']},{idf}" if p.get("filter") else idf
            j = self._get({**p, "per_page": OA_PAGE, "select": "id"})
            hits.update(w["id"].rsplit("/", 1)[-1] for w in j["results"])
        return hits


def _clean_html(s: str | None) -> str:
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", s or ""))).strip()


def abstract_from_inverted(inv: dict | None) -> str:
    if not inv:
        return ""
    pos = sorted((p, w) for w, ps in inv.items() for p in ps)
    return _clean_html(" ".join(w for _, w in pos))


def parse_openalex(w: dict) -> dict:
    ids = w.get("ids") or {}
    src = (w.get("primary_location") or {}).get("source") or {}
    mesh = list(dict.fromkeys(m.get("descriptor_name") for m in (w.get("mesh") or [])
                              if m.get("descriptor_name")))
    oid = (w.get("id") or "").rsplit("/", 1)[-1]
    return {
        **_empty_record(),
        "key": oid, "openalex_id": oid,
        "pmid": (ids.get("pmid") or "").rstrip("/").rsplit("/", 1)[-1],
        "doi": re.sub(r"^https?://(dx\.)?doi\.org/", "", (w.get("doi") or "")).lower(),
        "title": _clean_html(w.get("title") or w.get("display_name")),
        "abstract": abstract_from_inverted(w.get("abstract_inverted_index")),
        "keywords": [k["display_name"] for k in (w.get("keywords") or []) if k.get("display_name")],
        "mesh": mesh,
        "journal": src.get("display_name") or "",
        "year": w.get("publication_year"),
    }


# ----------------------------------------------------------------------------------------
# Documents and comparison-list parsing
# ----------------------------------------------------------------------------------------

def build_doc(rec: dict, use_mesh: bool) -> str:
    """Title first, then keywords, then abstract, so the parts that matter most survive
    the embedding model's token limit."""
    seen, kw = set(), []
    for k in rec["keywords"] + (rec["mesh"] if use_mesh else []):
        if k.lower() not in seen:
            seen.add(k.lower())
            kw.append(k)
    parts = [rec["title"].rstrip(".") + "."]
    if kw:
        parts.append("; ".join(kw) + ".")
    if rec["abstract"]:
        parts.append(rec["abstract"])
    return " ".join(parts).strip()


def classify_id(token: str) -> tuple[str | None, str]:
    t = token.strip().strip("\"'")
    t = re.sub(r"^(pmid|pubmed)\s*[:#]?\s*", "", t, flags=re.I)
    t = re.sub(r"^(https?://(dx\.)?doi\.org/|doi:\s*)", "", t, flags=re.I)
    t = re.sub(r"^https?://(api\.)?openalex\.org/(works/)?", "", t, flags=re.I)
    if re.fullmatch(r"\d{1,9}", t):
        return "pmid", t
    if re.fullmatch(r"[Ww]\d{4,12}", t):
        return "openalex", t.upper()
    if re.match(r"10\.\d{4,9}/\S+$", t):
        return "doi", t.lower().rstrip(".")
    return None, t


def entries_from_text(text: str, default_group: str = "Comparison list") -> list[dict]:
    """One or more IDs per line; an optional ' | group name' at the end of the line."""
    rows = []
    for line in (text or "").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        ids, _, grp = line.partition("|")
        grp = grp.strip() or default_group
        ids = re.sub(r"\b(pmid|pubmed)\b\s*[:#]?\s*", "", ids, flags=re.I)
        for tok in re.split(r"[\s,;]+", ids.strip()):
            if tok:
                kind, val = classify_id(tok)
                rows.append({"raw": tok, "group": grp, "kind": kind, "value": val})
    return rows


def entries_from_upload(fileinfo: list[dict]) -> list[dict]:
    path, name = fileinfo[0]["datapath"], fileinfo[0]["name"].lower()
    if not name.endswith((".csv", ".tsv")):
        with open(path, encoding="utf-8", errors="ignore") as fh:
            return entries_from_text(fh.read())
    df = pd.read_csv(path, sep=None, engine="python", dtype=str).fillna("")
    cols = {c.lower().strip(): c for c in df.columns}
    pmid_c, doi_c = cols.get("pmid"), cols.get("doi")
    oa_c = cols.get("openalex") or cols.get("openalex_id")
    grp_c = cols.get("group") or cols.get("label") or cols.get("list")
    id_c = cols.get("id") or (None if (pmid_c or doi_c or oa_c) else df.columns[0])
    rows = []
    for _, r in df.iterrows():
        raw = next((str(r[c]).strip() for c in (pmid_c, doi_c, oa_c, id_c)
                    if c and str(r[c]).strip()), "")
        if not raw:
            continue
        kind, val = classify_id(raw)
        grp = str(r[grp_c]).strip() if grp_c and str(r[grp_c]).strip() else "Comparison list"
        rows.append({"raw": raw, "group": grp, "kind": kind, "value": val})
    return rows


# ----------------------------------------------------------------------------------------
# Modelling
# ----------------------------------------------------------------------------------------

EMBEDDING_MODELS = {
    "all-MiniLM-L6-v2": "all-MiniLM-L6-v2 (fast, general)",
    "pritamdeka/S-PubMedBert-MS-MARCO": "S-PubMedBert-MS-MARCO (biomedical)",
    "sentence-transformers/allenai-specter": "SPECTER (scientific papers)",
    "all-mpnet-base-v2": "all-mpnet-base-v2 (general, slower)",
}

# Boilerplate common in abstracts that would otherwise dominate topic labels
EXTRA_STOPWORDS = {
    "study", "studies", "patient", "patients", "result", "results", "method", "methods",
    "conclusion", "conclusions", "background", "objective", "objectives", "aim", "aims",
    "purpose", "introduction", "using", "used", "use", "based", "associated", "compared",
    "significant", "significantly", "including", "included", "showed", "found", "data",
    "analysis", "group", "groups", "total", "years", "year", "new", "high", "higher", "low",
    "lower", "increased", "reported", "review", "observed", "role", "present", "potential",
}

_EMBEDDER_CACHE: dict = {}


def get_embedder(name: str):
    from sentence_transformers import SentenceTransformer
    if name not in _EMBEDDER_CACHE:
        _EMBEDDER_CACHE[name] = SentenceTransformer(name)
    return _EMBEDDER_CACHE[name]


def fit_topic_map(docs: list[str], model_name: str, min_topic_size: int, n_neighbors: int,
                  reduce_outliers: bool, auto_merge: bool, progress=None) -> dict:
    from bertopic import BERTopic
    from hdbscan import HDBSCAN
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer
    from umap import UMAP

    n = len(docs)
    nn = max(2, min(n_neighbors, n - 1))

    if progress:
        progress(40, "Embedding documents", f"{n:,} documents with {model_name}")
    embedder = get_embedder(model_name)
    emb = embedder.encode(docs, batch_size=32, show_progress_bar=False, normalize_embeddings=True)

    if progress:
        progress(65, "Fitting BERTopic", "UMAP → HDBSCAN → c-TF-IDF")
    stop = sorted(ENGLISH_STOP_WORDS | EXTRA_STOPWORDS)
    vectorizer = CountVectorizer(stop_words=stop, ngram_range=(1, 2), min_df=2 if n >= 200 else 1)
    topic_model = BERTopic(
        embedding_model=embedder,
        umap_model=UMAP(n_neighbors=nn, n_components=5, min_dist=0.0, metric="cosine",
                        random_state=42),
        hdbscan_model=HDBSCAN(min_cluster_size=min_topic_size, metric="euclidean",
                              cluster_selection_method="eom", prediction_data=True),
        vectorizer_model=vectorizer,
        nr_topics="auto" if auto_merge else None,
        calculate_probabilities=False,
        verbose=False,
    )
    topics, _ = topic_model.fit_transform(docs, emb)
    topics = np.asarray(topics)

    if reduce_outliers and (topics == -1).any() and (topics != -1).any():
        topics = np.asarray(topic_model.reduce_outliers(docs, topics, strategy="embeddings",
                                                        embeddings=emb))
        topic_model.update_topics(docs, topics=topics.tolist(), vectorizer_model=vectorizer)

    if progress:
        progress(85, "Computing 2D map", "UMAP (2 components)")
    reducer2d = UMAP(n_neighbors=nn, n_components=2, min_dist=0.1, metric="cosine",
                     random_state=42)
    xy = reducer2d.fit_transform(emb)

    labels, short, words = {}, {}, {}
    for t in sorted(set(topics.tolist())):
        if t == -1:
            labels[t], short[t], words[t] = "Outliers", "Outliers", ""
            continue
        w = [x for x, _ in (topic_model.get_topic(t) or [])][:10]
        labels[t] = f"T{t}: " + ", ".join(w[:4])
        short[t] = f"T{t} · " + ", ".join(w[:2])
        words[t] = ", ".join(w)

    return {"topics": topics, "xy": xy, "emb": emb, "topic_model": topic_model,
            "reducer2d": reducer2d, "embedder": embedder, "labels": labels,
            "short": short, "words": words}


# ----------------------------------------------------------------------------------------
# Plot
# ----------------------------------------------------------------------------------------

TOPIC_PALETTE = px.colors.qualitative.Dark24 + px.colors.qualitative.Light24 + px.colors.qualitative.Alphabet
COMPARE_COLORS = ["#000000", "#E4002B", "#0057B8", "#FF8C00", "#7B2D8E", "#00897B"]


def _wrap(s: str, width: int = 90) -> str:
    return "<br>".join(textwrap.wrap(html.escape(s or ""), width)) or "(no title)"


def _id_label(pmid: str, openalex_id: str) -> str:
    return f"PMID {pmid}" if pmid else (f"OpenAlex {openalex_id}" if openalex_id else "")


def build_figure(corp: dict, comp: pd.DataFrame | None, show_outliers: bool = True,
                 show_labels: bool = True, dim: bool = True) -> go.Figure:
    df = corp["df"]
    has_comp = comp is not None and bool(comp["placed"].any())
    base_op = 0.35 if (dim and has_comp) else 0.8
    fig = go.Figure()

    for t in sorted(df["topic"].unique()):
        if t == -1 and not show_outliers:
            continue
        sub = df[df["topic"] == t]
        out = t == -1
        fig.add_trace(go.Scattergl(
            x=sub["x"], y=sub["y"], mode="markers",
            name=f"{corp['labels'][t]} ({len(sub)})",
            marker=dict(size=4 if out else 6, opacity=base_op * (0.6 if out else 1),
                        color="#B0B0B0" if out else TOPIC_PALETTE[t % len(TOPIC_PALETTE)]),
            hovertext=sub["hover"], hoverinfo="text",
        ))

    if show_labels:
        cent = df[df["topic"] != -1].groupby("topic")[["x", "y"]].median()
        for t, row in cent.iterrows():
            fig.add_annotation(x=row["x"], y=row["y"], text=html.escape(corp["short"][t]),
                               showarrow=False, font=dict(size=11, color="#111"),
                               bgcolor="rgba(255,255,255,0.75)", borderpad=2)

    if has_comp:
        placed = comp[comp["placed"]]
        for gi, g in enumerate(dict.fromkeys(placed["group"])):
            color = COMPARE_COLORS[gi % len(COMPARE_COLORS)]
            for src, symbol, lab in (("corpus", "star", "in search results"),
                                     ("projected", "diamond", "projected (not in map corpus)")):
                sub = placed[(placed["group"] == g) & (placed["placement"] == src)]
                if sub.empty:
                    continue
                fig.add_trace(go.Scatter(
                    x=sub["x"], y=sub["y"], mode="markers",
                    name=f"{g}: {lab} ({len(sub)})",
                    marker=dict(symbol=symbol, size=15 if symbol == "star" else 11, color=color,
                                line=dict(color="white", width=1.3)),
                    hovertext=sub["hover"], hoverinfo="text",
                ))

    fig.update_layout(
        template="plotly_white", margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(itemsizing="constant", font=dict(size=11)),
        xaxis=dict(visible=False), yaxis=dict(visible=False, scaleanchor="x"),
        hoverlabel=dict(align="left"), dragmode="pan",
    )
    return fig


# ----------------------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------------------

DEFAULT_QUERY = {
    "pubmed": '("machine learning"[tiab] OR "deep learning"[tiab]) AND sepsis[mh]',
    "openalex": '("machine learning" OR "deep learning") AND sepsis',
}

app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.accordion(
            ui.accordion_panel(
                "1 · Search",
                ui.input_radio_buttons("source", "Database",
                                       {"pubmed": "PubMed", "openalex": "OpenAlex"}, inline=True),
                ui.input_text_area("query", "Search strategy", rows=6,
                                   value=DEFAULT_QUERY["pubmed"]),
                ui.panel_conditional(
                    "input.source === 'pubmed'",
                    ui.p("PubMed syntax, exactly as on the PubMed website (field tags such as "
                         "[tiab] and [mh] work).", class_="small text-muted"),
                    ui.input_text("email", "Your e-mail (required by NCBI)", ""),
                    ui.input_password("api_key", "NCBI API key (optional, faster)", ""),
                ),
                ui.panel_conditional(
                    "input.source === 'openalex'",
                    ui.p("Use AND, OR, NOT in capitals and \"quoted phrases\". PubMed field tags "
                         "like [tiab] or [mh] do not work here.", class_="small text-muted"),
                    ui.input_password("oa_key", "OpenAlex API key (free; needed beyond small tests)", ""),
                    ui.input_select("oa_scope", "Search in",
                                    {"tiab": "Title & abstract",
                                     "all": "Title, abstract & full text"}),
                    ui.input_text("oa_filters", "Extra OpenAlex filters (optional)",
                                  placeholder="publication_year:2015-2025,type:article"),
                ),
                ui.input_numeric("max_records", "Max. articles to retrieve", 2000,
                                 min=30, max=MAX_RECORDS, step=100),
                ui.input_select("sort", "Order when capped",
                                {"relevance": "Best match", "pub_date": "Most recent"}),
            ),
            ui.accordion_panel(
                "2 · Topic model",
                ui.input_select("model", "Embedding model", EMBEDDING_MODELS),
                ui.input_checkbox("use_mesh", "Add MeSH terms to the keywords", False),
                ui.input_numeric("min_topic_size", "Min. topic size", 15, min=3, max=500),
                ui.input_numeric("n_neighbors", "UMAP neighbours", 15, min=2, max=200),
                ui.input_checkbox("reduce_outliers", "Reassign outliers to nearest topic", False),
                ui.input_checkbox("auto_merge", "Merge similar topics automatically", False),
            ),
            ui.accordion_panel(
                "3 · Comparison list",
                ui.input_text_area(
                    "compare_ids", "PMIDs, DOIs or OpenAlex IDs, one per line (optional ' | group')",
                    rows=6,
                    placeholder="31234567\n10.1016/j.jclinepi.2020.01.001\n32345678 | Included studies"),
                ui.input_file("compare_file", "…or upload CSV/TXT (columns pmid, doi, group)",
                              accept=[".csv", ".tsv", ".txt"]),
                ui.input_checkbox("project_missing",
                                  "Project list articles not in the results onto the map", True),
                ui.layout_columns(
                    ui.input_action_button("overlay", "Overlay list", class_="btn-secondary"),
                    ui.input_action_button("clear_overlay", "Clear", class_="btn-outline-secondary"),
                    col_widths=[8, 4],
                ),
            ),
            id="settings", open=["1 · Search", "3 · Comparison list"], multiple=True,
        ),
        ui.input_action_button("run", "Run search & build map", class_="btn-primary w-100"),
        width=400,
    ),
    ui.navset_card_tab(
        ui.nav_panel(
            "Map",
            ui.output_ui("status"),
            ui.layout_columns(
                ui.input_switch("show_outliers", "Show outliers", True),
                ui.input_switch("show_labels", "Topic labels", True),
                ui.input_switch("dim", "Fade background when list is shown", True),
                col_widths=[3, 3, 6],
            ),
            output_widget("topic_map", height="720px"),
        ),
        ui.nav_panel("Topics", ui.output_data_frame("topics_table")),
        ui.nav_panel("Articles", ui.output_data_frame("articles_table")),
        ui.nav_panel(
            "Comparison",
            ui.output_ui("comparison_summary"),
            ui.output_data_frame("comparison_table"),
        ),
        ui.nav_panel(
            "Export",
            ui.p("Download the map and the underlying data."),
            ui.download_button("dl_map", "Interactive map (HTML)"),
            ui.download_button("dl_articles", "Articles with topics & coordinates (CSV)"),
            ui.download_button("dl_topics", "Topics (CSV)"),
            ui.download_button("dl_comparison", "Comparison list results (CSV)"),
        ),
    ),
    title="Literature topic map · BERTopic",
    fillable=False,
)


# ----------------------------------------------------------------------------------------
# Server
# ----------------------------------------------------------------------------------------

def server(input, output, session):
    corpus = reactive.value(None)       # dict: df, models, labels, source, query, hit count
    comparison = reactive.value(None)   # DataFrame of list articles and their placement
    status_msg = reactive.value("Choose a database, enter a search strategy, then click "
                                "<b>Run search &amp; build map</b>.")

    # Swap the example query when switching database, unless the user has edited it
    @reactive.effect
    @reactive.event(input.source, ignore_init=True)
    def _swap_example():
        if input.query().strip() in DEFAULT_QUERY.values():
            ui.update_text_area("query", value=DEFAULT_QUERY[input.source()])

    def make_client(source: str, scope: str = "tiab", extra: str = ""):
        if source == "pubmed":
            email = input.email().strip() or os.environ.get("NCBI_EMAIL", "")
            if "@" not in email:
                raise ValueError("NCBI requires a valid e-mail address for PubMed searches.")
            return PubMedClient(email, input.api_key().strip() or os.environ.get("NCBI_API_KEY"))
        return OpenAlexClient(input.oa_key().strip() or os.environ.get("OPENALEX_API_KEY"),
                              scope, extra)

    # ---- Build the map ------------------------------------------------------------------
    @reactive.effect
    @reactive.event(input.run)
    def _run():
        query = input.query().strip()
        source = input.source()
        if not query:
            ui.notification_show("Please enter a search strategy.", type="warning")
            return
        if source == "openalex" and re.search(r"\[[a-z ]+\]", query, flags=re.I):
            ui.notification_show("The search contains PubMed field tags like [tiab]; OpenAlex "
                                 "ignores or rejects these. Remove them for OpenAlex.",
                                 type="warning", duration=10)
        try:
            scope, extra = input.oa_scope(), input.oa_filters()
            client = make_client(source, scope, extra)
            if source == "openalex" and not client.api_key:
                ui.notification_show("No OpenAlex API key: fine for a quick test, but larger "
                                     "searches need a free key from openalex.org.",
                                     type="warning", duration=10)
            with ui.Progress(min=0, max=100) as p:
                p.set(2, message=f"Searching {client.name}")

                def fetch_prog(done, total):
                    p.set(2 + 33 * done / max(total, 1), message="Downloading records",
                          detail=f"{done:,} / {total:,}")

                count, recs = client.search_records(query, int(input.max_records() or 2000),
                                                    input.sort(), progress=fetch_prog)
                if count == 0:
                    raise ValueError("The search returned no results.")
                recs = [r for r in recs if r["key"] and r["title"]]
                df = pd.DataFrame(recs).drop_duplicates("key")
                df = df[~(df["doi"].ne("") & df["doi"].duplicated())]
                df = df[~df["title"].str.lower().str.strip(" .").duplicated()].reset_index(drop=True)
                df["doc"] = [build_doc(r, input.use_mesh()) for r in df.to_dict("records")]
                if len(df) < 20:
                    raise ValueError(f"Only {len(df)} usable records; BERTopic needs more "
                                     "documents (at least ~20, ideally hundreds).")

                res = fit_topic_map(
                    df["doc"].tolist(), input.model(), int(input.min_topic_size() or 15),
                    int(input.n_neighbors() or 15), input.reduce_outliers(), input.auto_merge(),
                    progress=lambda v, m, d: p.set(v, message=m, detail=d),
                )
                p.set(97, message="Drawing map")

            df["topic"] = res["topics"].astype(int)
            df["topic_label"] = df["topic"].map(res["labels"])
            df["x"], df["y"] = res["xy"][:, 0], res["xy"][:, 1]
            df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
            df["hover"] = [
                f"<b>{_wrap(r.title)}</b><br>{html.escape(r.journal or '')} "
                f"({r.year if pd.notna(r.year) else 'n.d.'}) · {_id_label(r.pmid, r.openalex_id)}<br>"
                f"Topic: {html.escape(r.topic_label)}"
                for r in df.itertuples()
            ]
            corpus.set({**res, "df": df, "source": source, "query": query, "count": count,
                        "scope": scope, "extra": extra})
            comparison.set(None)

            n_topics = len([t for t in res["labels"] if t != -1])
            n_out = int((df["topic"] == -1).sum())
            n_abs = int((df["abstract"].str.len() > 0).sum())
            cap = (f" <span class='text-warning'>(capped: the search has {count:,} hits; "
                   f"only {len(df):,} were mapped)</span>") if count > len(recs) else ""
            status_msg.set(f"{client.name}: <b>{count:,}</b> hits · <b>{len(df):,}</b> articles "
                           f"mapped ({n_abs:,} with abstract) · <b>{n_topics}</b> topics · "
                           f"{n_out:,} outliers{cap}")
            if input.compare_ids().strip() or input.compare_file():
                _do_overlay()
        except Exception as e:  # noqa: BLE001
            ui.notification_show(f"Error: {e}", type="error", duration=None)

    # ---- Overlay the comparison list ------------------------------------------------------
    def _collect_entries() -> list[dict]:
        rows = entries_from_text(input.compare_ids())
        if input.compare_file():
            rows += entries_from_upload(input.compare_file())
        return rows

    def _do_overlay():
        corp = corpus()
        if corp is None:
            ui.notification_show("Build the map first.", type="warning")
            return
        entries = _collect_entries()
        if not entries:
            ui.notification_show("The comparison list is empty.", type="warning")
            return
        df = corp["df"]
        db = DB_NAMES[corp["source"]]
        client = make_client(corp["source"], corp["scope"], corp["extra"])

        keys = set(df["key"])
        by_pmid = {p: k for p, k in zip(df["pmid"], df["key"]) if p}
        by_doi = {d: k for d, k in zip(df["doi"], df["key"]) if d}

        def corpus_key(kind: str, val: str) -> str | None:
            if kind == "pmid":
                return by_pmid.get(val)
            if kind == "doi":
                return by_doi.get(val)
            if kind == "openalex":
                return val if val in keys else None
            return None

        comp = pd.DataFrame(entries).drop_duplicates(["raw", "group"]).reset_index(drop=True)
        pairs = sorted({(e["kind"], e["value"]) for e in entries if e["kind"]})
        on_map = {kv: corpus_key(*kv) for kv in pairs}

        with ui.Progress(min=0, max=100) as p:
            p.set(10, message=f"Looking up list articles in {db}")
            missing = [kv for kv, k in on_map.items() if k is None]
            resolved = client.resolve(missing) if missing else {}

            outside: dict[str, dict] = {}       # record key -> record (not on the map)
            outside_of: dict[tuple, str] = {}   # (kind, value) -> record key
            for kv, rec in resolved.items():
                ck = rec["key"] if rec["key"] in keys else None
                ck = ck or (by_doi.get(rec["doi"]) if rec["doi"] else None)
                ck = ck or (by_pmid.get(rec["pmid"]) if rec["pmid"] else None)
                if ck:
                    on_map[kv] = ck
                else:
                    outside[rec["key"]] = rec
                    outside_of[kv] = rec["key"]

            p.set(35, message="Checking which list articles the search retrieves")
            matched = client.retrieved(corp["query"], list(outside.values())) if outside else set()

            proj = {}
            if input.project_missing() and outside:
                p.set(65, message="Projecting onto the map", detail="embedding + UMAP transform")
                okeys = list(outside)
                docs = [build_doc(outside[k], input.use_mesh()) for k in okeys]
                emb = corp["embedder"].encode(docs, batch_size=32, show_progress_bar=False,
                                              normalize_embeddings=True)
                topics, _ = corp["topic_model"].transform(docs, embeddings=emb)
                xy = corp["reducer2d"].transform(emb)
                proj = {k: (float(xy[i, 0]), float(xy[i, 1]), int(topics[i]))
                        for i, k in enumerate(okeys)}

        by_key = df.set_index("key")
        rows = []
        for e in comp.to_dict("records"):
            if not isinstance(e["kind"], str):  # pandas turns None into NaN
                e["kind"] = None
            kv = (e["kind"], e["value"])
            r = {"input": e["raw"], "group": e["group"], "pmid": "", "doi": "",
                 "openalex_id": "", "found": False, "retrieved_by_search": False,
                 "placement": "not placed", "placed": False, "x": np.nan, "y": np.nan,
                 "topic": pd.NA, "topic_label": "", "title": "", "year": pd.NA, "status": ""}
            ck = on_map.get(kv) if e["kind"] else None
            if e["kind"] is None:
                r["status"] = "Unrecognised identifier"
            elif ck:
                a = by_key.loc[ck]
                r.update(pmid=a["pmid"], doi=a["doi"], openalex_id=a["openalex_id"],
                         found=True, retrieved_by_search=True, placement="corpus",
                         placed=True, x=a["x"], y=a["y"], topic=int(a["topic"]),
                         topic_label=a["topic_label"], title=a["title"], year=a["year"],
                         status="In search results (on map)")
            elif kv in outside_of:
                k = outside_of[kv]
                rec = outside[k]
                hit = k in matched
                r.update(pmid=rec["pmid"], doi=rec["doi"], openalex_id=rec["openalex_id"],
                         found=True, retrieved_by_search=hit, title=rec["title"],
                         year=rec["year"])
                why = ("retrieved by the search but beyond the record cap" if hit
                       else "not retrieved by the search")
                if k in proj:
                    x, y, t = proj[k]
                    r.update(placement="projected", placed=True, x=x, y=y, topic=t,
                             topic_label=corp["labels"].get(t, f"T{t}"),
                             status=f"Projected: {why}")
                else:
                    r["status"] = f"Not placed: {why}"
            elif e["kind"] == "openalex" and corp["source"] == "pubmed":
                r["status"] = "OpenAlex ID: use a PMID or DOI with PubMed"
            else:
                r["status"] = f"Not found in {db}"
            rows.append(r)

        out = pd.DataFrame(rows)
        out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("Int64")
        out["hover"] = [
            f"<b>[{html.escape(r.group)}] {_wrap(r.title)}</b><br>"
            f"{_id_label(r.pmid, r.openalex_id)} · {html.escape(r.status)}<br>"
            f"Topic: {html.escape(r.topic_label)}"
            for r in out.itertuples()
        ]
        comparison.set(out)

    @reactive.effect
    @reactive.event(input.overlay)
    def _overlay():
        try:
            _do_overlay()
        except Exception as e:  # noqa: BLE001
            ui.notification_show(f"Error: {e}", type="error", duration=None)

    @reactive.effect
    @reactive.event(input.clear_overlay)
    def _clear():
        comparison.set(None)

    # ---- Outputs --------------------------------------------------------------------------
    @render.ui
    def status():
        return ui.div(ui.HTML(status_msg()), class_="small text-muted mb-2")

    @render_plotly
    def topic_map():
        corp = corpus()
        req(corp)
        return build_figure(corp, comparison(), input.show_outliers(), input.show_labels(),
                            input.dim())

    def topics_df() -> pd.DataFrame:
        corp = corpus()
        req(corp)
        df = corp["df"]
        t = df.groupby("topic").size().rename("articles").reset_index()
        t["label"] = t["topic"].map(corp["labels"])
        t["top_words"] = t["topic"].map(corp["words"])
        t["median_year"] = t["topic"].map(df.groupby("topic")["year"].median())
        comp = comparison()
        if comp is not None:
            placed = comp[comp["placed"]]
            for g in dict.fromkeys(placed["group"]):
                cnt = placed[placed["group"] == g].groupby("topic").size()
                t[f"list: {g}"] = t["topic"].map(cnt).fillna(0).astype(int)
        return t.sort_values("topic")[["topic", "label", "articles", "median_year", "top_words",
                                       *[c for c in t.columns if c.startswith("list: ")]]]

    @render.data_frame
    def topics_table():
        return render.DataGrid(topics_df(), filters=True, height="650px")

    def articles_df() -> pd.DataFrame:
        corp = corpus()
        req(corp)
        df = corp["df"].copy()
        df["keywords"] = df["keywords"].map("; ".join)
        df["mesh"] = df["mesh"].map("; ".join)
        df["has_abstract"] = df["abstract"].str.len() > 0
        return df[["pmid", "openalex_id", "doi", "year", "journal", "title", "topic",
                   "topic_label", "keywords", "mesh", "has_abstract", "abstract", "x", "y"]]

    @render.data_frame
    def articles_table():
        d = articles_df().drop(columns=["abstract", "mesh", "x", "y"])
        if corpus()["source"] == "pubmed":
            d = d.drop(columns=["openalex_id"])
        return render.DataGrid(d, filters=True, height="650px")

    @render.ui
    def comparison_summary():
        comp = comparison()
        if comp is None:
            return ui.p("No comparison list overlaid yet.", class_="text-muted")
        corp = corpus()
        db = DB_NAMES[corp["source"]] if corp else "the database"
        n = len(comp)
        found = int(comp["found"].sum())
        retr = int(comp["retrieved_by_search"].sum())
        on_map = int((comp["placement"] == "corpus").sum())
        projd = int((comp["placement"] == "projected").sum())
        recall = f"{100 * retr / found:.1f}%" if found else "n/a"
        notes = ""
        if corp and corp["count"] > len(corp["df"]):
            notes = ("<br><i>The map holds only part of the search results (record cap or "
                     "duplicates removed), so some retrieved articles are projected rather "
                     "than drawn from the corpus.</i>")
        return ui.div(ui.HTML(
            f"<b>{n}</b> identifiers · <b>{found}</b> found in {db} · "
            f"<b>{retr}</b> retrieved by the search strategy (<b>recall {recall}</b>) · "
            f"{on_map} on the map from the results · {projd} projected · "
            f"{n - on_map - projd} not placed{notes}"), class_="mb-3")

    @render.data_frame
    def comparison_table():
        comp = comparison()
        req(comp is not None)
        cols = ["input", "group", "status", "retrieved_by_search", "topic_label", "year",
                "title", "pmid", "doi", "openalex_id"]
        d = comp[cols].copy()
        d["year"] = d["year"].astype(str).replace("<NA>", "")
        return render.DataGrid(d, filters=True, height="550px")

    # ---- Downloads ------------------------------------------------------------------------
    @render.download(filename="topic_map.html")
    def dl_map():
        corp = corpus()
        req(corp)
        fig = build_figure(corp, comparison(), input.show_outliers(), input.show_labels(),
                           input.dim())
        yield fig.to_html(include_plotlyjs="cdn")

    @render.download(filename="topic_map_articles.csv")
    def dl_articles():
        yield articles_df().to_csv(index=False)

    @render.download(filename="topic_map_topics.csv")
    def dl_topics():
        yield topics_df().to_csv(index=False)

    @render.download(filename="topic_map_comparison.csv")
    def dl_comparison():
        comp = comparison()
        req(comp is not None)
        yield comp.drop(columns=["hover", "placed"]).to_csv(index=False)


app = App(app_ui, server)
