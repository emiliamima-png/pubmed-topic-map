"""
PubMed Topic Map — Shiny for Python + BERTopic
==============================================

1. Runs a PubMed search strategy through the NCBI E-utilities and downloads the
   records (title, author keywords, optional MeSH terms, abstract, journal, year, DOI).
2. Builds one document per article ("title. keywords. abstract") and embeds it with a
   sentence-transformers model.
3. Fits BERTopic (UMAP -> HDBSCAN -> c-TF-IDF) to find topics, and fits a separate 2D UMAP
   on the same embeddings to draw the map.
4. Overlays a comparison list (PMIDs and/or DOIs, optionally grouped). Articles from the
   list that are in the search results are drawn at their own position. Articles that are
   not in the results can be fetched, embedded and *projected* onto the same map (and assigned
   a topic) using the fitted UMAP and BERTopic models. The app also reports how many list
   articles the search strategy retrieves, which works as a recall check.

Run:   pip install -r requirements.txt
       shiny run --reload app.py
"""

from __future__ import annotations

import html
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

# ----------------------------------------------------------------------------------------
# PubMed client (NCBI E-utilities)
# ----------------------------------------------------------------------------------------

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
MAX_PUBMED_RECORDS = 9999  # E-utilities will not page past 9,999 records for PubMed


class PubMedClient:
    def __init__(self, email: str, api_key: str | None = None, tool: str = "pubmed_topic_map"):
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

    def search(self, term: str, sort: str = "relevance") -> tuple[int, str, str]:
        e = self._esr(self._get("esearch.fcgi", db="pubmed", term=term, usehistory="y",
                                retmax=0, retmode="json", sort=sort))
        return int(e["count"]), e["webenv"], e["querykey"]

    def fetch_history(self, webenv: str, query_key: str, n: int, batch: int = 200,
                      progress=None) -> list[dict]:
        recs: list[dict] = []
        for start in range(0, n, batch):
            r = self._get("efetch.fcgi", db="pubmed", WebEnv=webenv, query_key=query_key,
                          retstart=start, retmax=min(batch, n - start), retmode="xml")
            recs.extend(parse_pubmed_xml(r.content))
            if progress:
                progress(min(start + batch, n), n)
        return recs

    def fetch_pmids(self, pmids: list[str], batch: int = 200) -> list[dict]:
        recs: list[dict] = []
        for i in range(0, len(pmids), batch):
            r = self._get("efetch.fcgi", db="pubmed", id=",".join(pmids[i:i + batch]),
                          retmode="xml")
            recs.extend(parse_pubmed_xml(r.content))
        return recs

    def doi_to_pmid(self, doi: str) -> str | None:
        e = self._esr(self._get("esearch.fcgi", db="pubmed", term=f'"{doi}"[AID]',
                                retmode="json", retmax=2))
        ids = e.get("idlist", [])
        return ids[0] if len(ids) == 1 else None

    def filter_by_query(self, query: str, pmids: list[str], batch: int = 200) -> set[str]:
        """Return the subset of `pmids` that the search strategy retrieves (recall check)."""
        hits: set[str] = set()
        for i in range(0, len(pmids), batch):
            chunk = pmids[i:i + batch]
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
        out.append({
            "pmid": _text(mc.find("PMID")),
            "title": _text(a.find("ArticleTitle")),
            "abstract": abstract,
            "keywords": [k for k in (_text(x) for x in mc.findall("KeywordList/Keyword")) if k],
            "mesh": [m for m in (_text(x) for x in
                                 mc.findall("MeshHeadingList/MeshHeading/DescriptorName")) if m],
            "journal": _text(a.find("Journal/Title")),
            "year": _year(a),
            "doi": doi,
        })
    return out


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


# ----------------------------------------------------------------------------------------
# Comparison-list parsing
# ----------------------------------------------------------------------------------------

def classify_id(token: str) -> tuple[str | None, str]:
    t = token.strip().strip("\"'")
    t = re.sub(r"^(pmid|pubmed)\s*[:#]?\s*", "", t, flags=re.I)
    t = re.sub(r"^(https?://(dx\.)?doi\.org/|doi:\s*)", "", t, flags=re.I)
    if re.fullmatch(r"\d{1,9}", t):
        return "pmid", t
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
    grp_c = cols.get("group") or cols.get("label") or cols.get("list")
    id_c = cols.get("id") or (None if (pmid_c or doi_c) else df.columns[0])
    rows = []
    for _, r in df.iterrows():
        raw = (r[pmid_c] if pmid_c else "") or (r[doi_c] if doi_c else "") or (r[id_c] if id_c else "")
        raw = str(raw).strip()
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

app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.accordion(
            ui.accordion_panel(
                "1 · PubMed search",
                ui.input_text_area(
                    "query", "Search strategy (PubMed syntax)", rows=6,
                    value='("machine learning"[tiab] OR "deep learning"[tiab]) AND sepsis[mh]'),
                ui.input_text("email", "Your e-mail (required by NCBI)", ""),
                ui.input_password("api_key", "NCBI API key (optional, faster)", ""),
                ui.input_numeric("max_records", "Max. articles to retrieve", 2000,
                                 min=30, max=MAX_PUBMED_RECORDS, step=100),
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
                    "compare_ids", "PMIDs or DOIs, one per line (optional ' | group')", rows=6,
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
            id="settings", open=["1 · PubMed search", "3 · Comparison list"], multiple=True,
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
    title="PubMed topic map · BERTopic",
    fillable=False,
)


# ----------------------------------------------------------------------------------------
# Server
# ----------------------------------------------------------------------------------------

def server(input, output, session):
    corpus = reactive.value(None)       # dict: df, models, labels, query, hit count
    comparison = reactive.value(None)   # DataFrame of list articles and their placement
    status_msg = reactive.value("Enter a search strategy and your e-mail, then click "
                                "<b>Run search &amp; build map</b>.")

    def client() -> PubMedClient:
        email = input.email().strip()
        if "@" not in email:
            raise ValueError("NCBI requires a valid e-mail address for E-utilities requests.")
        return PubMedClient(email, input.api_key().strip() or None)

    # ---- Build the map ------------------------------------------------------------------
    @reactive.effect
    @reactive.event(input.run)
    def _run():
        query = input.query().strip()
        if not query:
            ui.notification_show("Please enter a search strategy.", type="warning")
            return
        try:
            with ui.Progress(min=0, max=100) as p:
                pm = client()
                p.set(2, message="Searching PubMed")
                count, webenv, qk = pm.search(query, sort=input.sort())
                if count == 0:
                    raise ValueError("The search returned no results.")
                n = min(count, int(input.max_records() or 2000), MAX_PUBMED_RECORDS)

                def fetch_prog(done, total):
                    p.set(2 + 33 * done / total, message="Downloading records",
                          detail=f"{done:,} / {total:,}")

                recs = pm.fetch_history(webenv, qk, n, progress=fetch_prog)
                recs = [r for r in recs if r["pmid"] and r["title"]]
                df = pd.DataFrame(recs).drop_duplicates("pmid").reset_index(drop=True)
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
            df["year"] = df["year"].astype("Int64")
            df["hover"] = [
                f"<b>{_wrap(r.title)}</b><br>{html.escape(r.journal or '')} "
                f"({r.year if pd.notna(r.year) else 'n.d.'}) · PMID {r.pmid}<br>"
                f"Topic: {html.escape(r.topic_label)}"
                for r in df.itertuples()
            ]
            corpus.set({**res, "df": df, "query": query, "count": count})
            comparison.set(None)

            n_topics = len([t for t in res["labels"] if t != -1])
            n_out = int((df["topic"] == -1).sum())
            cap = (f" <span class='text-warning'>(capped: the search has {count:,} hits; "
                   f"only {len(df):,} were mapped)</span>") if count > n else ""
            status_msg.set(f"<b>{count:,}</b> PubMed hits · <b>{len(df):,}</b> articles mapped · "
                           f"<b>{n_topics}</b> topics · {n_out:,} outliers{cap}")
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
        pm = client()
        with ui.Progress(min=0, max=100) as p:
            p.set(5, message="Resolving identifiers")
            doi_map = {d: pid for d, pid in zip(df["doi"], df["pmid"]) if d}
            for e in entries:
                if e["kind"] == "pmid":
                    e["pmid"] = e["value"]
                elif e["kind"] == "doi":
                    e["pmid"] = doi_map.get(e["value"]) or pm.doi_to_pmid(e["value"])
                else:
                    e["pmid"] = None
            comp = pd.DataFrame(entries).drop_duplicates(["raw", "group"]).reset_index(drop=True)

            resolved = sorted({p_ for p_ in comp["pmid"] if p_})
            corpus_set = set(df["pmid"])
            p.set(25, message="Checking which list articles the search retrieves")
            matched = pm.filter_by_query(corp["query"], resolved) if resolved else set()

            outside = [p_ for p_ in resolved if p_ not in corpus_set]
            p.set(45, message="Fetching list articles not on the map", detail=f"{len(outside)}")
            rec_map = {r["pmid"]: r for r in (pm.fetch_pmids(outside) if outside else [])}

            proj = {}
            if input.project_missing() and rec_map:
                p.set(70, message="Projecting onto the map", detail="embedding + UMAP transform")
                pmids = list(rec_map)
                docs = [build_doc(rec_map[k], input.use_mesh()) for k in pmids]
                emb = corp["embedder"].encode(docs, batch_size=32, show_progress_bar=False,
                                              normalize_embeddings=True)
                topics, _ = corp["topic_model"].transform(docs, embeddings=emb)
                xy = corp["reducer2d"].transform(emb)
                proj = {k: (float(xy[i, 0]), float(xy[i, 1]), int(topics[i]))
                        for i, k in enumerate(pmids)}

        by_pmid = df.set_index("pmid")
        rows = []
        for e in comp.to_dict("records"):
            pid = e["pmid"]
            r = {"input": e["raw"], "group": e["group"], "pmid": pid or "",
                 "in_pubmed": False, "retrieved_by_search": bool(pid and pid in matched),
                 "on_map": False, "placement": "not placed", "placed": False,
                 "x": np.nan, "y": np.nan, "topic": pd.NA, "topic_label": "", "title": "",
                 "year": pd.NA, "status": ""}
            if e["kind"] is None:
                r["status"] = "Unrecognised identifier"
            elif not pid:
                r["status"] = "DOI not found in PubMed"
            elif pid in corpus_set:
                a = by_pmid.loc[pid]
                r.update(in_pubmed=True, on_map=True, placement="corpus", placed=True,
                         x=a["x"], y=a["y"], topic=int(a["topic"]),
                         topic_label=a["topic_label"], title=a["title"], year=a["year"],
                         status="In search results (on map)")
            elif pid in rec_map:
                rec = rec_map[pid]
                r.update(in_pubmed=True, title=rec["title"], year=rec["year"])
                why = ("retrieved by the search but beyond the record cap"
                       if r["retrieved_by_search"] else "not retrieved by the search")
                if pid in proj:
                    x, y, t = proj[pid]
                    r.update(placement="projected", placed=True, x=x, y=y, topic=t,
                             topic_label=corp["labels"].get(t, f"T{t}"),
                             status=f"Projected: {why}")
                else:
                    r["status"] = f"Not placed: {why}"
            else:
                r["status"] = "PMID not found in PubMed"
            rows.append(r)

        out = pd.DataFrame(rows)
        out["hover"] = [
            f"<b>[{html.escape(r.group)}] {_wrap(r.title)}</b><br>PMID {r.pmid} · "
            f"{html.escape(r.status)}<br>Topic: {html.escape(r.topic_label)}"
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
        return df[["pmid", "year", "journal", "title", "topic", "topic_label", "keywords",
                   "mesh", "has_abstract", "doi", "abstract", "x", "y"]]

    @render.data_frame
    def articles_table():
        d = articles_df().drop(columns=["abstract", "mesh", "x", "y"])
        return render.DataGrid(d, filters=True, height="650px")

    @render.ui
    def comparison_summary():
        comp = comparison()
        if comp is None:
            return ui.p("No comparison list overlaid yet.", class_="text-muted")
        corp = corpus()
        n = len(comp)
        in_pm = int(comp["in_pubmed"].sum())
        retr = int(comp["retrieved_by_search"].sum())
        on_map = int((comp["placement"] == "corpus").sum())
        projd = int((comp["placement"] == "projected").sum())
        recall = f"{100 * retr / in_pm:.1f}%" if in_pm else "n/a"
        notes = ""
        if corp and corp["count"] > len(corp["df"]):
            notes = ("<br><i>The map holds only part of the search results (record cap), so "
                     "some retrieved articles are projected rather than drawn from the corpus.</i>")
        return ui.div(ui.HTML(
            f"<b>{n}</b> identifiers · <b>{in_pm}</b> found in PubMed · "
            f"<b>{retr}</b> retrieved by the search strategy (<b>recall {recall}</b>) · "
            f"{on_map} on the map from the results · {projd} projected · "
            f"{n - on_map - projd} not placed{notes}"), class_="mb-3")

    @render.data_frame
    def comparison_table():
        comp = comparison()
        req(comp is not None)
        cols = ["input", "group", "pmid", "status", "retrieved_by_search", "topic_label",
                "year", "title"]
        return render.DataGrid(comp[cols], filters=True, height="550px")

    # ---- Downloads ------------------------------------------------------------------------
    @render.download(filename="pubmed_topic_map.html")
    def dl_map():
        corp = corpus()
        req(corp)
        fig = build_figure(corp, comparison(), input.show_outliers(), input.show_labels(),
                           input.dim())
        yield fig.to_html(include_plotlyjs="cdn")

    @render.download(filename="pubmed_topic_map_articles.csv")
    def dl_articles():
        yield articles_df().to_csv(index=False)

    @render.download(filename="pubmed_topic_map_topics.csv")
    def dl_topics():
        yield topics_df().to_csv(index=False)

    @render.download(filename="pubmed_topic_map_comparison.csv")
    def dl_comparison():
        comp = comparison()
        req(comp is not None)
        yield comp.drop(columns=["hover", "placed"]).to_csv(index=False)


app = App(app_ui, server)
