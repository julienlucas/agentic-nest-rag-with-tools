"""
Exporte les données déjà produites par ../agentic-rag (Python) vers data/, au format JSON
lu par le backend Nest. Aucun appel d'API : on réutilise l'OCR et le chunking existants,
pour que la comparaison avec le score Python ne mesure que le changement de stack.

À lancer avec l'environnement Python du projet d'origine (les pickles contiennent des
objets LangChain) :

    ../agentic-rag/.venv/bin/python scripts/export_from_python.py

Produit :
    data/financebench/<DOC>.pages.json     pages OCR (markdown), index 0
    data/financebench/<DOC>.chunks.jsonl   chunks enfants (texte + metadata, parent inclus)
    data/financebench/dataset.jsonl        26 questions, champs listes en vrai JSON
    data/financebench/sentinels.json       questions sentinelles du garde-fou
    data/financebench/retrieval_baseline.json  baseline de retrieval Python (référence)
    data/document_cache/<sha256>.json      documents d'exemple déjà OCRisés (démo)
"""
import ast
import json
import pickle
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT.parent / "agentic-rag"
FB_SRC = SRC / "evaluation" / "financebench"
OUT = ROOT / "data"
FB_OUT = OUT / "financebench"


def _page(value):
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _chunk(doc) -> dict:
    meta = dict(doc.metadata or {})
    return {
        "content": doc.page_content,
        "metadata": {
            "source": str(meta.get("source") or ""),
            "docName": str(meta.get("doc_name") or "") or None,
            "page": _page(meta.get("page")),
            "parentId": meta.get("parent_id"),
            "parentContent": meta.get("parent_content"),
        },
    }


def _as_list(value):
    """Les champs liste du dataset sont des repr Python ("[['AMD_2022_10K', 12]]")."""
    if isinstance(value, list):
        return value
    if not value:
        return []
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return [value]


def export_financebench():
    FB_OUT.mkdir(parents=True, exist_ok=True)
    cache = FB_SRC / "cache"
    for ocr in sorted(cache.glob("*.ocr.json")):
        doc = ocr.name[: -len(".ocr.json")]
        data = json.loads(ocr.read_text(encoding="utf-8"))
        if isinstance(data, list):  # ancien format du cache : liste de pages
            pages = dict(enumerate(data))
        else:
            pages = {int(k): v for k, v in (data.get("pages") or {}).items()}
        ordered = [pages.get(i, "") for i in range(max(pages) + 1)] if pages else []
        (FB_OUT / f"{doc}.pages.json").write_text(json.dumps(ordered, ensure_ascii=False), encoding="utf-8")

        pkl = cache / f"{doc}.chunks.pkl"
        if pkl.exists():
            chunks = pickle.loads(pkl.read_bytes())
            with open(FB_OUT / f"{doc}.chunks.jsonl", "w", encoding="utf-8") as f:
                for c in chunks:
                    f.write(json.dumps(_chunk(c), ensure_ascii=False) + "\n")
            print(f"{doc}: {len(ordered)} pages, {len(chunks)} chunks")

    rows = []
    for raw in (FB_SRC / "dataset.jsonl").read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        ex = json.loads(line)
        for key in ("answer_keywords", "gold_passages", "gold_pages"):
            ex[key] = _as_list(ex.get(key))
        rows.append(ex)
    with open(FB_OUT / "dataset.jsonl", "w", encoding="utf-8") as f:
        for ex in rows:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"dataset: {len(rows)} questions")

    for name in ("sentinels.json", "retrieval_baseline.json"):
        src = FB_SRC / "regression" / name
        if src.exists():
            shutil.copy(src, FB_OUT / name)


def export_document_cache():
    out = OUT / "document_cache"
    out.mkdir(parents=True, exist_ok=True)
    for pkl in sorted((SRC / "document_cache").glob("*-v3.pkl")):
        sha = pkl.name.split("-")[0]
        data = pickle.loads(pkl.read_bytes())
        chunks = [_chunk(c) for c in data["chunks"]]
        source = Path(chunks[0]["metadata"]["source"]).name if chunks else ""
        for c in chunks:
            c["metadata"]["source"] = source
        payload = {"source": source, "pages": data.get("pages") or [], "chunks": chunks}
        (out / f"{sha}.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        print(f"cache démo: {source} ({len(chunks)} chunks, {len(payload['pages'])} pages)")


if __name__ == "__main__":
    if not SRC.exists():
        sys.exit(f"Projet Python introuvable: {SRC}")
    export_financebench()
    export_document_cache()
