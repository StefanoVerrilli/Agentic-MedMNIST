"""Bounded arXiv retrieval, immutable snapshots and citation validation.

Only publisher metadata/abstracts are retrieved. Retrieved prose is untrusted
evidence, never instructions or executable code. Offline excerpts are explicitly
labelled as curated excerpts, not as fresh web retrieval.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Callable

from contracts import CitedIdea, PriorArtBrief, SourceRecord, sha256_file, utc_now

SEED_IDS = ("2110.14795", "2104.05704", "2010.11929")
DEFAULT_QUERY = 'all:PathMNIST OR (all:histology AND all:"compact transformer")'
CURATED = (
    ("2110.14795", "MedMNIST v2 -- A large-scale lightweight benchmark for 2D and 3D biomedical image classification",
     "All images are pre-processed into a small size of 28x28 (2D)"),
    ("2104.05704", "Escaping the Big Data Paradigm with Compact Transformers",
     "with the right size, convolutional tokenization, transformers can avoid overfitting"),
    ("2010.11929", "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale",
     "a pure transformer applied directly to sequences of image patches"),
)


def canonical_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def contained_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("evidence path escapes its root")
    return path


def fetch_arxiv(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "HAT-MedMNIST/2 (research demonstrator)"})
    with urllib.request.urlopen(request, timeout=30) as response:
        if urllib.parse.urlparse(response.geturl()).hostname not in {"export.arxiv.org", "arxiv.org"}:
            raise ValueError("unexpected arXiv API redirect")
        data = response.read(2_000_001)
    if len(data) > 2_000_000:
        raise ValueError("arXiv response exceeds retrieval budget")
    return data


def parse_feed(data: bytes) -> list[dict[str, str]]:
    ns = {"a": "http://www.w3.org/2005/Atom"}
    records = []
    for entry in ET.fromstring(data).findall("a:entry", ns):
        identifier = entry.findtext("a:id", "", ns).rsplit("/", 1)[-1]
        if not re.fullmatch(r"\d{4}\.\d{4,5}(v\d+)?", identifier):
            continue
        title = " ".join(entry.findtext("a:title", "", ns).split())
        abstract = " ".join(entry.findtext("a:summary", "", ns).split())
        if len(title) >= 3 and len(abstract) >= 10:
            records.append({"id": identifier, "title": title[:1000], "text": abstract[:20000]})
    return records


class LiteratureStore:
    def __init__(self, root: Path, *, online: bool = False, require_live: bool = False,
                 query: str = DEFAULT_QUERY, limit: int = 6,
                 transport: Callable[[str], bytes] = fetch_arxiv):
        if not 3 <= limit <= 8:
            raise ValueError("literature limit must be between 3 and 8")
        self.root, self.online, self.require_live = Path(root), online, require_live
        self.query, self.limit, self.transport = query, limit, transport

    def retrieve(self, run_root: Path) -> tuple[list[SourceRecord], str, list[str]]:
        key = canonical_hash({"query": self.query, "limit": self.limit, "seeds": SEED_IDS, "version": 1})
        manifest = self.root / f"{key}.json"
        errors: list[str] = []
        if manifest.exists():
            document = json.loads(manifest.read_text(encoding="utf-8"))
            if document.get("key") != key or canonical_hash(document["sources"]) != document.get("sha256"):
                raise ValueError("literature cache manifest checksum mismatch")
            records = document["sources"]
            errors = document.get("errors", [])
            for record in records:
                path = contained_path(self.root, record["file"])
                if sha256_file(path) != record["sha256"]:
                    raise ValueError("literature cache snapshot checksum mismatch")
            mode = "cache"
        else:
            retrieved: list[dict[str, str]] = []
            if self.online:
                try:
                    base = "https://export.arxiv.org/api/query?"
                    retrieved = parse_feed(self.transport(base + urllib.parse.urlencode(
                        {"id_list": ",".join(SEED_IDS), "max_results": 3})))
                    # arXiv requests must be spaced by at least three seconds.
                    time.sleep(3)
                    extra = parse_feed(self.transport(base + urllib.parse.urlencode(
                        {"search_query": self.query, "start": 0, "max_results": self.limit,
                         "sortBy": "relevance", "sortOrder": "descending"})))
                    retrieved.extend(extra)
                except (OSError, ValueError, ET.ParseError) as exc:
                    errors.append(f"{type(exc).__name__}: {str(exc)[:300]}")
            seen = set()
            items = []
            for item in retrieved:
                stem = re.sub(r"v\d+$", "", item["id"])
                if stem not in seen:
                    items.append({**item, "origin": "arxiv_api"})
                    seen.add(stem)
            for identifier, title, excerpt in CURATED:
                if identifier not in seen:
                    items.append({"id": identifier, "title": title, "text": excerpt,
                                  "origin": "curated_excerpt"})
            # Always reserve evidence for the benchmark, CCT and ViT.
            items.sort(key=lambda x: (re.sub(r"v\d+$", "", x["id"]) not in SEED_IDS,
                                      x["id"]))
            self.root.mkdir(parents=True, exist_ok=True)
            records = []
            for item in items[:self.limit]:
                raw = {**item, "retrieved_at": utc_now()}
                digest = canonical_hash(raw)
                filename = f"source_{digest}.json"
                path = self.root / filename
                if not path.exists():
                    path.write_text(json.dumps(raw, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                records.append({"file": filename, "sha256": sha256_file(path)})
            mode = "online" if any(i["origin"] == "arxiv_api" for i in items) else "bundled"
            manifest.write_text(json.dumps({"key": key, "sources": records, "errors": errors,
                                           "sha256": canonical_hash(records)}, indent=2) + "\n", encoding="utf-8")
        result = []
        destination = run_root / "blobs" / "literature"
        destination.mkdir(parents=True, exist_ok=True)
        for record in records:
            source_path = contained_path(self.root, record["file"])
            raw_bytes = source_path.read_bytes()
            if hashlib.sha256(raw_bytes).hexdigest() != record["sha256"]:
                raise ValueError("source checksum mismatch")
            raw = json.loads(raw_bytes)
            target = destination / record["file"]
            if target.exists() and target.read_bytes() != raw_bytes:
                raise ValueError("refusing to overwrite a literature snapshot")
            target.write_bytes(raw_bytes)
            result.append(SourceRecord(
                source_id="arxiv_" + raw["id"].replace(".", "_"), title=raw["title"],
                url=f"https://arxiv.org/abs/{raw['id']}", text=raw["text"], origin=raw["origin"],
                retrieved_at=raw["retrieved_at"], snapshot_path=target.relative_to(run_root).as_posix(),
                snapshot_sha256=record["sha256"]))
        if self.require_live and any(s.origin != "arxiv_api" for s in result):
            raise ValueError("live literature required; curated evidence remains (use a fresh cache for online retrieval)")
        return result, mode, errors


def fallback_ideas(sources: list[SourceRecord]) -> list[CitedIdea]:
    result = []
    for source in sources[:3]:
        if "2110_14795" in source.source_id:
            target, hypothesis = "representation", "Keep the standardized 28x28 benchmark input; compare train-only normalization."
        elif "2104_05704" in source.source_id:
            target, hypothesis = "architecture", "Compare compact convolutional tokenization with CNNs under identical validation budgets."
        else:
            target, hypothesis = "architecture", "Evaluate patch-based attention as a bounded hypothesis; do not transfer published accuracy claims."
        result.append(CitedIdea(idea_id=f"idea_{len(result) + 1}", target=target,
                               hypothesis=hypothesis, source_id=source.source_id,
                               evidence_quote=source.text[:240]))
    return result


def citation_issues(brief: PriorArtBrief, root: Path) -> list[str]:
    issues = []
    sources = {source.source_id: source for source in brief.sources}
    if len(sources) != len(brief.sources):
        issues.append("duplicate source identifiers")
    for source in brief.sources:
        try:
            path = contained_path(root, source.snapshot_path)
            if sha256_file(path) != source.snapshot_sha256:
                raise ValueError("snapshot hash mismatch")
            raw = json.loads(path.read_text(encoding="utf-8"))
            expected_id = "arxiv_" + raw["id"].replace(".", "_")
            if (source.source_id != expected_id or source.url != f"https://arxiv.org/abs/{raw['id']}"
                    or any(getattr(source, name) != raw[name] for name in ("title", "text", "origin", "retrieved_at"))):
                raise ValueError("citation metadata does not match snapshot")
        except (OSError, ValueError, KeyError) as exc:
            issues.append(f"{source.source_id}: {exc}")
    for idea in brief.ideas:
        source = sources.get(idea.source_id)
        if source is None or " ".join(idea.evidence_quote.split()) not in " ".join(source.text.split()):
            issues.append(f"{idea.idea_id}: unresolved citation or unsupported evidence quote")
    if len({idea.idea_id for idea in brief.ideas}) != len(brief.ideas):
        issues.append("duplicate idea identifiers")
    for proposal in brief.proposals:
        if any(identifier not in sources for identifier in proposal.source_ids):
            issues.append(f"{proposal.proposal_id}: unresolved proposal citation")
    if len({p.proposal_id for p in brief.proposals}) != len(brief.proposals):
        issues.append("duplicate proposal identifiers")
    return issues
