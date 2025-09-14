import json
import re
from pathlib import Path
from typing import List, Dict, Any, Set, Optional 
from tqdm import trange, tqdm

PREFIX_MAP = {
    "http://dbpedia.org/ontology/":               "dbo",
    "http://dbpedia.org/property/":               "dbp",
    "http://dbpedia.org/resource/":               "res",
    "http://dbpedia.org/class/yago/":             "yago",
    "http://www.w3.org/2000/01/rdf-schema#":      "rdfs",
    "http://www.w3.org/1999/02/22-rdf-syntax-ns#":"rdf",
    "http://www.w3.org/2002/07/owl#":             "owl",
    "http://www.w3.org/2001/XMLSchema#":          "xsd",
    "http://xmlns.com/foaf/0.1/":                 "foaf",
    "http://purl.org/dc/elements/1.1/":           "dc",
    "http://purl.org/dc/terms/":                  "dcterms",
    "http://www.w3.org/2004/02/skos/core#":       "skos",
    "http://www.w3.org/2003/01/geo/wgs84_pos#":   "geo",
    "http://www.georss.org/georss/":              "georss",
    "http://dbpedia.org/":                        "dbpedia",
    "http://purl.org/linguistics/gold/":          "gold",
}
_PREFIX_ORDER = sorted(PREFIX_MAP.items(), key=lambda kv: -len(kv[0]))

WIKIDATA_PREFIX_MAP = {
    "http://www.wikidata.org/entity/":                      "wd",
    "http://www.wikidata.org/prop/direct/":                 "wdt",
    "http://www.wikidata.org/prop/":                        "p",
    "http://www.wikidata.org/prop/statement/":              "ps",
    "http://www.wikidata.org/prop/statement/value/":        "psv",
    "http://www.wikidata.org/prop/statement/value-normalized/": "psn",
    "http://www.wikidata.org/prop/qualifier/":              "pq",
    "http://www.wikidata.org/prop/qualifier/value/":        "pqv",
    "http://www.wikidata.org/prop/qualifier/value-normalized/": "pqn",
    "http://www.wikidata.org/prop/reference/":              "pr",
    "http://www.wikidata.org/prop/reference/value/":        "prv",
    "http://www.wikidata.org/prop/reference/value-normalized/": "prn",
    "http://www.wikidata.org/prop/direct-normalized/":      "wdtn",
    "http://www.wikidata.org/prop/novalue/":                "wdno",
    "http://www.wikidata.org/entity/statement/":            "wds",
    "http://www.wikidata.org/reference/":                   "wdref",
    "http://www.wikidata.org/value/":                       "wdv",
    "http://www.wikidata.org/wiki/Special:EntityData/":     "wdata",

    "http://www.w3.org/1999/02/22-rdf-syntax-ns#":          "rdf",
    "http://www.w3.org/2000/01/rdf-schema#":                "rdfs",
    "http://www.w3.org/2002/07/owl#":                       "owl",
    "http://www.w3.org/2001/XMLSchema#":                    "xsd",
    "http://www.w3.org/2004/02/skos/core#":                 "skos",
    "http://schema.org/":                                    "schema",
    "http://www.opengis.net/ont/geosparql#":                "geo",
    "http://www.w3.org/ns/prov#":                           "prov",
    "http://www.w3.org/ns/lemon/ontolex#":                  "ontolex",
    "http://www.bigdata.com/rdf#":                          "bd",
    "http://www.bigdata.com/queryHints#":                   "hint",
}

_WIKIDATA_PREFIX_ORDER = sorted(WIKIDATA_PREFIX_MAP.items(), key=lambda kv: -len(kv[0]))

IRI_RX    = re.compile(r"<\s*([^>\s]+)\s*>")
SELECT_RX = re.compile(r"\bSELECT\b", re.I)


def _replace_uri(match: re.Match) -> str:
    uri = match.group(1).strip()
    for ns, prefix in _PREFIX_ORDER:
        if uri.startswith(ns):
            return f"{prefix}:{uri[len(ns):].lstrip(':')}"
    return match.group(0)


def uri_collapse_after_select(sparql: str) -> str:
    m = SELECT_RX.search(sparql)
    if not m:
        return sparql

    body = sparql[m.start():].strip()
    return IRI_RX.sub(_replace_uri, body)


STRING_RE = r'"(?:[^"\\]|\\.)*"(?:@[A-Za-z\-]+|\^\^[^\s;,.{}()]+)?'
IRI_RE    = r'<[^>]*>'
PNAME_RE  = r'[^\s;,.{}()]+'
SEP_RE    = r'[;,.]'
TOKEN_RE  = rf'(?:{STRING_RE}|{IRI_RE}|{PNAME_RE}|{SEP_RE})'


def _extract_triples(block_text: str) -> List[List[str]]:
    strip = [
        r'FILTER\s*\([^)]*\)', r'BIND\s*\([^)]*\)', r'GROUP\s+BY[^.}]*',
        r'HAVING[^.}]*', r'ORDER\s+BY[^.}]*', r'LIMIT\s+\d+', r'OFFSET\s+\d+',
    ]
    for pat in strip:
        block_text = re.sub(pat, '', block_text, flags=re.I | re.S)

    tokens, triples, subj, pred = re.findall(TOKEN_RE, block_text), [], None, None
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t == '.':
            subj = pred = None
        elif t == ';':
            pred = None
        elif t == ',':
            pass
        elif subj is None:
            subj = t
        elif pred is None:
            pred = t
        else:
            triples.append([subj, pred, t])
        i += 1
    return triples


def _top_level_blocks(where_text: str) -> List[str]:
    blocks, buf, depth = [], [], 0
    for ch in where_text:
        if ch == "{":
            if depth == 0 and buf:
                blocks.append("".join(buf).strip()); buf = []
            depth += 1
            if depth > 1:
                buf.append(ch)
        elif ch == "}":
            depth -= 1
            if depth > 0:
                buf.append(ch)
            else:
                blocks.append("".join(buf).strip()); buf = []
        else:
            buf.append(ch)
    tail = "".join(buf).strip()
    if tail:
        blocks.append(tail)
    return blocks


PREFIX_PATTERN = re.compile(r'PREFIX\s+([a-z0-9]+):\s*<([^>]+)>', re.I)
WHERE_PATTERN  = re.compile(r'WHERE\s*\{(.+?)\}', re.I | re.S)


class QALDPreprocessor:
    def __init__(self, include_all_langs: bool = False):
        self.include_all_langs = include_all_langs

    def _load(self, path: str) -> List[Dict[str, Any]]:
        with open(path, encoding="utf-8") as f:
            questions = json.load(f)["questions"]

        for q in questions:
            if "new_query" not in q:
                raw = q.get("query", {}).get("sparql", "")
                q["new_query"] = uri_collapse_after_select(raw)
        return questions

    @staticmethod
    def _save(data: List[Dict[str, Any]], path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

    def _filter_english(self, qs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out = []
        for q in qs:
            entry = {
                "id":            q.get("id", ""),
                "question":      "",
                "formated_query": q["new_query"],
                "answers":       [],
            }

            for phr in q.get("question", []):
                if phr.get("language") == "en":
                    entry["question"] = phr.get("string", "")
                if not self.include_all_langs:
                    break

            a0 = (q.get("answers") or [{}])[0]
            if "results" in a0:
                entry["answers"] = a0["results"]["bindings"]
            elif "boolean" in a0:
                entry["answers"] = [a0["boolean"]]

            out.append(entry)
        return out

    def run(self, in_path: str, out_path: str) -> List[Dict[str, Any]]:
        data = self._filter_english(self._load(in_path))
        self._save(data, out_path)
        return data


class SparqlParser:
    def __init__(self) -> None:
        self.global_prefixes: Dict[str, str] = {}

    def parse_sparql(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        for idx in trange(len(samples), desc="Parsing SPARQL"):
            s          = samples[idx]
            query_text = s["formated_query"]

            # answer map
            ans_map: Dict[str, List[str]] = {}
            for a in s.get("answers", []):
                if isinstance(a, bool):
                    continue
                v = next(iter(a))
                ans_map.setdefault(v, []).append(a[v]["value"])
            s["answers_value"] = ans_map

            # PREFIX lines
            local = dict(PREFIX_PATTERN.findall(query_text))
            for k, v in local.items():
                self.global_prefixes.setdefault(k, v)

            m = WHERE_PATTERN.search(query_text)
            if not m:
                s["triples"] = []
                continue
            where_txt = m.group(1)

            flat = []
            for block in _top_level_blocks(where_txt):
                flat.extend(_extract_triples(block))

            expanded = []
            for tri in flat:
                exp = []
                for tok in tri:
                    p, *tail = tok.split(":", 1)
                    exp.append(
                        f"{local[p]}:{tail[0]}" if tail and p in local else tok
                    )
                expanded.append(exp)
            s["triples"] = expanded
        return samples

    def run(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return self.parse_sparql(data)


class LCQAPreprocessor:

    def _load(self, path: str) -> List[Dict[str, Any]]:
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def _save(self, data: List[Dict[str, Any]], path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

    # ----------------- public -----------------
    def run(self, in_path: str, out_path: str) -> List[Dict[str, Any]]:
        proc: List[Dict[str, Any]] = []

        for rec in self._load(in_path):
            r = rec.copy()

            # --- field harmonisation ---
            r["id"]        = r.pop("_id",               r.get("id"))
            r["question"]  = r.pop("corrected_question", r.get("question"))
            raw_query      = r.pop("sparql_query",       r.get("query"))
            r["query"]     = raw_query                       # keep a copy
            r["formated_query"] = uri_collapse_after_select(raw_query)

            # --- answer normalisation ---
            ans_in = r.get("answers") or []

            # LCQuAD v1 → list[str]; v2 → list[dict] already
            norm: List[Dict[str, Dict[str, str]]] = []
            for a in ans_in:
                if isinstance(a, dict):                   # already in bindings style
                    norm.append(a)
                else:                                     # scalar → wrap
                    norm.append({"callret-0": {"value": str(a)}})

            r["answers"] = norm
            proc.append(r)

        self._save(proc, out_path)
        return proc


class SparqlParserLCQuad(SparqlParser):
    def parse_sparql(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        for s in samples:
            # guarantee list of dicts – should already be true after pre-proc
            if isinstance(s.get("answers"), dict):
                s["answers"] = [s["answers"]]
        return super().parse_sparql(samples)



class RankedTripleEntityExtractor:
    """
    • Reads at most *top_k* triples from ``retrived_triples_ranked`` (default 100)
    • Extracts DBpedia resource IDs into ``sample["ranked_entities"]``
    • Optionally stores the truncated list back, so every downstream step
      really sees only the top-k.
    """

    _ANGLE_RX  = re.compile(r"<\s*http://dbpedia\.org/resource/([^>\s]+)\s*>")
    _PREFIX_RX = re.compile(r"^(?:res|dbr):(.+)$", re.I)

    # ---------- construction ------------------------------------------------

    def __init__(
        self,
        *,
        deduplicate: bool = True,
        top_k: int = 100,
        overwrite_list: bool = True,
    ) -> None:
        self.deduplicate     = deduplicate
        self.top_k           = top_k
        self.overwrite_list  = overwrite_list

    # ---------- helpers -----------------------------------------------------

    @classmethod
    def _token_to_entity(cls, tok: str) -> Optional[str]:
        m = cls._ANGLE_RX.match(tok)
        if m:
            return m.group(1)
        m = cls._PREFIX_RX.match(tok)
        if m:
            return m.group(1)
        return None

    @classmethod
    def _entities_from_triple(cls, triple: List[str]) -> Set[str]:
        return {e for t in triple for e in [cls._token_to_entity(t)] if e}

    # ---------- main --------------------------------------------------------

    def run(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        for sample in tqdm(
            data,
            desc=f"Extracting entities (top-{self.top_k})",
            total=len(data),
        ):
            if "ranked_entities" in sample:
                continue

            ranked_list = sample.get("retrived_triples_ranked", [])
            sliced      = ranked_list[: self.top_k]        # ← top-k here

            if self.overwrite_list:
                sample["retrived_triples_ranked"] = sliced  # keep list small

            ents: Set[str] = set()
            for obj in sliced:
                tri = obj.get("triple", [])
                if len(tri) == 3:
                    ents.update(self._entities_from_triple(tri))

            sample["ranked_entities"] = (
                sorted(ents) if self.deduplicate else list(ents)
            )

        return data
    
class LCQUAP2processor:

    def _load(self, path: str):
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def _save(self, data: List[Dict[str, Any]], path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

    # ----------------- public -----------------
    def run(self, in_path: str, out_path: str):
        proc: List[Dict[str, Any]] = []

        for rec in self._load(in_path):
            r = rec.copy()

            r["id"]        = r.pop("uid")
            r["question"]  = r.pop("question")
            raw_query      = r.pop("sparql_wikidata")
            r["query"]     = raw_query
            r["temlate_id"] = r.pop("template_id", "")
            r["template"]  = r.pop("template", "")
            r["answers"] = r.pop("answers", [])
            # r["formated_query"] = uri_collapse_after_select(raw_query)

            # --- answer normalisation ---
            # ans_in = r.get("answers") or []

            # # LCQuAD v1 → list[str]; v2 → list[dict] already
            # norm: List[Dict[str, Dict[str, str]]] = []
            # for a in ans_in:
            #     if isinstance(a, dict):                   # already in bindings style
            #         norm.append(a)
            #     else:                                     # scalar → wrap
            #         norm.append({"callret-0": {"value": str(a)}})

            # r["answers"] = norm
            proc.append(r)

        self._save(proc, out_path)
        return proc
    

class SparqlParserLCQuad2:
    """
    SPARQL parser for LC-QuAD 2.0 that:
    - Extracts triples from the WHERE clause
    - Keeps prefix notation (e.g., dbo:birthPlace)
    - Does NOT expand prefixes or process answers
    - Does NOT remove PREFIX lines
    """

    def parse_sparql(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        for s in samples:
            query_text = s.get("query", "")
            m = WHERE_PATTERN.search(query_text)
            if not m:
                s["triples"] = []
                continue

            where_txt = m.group(1)

            flat_triples = []
            for block in _top_level_blocks(where_txt):
                flat_triples.extend(_extract_triples(block))

            s["triples"] = flat_triples
        return samples

    def run(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return self.parse_sparql(data)
    

class QALD10WikidataPreprocessor(QALDPreprocessor):
    @staticmethod
    def _replace_uri_wd(match: re.Match) -> str:
        uri = match.group(1).strip()
        for ns, prefix in _WIKIDATA_PREFIX_ORDER:
            if uri.startswith(ns):
                return f"{prefix}:{uri[len(ns):].lstrip(':')}"
        return match.group(0)

    @classmethod
    def _collapse_after_select_wd(cls, sparql: str) -> str:
        m = SELECT_RX.search(sparql)
        if not m:
            return sparql
        body = sparql[m.start():].strip()
        return IRI_RX.sub(cls._replace_uri_wd, body)

    def _load(self, path: str) -> List[Dict[str, Any]]:
        with open(path, encoding="utf-8") as f:
            questions = json.load(f)["questions"]

        for q in questions:
            if "new_query" not in q:
                raw = q.get("query", {}).get("sparql", "")
                q["new_query"] = self._collapse_after_select_wd(raw)
        return questions
    

class WikidataSparqlParser(SparqlParser):
    def __init__(self, default_prefixes: Optional[Dict[str, str]] = None) -> None:
        super().__init__()
        self.default_prefixes: Dict[str, str] = dict(default_prefixes or {})
        if not self.default_prefixes:
            self.default_prefixes = {pfx: ns for ns, pfx in WIKIDATA_PREFIX_MAP.items()}

    def parse_sparql(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        for idx in trange(len(samples), desc="Parsing SPARQL (Wikidata)"):
            s          = samples[idx]
            query_text = s["formated_query"]

            ans_map: Dict[str, List[str]] = {}
            for a in s.get("answers", []):
                if isinstance(a, bool):
                    continue
                v = next(iter(a))
                ans_map.setdefault(v, []).append(a[v]["value"])
            s["answers_value"] = ans_map

            local = dict(PREFIX_PATTERN.findall(query_text))

            prefixes: Dict[str, str] = dict(self.default_prefixes)
            prefixes.update(self.global_prefixes)
            prefixes.update(local)

            for k, v in local.items():
                self.global_prefixes.setdefault(k, v)

            m = WHERE_PATTERN.search(query_text)
            if not m:
                s["triples"] = []
                continue
            where_txt = m.group(1)

            flat = []
            for block in _top_level_blocks(where_txt):
                flat.extend(_extract_triples(block))

            expanded = []
            for subj, pred, obj in flat:
                row = []
                for tok in (subj, pred, obj):
                    p, *tail = tok.split(":", 1)
                    if tail and p in prefixes:
                        row.append(f"{prefixes[p]}:{tail[0]}")
                    else:
                        row.append(tok)
                expanded.append(row)
            s["triples"] = expanded
        return samples
    
class VQuandaPreprocessor:
    """
    Normalize VQuanda JSON (array of rows with uid, question, query, …)
    to your QALD-9-shaped records:
      { id, question, formated_query, answers }
    """
    def _load(self, path: str) -> List[Dict[str, Any]]:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("VQuanda input must be a JSON array.")
        return data

    @staticmethod
    def _save(data: List[Dict[str, Any]], path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

    def _normalize(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for r in rows:
            uid       = str(r.get("uid", ""))
            question  = r.get("question", "")
            raw_query = r.get("query", "")

            out.append({
                "id":             uid,
                "question":       question,
                "formated_query": uri_collapse_after_select(raw_query),
                "answers":        [],   # VQuanda doesn’t ship gold bindings/booleans
            })
        return out

    def run(self, in_path: str, out_path: str) -> List[Dict[str, Any]]:
        data = self._normalize(self._load(in_path))
        self._save(data, out_path)
        return data