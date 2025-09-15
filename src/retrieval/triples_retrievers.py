import json, os, re, time, urllib.error
from typing import List, Dict, Any, Tuple
from tqdm import trange
from SPARQLWrapper import SPARQLWrapper, JSON as SPARQLJSON


# ─────────────────────────── PREFIX MAP  (long → short) ─────────────────────────
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
    "http://www.wikidata.org/entity/":                     "wd",
    "http://www.wikidata.org/prop/direct/":                "wdt",
    "http://www.wikidata.org/prop/":                       "p",
    "http://www.wikidata.org/prop/statement/":             "ps",
    "http://www.wikidata.org/prop/statement/value/":       "psv",
    "http://www.wikidata.org/prop/statement/value-normalized/": "psn",
    "http://www.wikidata.org/prop/qualifier/":             "pq",
    "http://www.wikidata.org/prop/qualifier/value/":       "pqv",
    "http://www.wikidata.org/prop/qualifier/value-normalized/": "pqn",
    "http://www.wikidata.org/prop/reference/":             "pr",
    "http://www.wikidata.org/prop/reference/value/":       "prv",
    "http://www.wikidata.org/prop/reference/value-normalized/": "prn",
    "http://www.wikidata.org/prop/direct-normalized/":     "wdtn",
    "http://www.wikidata.org/prop/novalue/":               "wdno",
    "http://www.wikidata.org/entity/statement/":           "wds",
    "http://www.wikidata.org/reference/":                  "wdref",
    "http://www.wikidata.org/value/":                      "wdv",
    "http://www.w3.org/1999/02/22-rdf-syntax-ns#":         "rdf",
    "http://www.w3.org/2000/01/rdf-schema#":               "rdfs",
    "http://www.w3.org/2002/07/owl#":                      "owl",
    "http://www.w3.org/2001/XMLSchema#":                   "xsd",
    "http://www.w3.org/2004/02/skos/core#":                "skos",
    "http://schema.org/":                                   "schema",
    "http://www.opengis.net/ont/geosparql#":               "geo",
    "http://www.w3.org/ns/prov#":                          "prov",
    "http://www.bigdata.com/rdf#":                         "bd",
    "http://www.bigdata.com/queryHints#":                  "hint",
}
_WD_PREFIX_ORDER = sorted(WIKIDATA_PREFIX_MAP.items(), key=lambda kv: -len(kv[0]))

def _wd_shorten(uri: str) -> str:
    """Shorten WD URIs to curies (wd:, wdt:, …). Leaves non-matching strings as-is."""
    u = uri.strip()
    if u.startswith("<") and u.endswith(">"):
        u = u[1:-1]
    for ns, pfx in _WD_PREFIX_ORDER:
        if u.startswith(ns):
            return f"{pfx}:{u[len(ns):].lstrip(':')}"
    return u

def _shorten(uri: str) -> str:
    """Return `prefix:localName` if a namespace matches, else the original URI."""
    # remove surrounding angle-brackets if present
    if uri.startswith("<") and uri.endswith(">"):
        uri = uri[1:-1]

    for ns, p in _PREFIX_ORDER:
        if uri.startswith(ns):
            return f"{p}:{uri[len(ns):].lstrip(':')}"
    return uri   # fallback – no rewrite
# ────────────────────────────────────────────────────────────────────────────────


class DBpediaRetriever:
    """
    Retrieves DBpedia triples for each entity in the dataset, with automatic URI
    shortening (`<http://dbpedia.org/resource/Paris>` → `res:Paris`).
    """

    def __init__(
        self,
        endpoint: str = "https://dbpedia.org/sparql",
        timeout: int = 3000,
        max_retries: int = 5,
        retry_sleep: int = 10,
        checkpoint_file: str = "checkpoint.json",
        remove_checkpoint_on_complete: bool = True,
    ):
        self.endpoint  = endpoint
        self.timeout   = timeout
        self.max_retries = max_retries
        self.retry_sleep = retry_sleep
        self.checkpoint_file = checkpoint_file
        self.remove_checkpoint_on_complete = remove_checkpoint_on_complete

    # ────────────────────────────── public ──────────────────────────────
    def run(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        existing = self._load_checkpoint()
        done_ids = {rec["id"] for rec in existing}
        cache    = {rec["id"]: rec for rec in existing}

        for i in trange(len(data), desc="Retrieving DBpedia triples"):
            sample = data[i]
            sid    = sample["id"]

            if sid in done_ids:        # resume from checkpoint
                data[i]["retrieved_triples"] = cache[sid]["retrieved_triples"]
                continue

            triples_by_entity = []
            for ent in sample.get("entities", []):
                cleaned = self._clean_uri(ent)
                if not cleaned:
                    triples_by_entity.append([("SKIPPED", "SKIPPED", _shorten(ent))])
                    continue
                triples = self._fetch_dbpedia_triples(cleaned)
                triples_by_entity.append(triples)

            sample["retrieved_triples"] = triples_by_entity

            cache[sid] = {
                "id": sid,
                "retrieved_triples": triples_by_entity,
            }
            done_ids.add(sid)
            if i % 10 == 0:
                self._save_checkpoint(list(cache.values()))

        self._save_checkpoint(list(cache.values()))
        if self.remove_checkpoint_on_complete and os.path.exists(self.checkpoint_file):
            os.remove(self.checkpoint_file)
        return data
    # ────────────────────────────────────────────────────────────────────

    # --------------------------- low-level helpers -----------------------------
    def _fetch_dbpedia_triples(self, entity_uri: str) -> List[Tuple[str, str, str]]:
        sparql = SPARQLWrapper(self.endpoint)
        sparql.setTimeout(self.timeout)
        q = f"""
            SELECT ?subject ?predicate ?object WHERE {{
              {{ <http://dbpedia.org/resource/{entity_uri}> ?predicate ?object .
                 FILTER(STRSTARTS(STR(?object), "http://dbpedia.org/resource/")) }}
              UNION
              {{ ?subject ?predicate <http://dbpedia.org/resource/{entity_uri}> .
                 FILTER(STRSTARTS(STR(?subject), "http://dbpedia.org/resource/")) }}
            }}
        """
        sparql.setQuery(q)
        sparql.setReturnFormat(SPARQLJSON)

        for attempt in range(self.max_retries):
            try:
                res = sparql.query().convert()
                break
            except (urllib.error.HTTPError, urllib.error.URLError) as e:
                print(f"[Retry {attempt+1}/{self.max_retries}] {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_sleep)
                else:
                    return [("SKIPPED", "SKIPPED", _shorten(entity_uri))]
            except Exception as e:
                print(f"SPARQL error: {e}")
                return [("SKIPPED", "SKIPPED", _shorten(entity_uri))]

        triples = []
        for b in res["results"]["bindings"]:
            subj = b.get("subject",  {"value": f"http://dbpedia.org/resource/{entity_uri}"} )["value"]
            pred = b.get("predicate",{"value": "UNKNOWN"})["value"]
            obj  = b.get("object",  {"value": f"http://dbpedia.org/resource/{entity_uri}"} )["value"]
            triples.append((_shorten(subj), _shorten(pred), _shorten(obj)))
        return triples

    def _clean_uri(self, entity: str) -> str:
        entity = re.sub(r"[^\w\s-]", "", entity).replace(" ", "_")
        return entity.strip()

    # checkpoint helpers --------------------------------------------------------
    def _load_checkpoint(self) -> List[Dict[str, Any]]:
        try:
            with open(self.checkpoint_file, encoding="utf-8") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return []

    def _save_checkpoint(self, results: List[Dict[str, Any]]) -> None:
        with open(self.checkpoint_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4)


# ===================== Second-Stage Retriever (ranked entities) =================
class RankedEntityDBpediaRetriever(DBpediaRetriever):
    def __init__(
        self,
        endpoint: str = "https://dbpedia.org/sparql",
        timeout: int = 3000,
        max_retries: int = 5,
        retry_sleep: int = 10,
        checkpoint_file: str = "stage2_checkpoint.json",
        remove_checkpoint_on_complete: bool = True,
        input_field: str = "ranked_entities",
        output_field: str = "retrieved_ranked_triples",
    ):
        super().__init__(
            endpoint, timeout, max_retries, retry_sleep,
            checkpoint_file, remove_checkpoint_on_complete
        )
        self.input_field  = input_field
        self.output_field = output_field

    def run(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        existing = self._load_checkpoint()
        done_ids = {rec["id"] for rec in existing}
        cache    = {rec["id"]: rec for rec in existing}

        for i in trange(len(data),
                        desc=f"Retrieving triples for '{self.input_field}'"):
            samp, sid = data[i], data[i]["id"]

            if sid in done_ids:
                data[i][self.output_field] = cache[sid][self.output_field]
                continue

            triples_by_ent = []
            for ent in samp.get(self.input_field, []):
                cleaned = self._clean_uri(ent)
                if not cleaned:
                    triples_by_ent.append([("SKIPPED", "SKIPPED", _shorten(ent))])
                    continue
                triples_by_ent.append(self._fetch_dbpedia_triples(cleaned))

            samp[self.output_field] = triples_by_ent
            cache[sid] = {"id": sid, self.output_field: triples_by_ent}
            done_ids.add(sid)

            if i % 10 == 0:
                self._save_checkpoint(list(cache.values()))

        self._save_checkpoint(list(cache.values()))
        if self.remove_checkpoint_on_complete and os.path.exists(self.checkpoint_file):
            os.remove(self.checkpoint_file)
        return data
    
# ===================== Wikidata Retriever =================

class WikidataRetriever(DBpediaRetriever):

    def __init__(
        self,
        endpoint: str = "https://query.wikidata.org/sparql",
        timeout: int = 3000,
        max_retries: int = 5,
        retry_sleep: int = 10,
        checkpoint_file: str = "wikidata_checkpoint.json",
        remove_checkpoint_on_complete: bool = True,
        input_field: str = "entities",
        output_field: str = "retrieved_triples"
    ):
        super().__init__(endpoint, timeout, max_retries, retry_sleep, checkpoint_file, remove_checkpoint_on_complete)
        self.input_field = input_field
        self.output_field = output_field

    def run(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        existing = self._load_checkpoint()
        done_ids = {rec["id"] for rec in existing}
        cache    = {rec["id"]: rec for rec in existing}

        for i in trange(len(data), desc="Retrieving Wikidata triples"):
            sample = data[i]
            sid    = sample["id"]

            if sid in done_ids:
                data[i][self.output_field] = cache[sid][self.output_field]
                continue

            triples_by_entity = []
            for qid in sample.get(self.input_field, []):
                triples = self._fetch_wikidata_triples(qid)
                triples_by_entity.append(triples)

            sample[self.output_field] = triples_by_entity
            cache[sid] = {"id": sid, self.output_field: triples_by_entity}
            done_ids.add(sid)

            if i % 10 == 0:
                self._save_checkpoint(list(cache.values()))

        self._save_checkpoint(list(cache.values()))
        if self.remove_checkpoint_on_complete and os.path.exists(self.checkpoint_file):
            os.remove(self.checkpoint_file)
        return data

    def _fetch_wikidata_triples(self, qid: str) -> List[Tuple[str, str, str]]:
        sparql = SPARQLWrapper(self.endpoint)
        sparql.setTimeout(self.timeout)

        query = f"""
        SELECT ?s ?p ?o WHERE {{
          {{ wd:{qid} ?p ?o .
             FILTER(STRSTARTS(STR(?p), "http://www.wikidata.org/prop")) }}
          UNION
          {{ ?s ?p wd:{qid} .
             FILTER(STRSTARTS(STR(?p), "http://www.wikidata.org/prop")) }}
        }}
        """

        sparql.setQuery(query)
        sparql.setReturnFormat(SPARQLJSON)

        for attempt in range(self.max_retries):
            try:
                res = sparql.query().convert()
                break
            except (urllib.error.HTTPError, urllib.error.URLError) as e:
                print(f"[Retry {attempt+1}/{self.max_retries}] {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_sleep)
                else:
                    return [("SKIPPED", "SKIPPED", qid)]
            except Exception as e:
                print(f"SPARQL error: {e}")
                return [("SKIPPED", "SKIPPED", qid)]

        triples = []
        for b in res["results"]["bindings"]:
            subj = b.get("s", {"value": f"wd:{qid}"}).get("value")
            pred = b.get("p", {"value": "UNKNOWN"}).get("value")
            obj  = b.get("o", {"value": f"wd:{qid}"}).get("value")
            triples.append((subj, pred, obj))
        return triples
    


class QALDWikidataRetriever:
    def __init__(
        self,
        endpoint: str = "https://query.wikidata.org/sparql",
        timeout: int = 60,
        max_retries: int = 5,
        retry_sleep: int = 10,
        checkpoint_file: str = "checkpoint_wd.json",
        remove_checkpoint_on_complete: bool = True,
        user_agent: str = "QALD-Retriever/1.0 (contact: you@example.com)"
    ):
        self.endpoint = endpoint
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_sleep = retry_sleep
        self.checkpoint_file = checkpoint_file
        self.remove_checkpoint_on_complete = remove_checkpoint_on_complete
        self.user_agent = user_agent

    # ────────────────────────────── public ──────────────────────────────
    def run(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        existing = self._load_checkpoint()
        done_ids = {rec["id"] for rec in existing}
        cache    = {rec["id"]: rec for rec in existing}

        for i in trange(len(data), desc="Retrieving Wikidata triples"):
            sample = data[i]
            sid    = sample["id"]

            if sid in done_ids:  # resume
                data[i]["retrieved_triples"] = cache[sid]["retrieved_triples"]
                continue

            triples_by_entity: List[List[Tuple[str, str, str]]] = []
            for ent in sample.get("entities", []):
                qid = self._qid_from_any(ent)
                if not qid:
                    triples_by_entity.append([("SKIPPED", "SKIPPED", _wd_shorten(str(ent)))])
                    continue
                triples = self._fetch_wikidata_triples(qid)
                triples_by_entity.append(triples)

            sample["retrieved_triples"] = triples_by_entity
            cache[sid] = {"id": sid, "retrieved_triples": triples_by_entity}
            done_ids.add(sid)

            if i % 10 == 0:
                self._save_checkpoint(list(cache.values()))

        self._save_checkpoint(list(cache.values()))
        if self.remove_checkpoint_on_complete and os.path.exists(self.checkpoint_file):
            os.remove(self.checkpoint_file)
        return data

    # --------------------------- low-level helpers -----------------------------
    QID_RX   = re.compile(r"^Q\d+$")
    WD_RX    = re.compile(r"^wd:Q\d+$")
    URI_RX   = re.compile(r"^https?://www\.wikidata\.org/entity/(Q\d+)$")

    def _qid_from_any(self, s: str) -> str:
        """Accept 'Q42', 'wd:Q42', or 'http://www.wikidata.org/entity/Q42' → 'Q42'."""
        s = (s or "").strip()
        if self.QID_RX.match(s):
            return s
        if self.WD_RX.match(s):
            return s.split(":")[1]
        m = self.URI_RX.match(s)
        if m:
            return m.group(1)
        return ""

    def _fetch_wikidata_triples(self, qid: str) -> List[Tuple[str, str, str]]:
        # Use direct properties (wdt:) both outbound and inbound.
        q = f"""
        PREFIX wd:  <http://www.wikidata.org/entity/>
        PREFIX wdt: <http://www.wikidata.org/prop/direct/>

        SELECT ?s ?p ?o WHERE {{
        {{ BIND(wd:{qid} AS ?s)
            ?s ?p ?o .
            FILTER(STRSTARTS(STR(?p), "http://www.wikidata.org/prop/direct/")) }}
        UNION
        {{ ?s ?p wd:{qid} .
            FILTER(STRSTARTS(STR(?p), "http://www.wikidata.org/prop/direct/")) }}
        }}
        """
        sparql = SPARQLWrapper(self.endpoint)
        sparql.setTimeout(self.timeout)
        sparql.setQuery(q)
        sparql.setReturnFormat(SPARQLJSON)
        try:
            sparql.setAgent(self.user_agent)
        except Exception:
            pass

        for attempt in range(self.max_retries):
            try:
                res = sparql.query().convert()
                break
            except (urllib.error.HTTPError, urllib.error.URLError) as e:
                print(f"[Retry {attempt+1}/{self.max_retries}] {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_sleep)
                else:
                    return [("SKIPPED", "SKIPPED", f"wd:{qid}")]
            except Exception as e:
                print(f"SPARQL error: {e}")
                return [("SKIPPED", "SKIPPED", f"wd:{qid}")]

        triples: List[Tuple[str, str, str]] = []
        for b in res["results"]["bindings"]:
            subj = b.get("s", {}).get("value", f"http://www.wikidata.org/entity/{qid}")
            pred = b.get("p", {}).get("value", "UNKNOWN")
            obj  = b.get("o", {}).get("value", f"http://www.wikidata.org/entity/{qid}")
            triples.append((_wd_shorten(subj), _wd_shorten(pred), _wd_shorten(obj)))
        return triples

    # checkpoint helpers --------------------------------------------------------
    def _load_checkpoint(self) -> List[Dict[str, Any]]:
        try:
            with open(self.checkpoint_file, encoding="utf-8") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return []

    def _save_checkpoint(self, results: List[Dict[str, Any]]) -> None:
        with open(self.checkpoint_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4)