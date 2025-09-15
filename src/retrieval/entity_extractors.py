from typing import List, Dict, Any
from tqdm import tqdm

class EntityExtractorBase:
    def extract_entities(self, text: str) -> List[str]:
        raise NotImplementedError("Subclasses must implement this method.")

    def run(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        for i in tqdm(range(len(data)), desc=f"Entity Extraction with {self.__class__.__name__}"):
            question_text = data[i].get('question', '')
            entities = self.extract_entities(question_text)
            data[i]['entities'] = entities
        return data


# ------------------------ Refined Approach ------------------------
class RefinedEntityExtractor(EntityExtractorBase):
    def __init__(self, refined_model):
        super().__init__()
        self.refined_model = refined_model

    def extract_entities(self, text: str) -> List[str]:
        spans = self.refined_model.process_text(text)
        entities = [
            span.predicted_entity.wikipedia_entity_title.replace(' ', '_')
            for span in spans if span.predicted_entity.wikipedia_entity_title
        ]
        return entities


# ------------------------ FALCON Approach ------------------------
class FalconEntityExtractor(EntityExtractorBase):

    def __init__(self, api_url: str = "https://labs.tib.eu/falcon/api?mode=long",
                 delay: float = 0.5):
        super().__init__()
        self.api_url = api_url
        self.delay = delay
        self.headers = {'Content-Type': 'application/json'}

    def extract_entities(self, text: str) -> List[str]:
        import requests, time

        payload = {"text": text}
        try:
            response = requests.post(self.api_url, headers=self.headers, json=payload, timeout=30)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            print(f"Falcon extraction error for '{text[:50]}...' -> {e}")
            return []

        raw_entities = data.get('entities', [])
        uris = []
        if raw_entities:
            if isinstance(raw_entities[0], dict):
                uris = [ent.get("URI") for ent in raw_entities if ent.get("URI")]
            elif all(isinstance(ent, str) for ent in raw_entities):
                uris = raw_entities

        entities = []
        for uri in uris:
            if uri and "/" in uri:
                entities.append(uri.split('/')[-1])
            else:
                entities.append(uri)

        time.sleep(self.delay)
        return entities
    
    
    # ------------------------ Refined Wikidata Approach ------------------------
class RefinedWikidataEntityExtractor(EntityExtractorBase):
    def __init__(
        self,
        refined_model,
        output_format: str = "qid",
        dedupe: bool = True,
        return_meta: bool = False,
    ):
        super().__init__()
        self.refined_model = refined_model
        self.output_format = output_format
        self.dedupe = dedupe
        self.return_meta = return_meta

    @staticmethod
    def _fmt(qid: str, fmt: str) -> str:
        if fmt == "wd":
            return f"wd:{qid}"
        if fmt == "uri":
            return f"http://www.wikidata.org/entity/{qid}"
        return qid  # "qid"

    def extract_entities(self, text: str):
        spans = self.refined_model.process_text(text)

        seen = set()
        out_simple = []
        out_meta = []

        for span in spans:
            ent = getattr(span, "predicted_entity", None)
            qid = getattr(ent, "wikidata_entity_id", None) if ent else None
            if not qid:
                continue  # skip NIL

            if self.dedupe and qid in seen:
                continue
            seen.add(qid)

            formatted = self._fmt(qid, self.output_format)

            if self.return_meta:
                mention = getattr(span, "text", None) or getattr(span, "mention", None)
                score   = getattr(ent, "confidence", None) or getattr(ent, "score", None)
                start   = getattr(span, "start", None)
                end     = getattr(span, "end", None)
                out_meta.append(
                    {"mention": mention, "qid": qid, "value": formatted,
                     "score": score, "start": start, "end": end}
                )
            else:
                out_simple.append(formatted)

        return out_meta if self.return_meta else out_simple

    def run(
        self,
        data, 
        question_key: str = "question",
        entity_key: str = "entities",
        meta_key: str = "entities_meta",
    ):
        """
        Writes:
          - data[i][entity_key] = List[str]   (QIDs or wd:/URI per output_format)
          - if return_meta=True: data[i][meta_key] = List[dict] with rich span info
        Returns the same list it receives (mutates in place).
        """
        for item in data:
            text = item.get(question_key, "")
            ents = self.extract_entities(text)
            if self.return_meta:
                item[meta_key] = ents
                item[entity_key] = [m["value"] for m in ents]
            else:
                item[entity_key] = ents
        return data