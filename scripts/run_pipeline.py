#!/usr/bin/env python3
"""
Main pipeline orchestrator for SPARQL generation.

This script runs the complete pipeline:
1. Preprocessing
2. First Stage Retrieval
3. Ranking (ColBERT)
4. Finetuning
5. Initial SPARQL Generation
6. Dynamic Few-Shot Selection
7. Chain of Thought Generation
8. Final SPARQL Generation
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, List

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from preprocessing.data_processing import (
    QALDPreprocessor, SparqlParser, LCQAPreprocessor,
    VQuandaPreprocessor
)
from retrieval.entity_extractors import RefinedEntityExtractor
from retrieval.triples_retrievers import DBpediaRetriever
from refined.inference.processor import Refined


class PipelineOrchestrator:
    """Orchestrates the complete SPARQL generation pipeline."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.dataset_name = config.get("dataset", "qald")
        self.split = config.get("split", "train")
        
    def run_preprocessing(self) -> List[Dict[str, Any]]:
        """Stage 1: Preprocessing"""
        print("=" * 60)
        print("Stage 1: Preprocessing")
        print("=" * 60)
        
        input_path = self.config["paths"]["raw_data"]
        output_path = self.config["paths"]["preprocessed_data"]
        
        if self.dataset_name == "qald":
            preprocessor = QALDPreprocessor(include_all_langs=False)
        elif self.dataset_name == "lcquad":
            preprocessor = LCQAPreprocessor()
        elif self.dataset_name == "vquanda":
            preprocessor = VQuandaPreprocessor()
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")
        
        data = preprocessor.run(input_path, output_path)
        print(f"✅ Preprocessed {len(data)} examples")
        return data
    
    def run_sparql_parsing(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Stage 1.5: Parse SPARQL queries to extract triples"""
        print("\n" + "=" * 60)
        print("Stage 1.5: SPARQL Parsing")
        print("=" * 60)
        
        parser = SparqlParser()
        data = parser.run(data)
        
        output_path = self.config["paths"]["parsed_data"]
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Parsed SPARQL for {len(data)} examples")
        return data
    
    def run_entity_extraction(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Stage 2: Entity Extraction"""
        print("\n" + "=" * 60)
        print("Stage 2: Entity Extraction")
        print("=" * 60)
        
        refined_model = Refined.from_pretrained(
            model_name='wikipedia_model',
            entity_set='wikipedia'
        )
        extractor = RefinedEntityExtractor(refined_model)
        data = extractor.run(data)
        
        output_path = self.config["paths"]["entities_data"]
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Extracted entities for {len(data)} examples")
        return data
    
    def run_triple_retrieval(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Stage 3: Triple Retrieval"""
        print("\n" + "=" * 60)
        print("Stage 3: Triple Retrieval")
        print("=" * 60)
        
        retriever = DBpediaRetriever(
            endpoint=self.config.get("retrieval", {}).get("endpoint", "https://dbpedia.org/sparql"),
            timeout=self.config.get("retrieval", {}).get("timeout", 3000),
            max_retries=self.config.get("retrieval", {}).get("max_retries", 5),
            retry_sleep=self.config.get("retrieval", {}).get("retry_sleep", 10),
            checkpoint_file=self.config.get("retrieval", {}).get("checkpoint_file", "checkpoint.json")
        )
        data = retriever.run(data)
        
        output_path = self.config["paths"]["retrieved_data"]
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Retrieved triples for {len(data)} examples")
        return data
    
    def run_ranking(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Stage 4: ColBERT Ranking"""
        print("\n" + "=" * 60)
        print("Stage 4: ColBERT Ranking")
        print("=" * 60)
        
        # Import ranking module
        from ranking.colbert_wrapper import rank_triples
        
        data = rank_triples(
            data,
            collection_path=self.config["paths"]["collection_tsv"],
            index_path=self.config["paths"]["colbert_index"],
            top_k=self.config.get("ranking", {}).get("top_k", 200),
            experiment_name=self.config.get("ranking", {}).get("experiment_name", "colbert_ranking"),
            root_dir=self.config.get("ranking", {}).get("root_dir", "./experiments"),
            nbits=self.config.get("ranking", {}).get("nbits", 2),
            overwrite_index=self.config.get("ranking", {}).get("overwrite_index", False)
        )
        
        output_path = self.config["paths"]["ranked_data"]
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Ranked triples for {len(data)} examples")
        return data
    
    def run_full_pipeline(self, stages: List[str] = None):
        """Run the complete pipeline or specified stages."""
        if stages is None:
            stages = [
                "preprocessing", "sparql_parsing", "entity_extraction",
                "triple_retrieval", "ranking"
            ]
        
        data = None
        
        if "preprocessing" in stages:
            data = self.run_preprocessing()
        
        if "sparql_parsing" in stages and data is not None:
            data = self.run_sparql_parsing(data)
        
        if "entity_extraction" in stages and data is not None:
            data = self.run_entity_extraction(data)
        
        if "triple_retrieval" in stages and data is not None:
            data = self.run_triple_retrieval(data)
        
        if "ranking" in stages and data is not None:
            data = self.run_ranking(data)
        
        print("\n" + "=" * 60)
        print("Pipeline completed successfully!")
        print("=" * 60)
        
        return data


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Run SPARQL generation pipeline")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration JSON file"
    )
    parser.add_argument(
        "--stages",
        type=str,
        nargs="+",
        help="Specific stages to run (default: all)",
        choices=["preprocessing", "sparql_parsing", "entity_extraction", 
                 "triple_retrieval", "ranking"]
    )
    
    args = parser.parse_args()
    
    config = load_config(args.config)
    orchestrator = PipelineOrchestrator(config)
    orchestrator.run_full_pipeline(stages=args.stages)


if __name__ == "__main__":
    main()
