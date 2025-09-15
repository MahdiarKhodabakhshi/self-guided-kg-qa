import json
import os
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Indexer, Searcher

def main():
    input_json = "/home/m2khoda/dual_retriever/datasets/vquanda/vquanda_test_retrieved_triples.json"
    output_json = "/home/m2khoda/dual_retriever/datasets/vquanda/vquanda_test_ranked.json"

    with open(input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    collection_tsv = "/home/m2khoda/dual_retriever/datasets/lcquad/first_stage_ranker/vquanda_test_ranker_collection.tsv"
    os.makedirs(os.path.dirname(collection_tsv), exist_ok=True)

    doc_id = 0
    doc_id_to_triple = {}
    doc_id_to_question_id = {}
    with open(collection_tsv, "w", encoding="utf-8") as fout:
        for entry in data:
            q_id = entry["id"]
            question = entry["question"]

            retrieved_lists = entry.get("retrieved_triples", [])
            if not retrieved_lists:
                continue

            all_triples_for_this_question = []
            for triple_chunk in retrieved_lists:
                if (isinstance(triple_chunk, list)
                    and len(triple_chunk) > 0
                    and isinstance(triple_chunk[0], list)):
                    all_triples_for_this_question.extend(triple_chunk)
                else:
                    all_triples_for_this_question.extend(retrieved_lists)
                    break

            for triple in all_triples_for_this_question:
                triple_str = " ".join(str(x) for x in triple)

                doc_id_str = str(doc_id)
                fout.write(f"{doc_id_str}\t{triple_str}\n")

                doc_id_to_triple[doc_id_str] = triple
                doc_id_to_question_id[doc_id_str] = q_id

                doc_id += 1

    # --------------------------------------------------------------------------
    # 3. Index the collection using ColBERT
    # --------------------------------------------------------------------------
    with Run().context(RunConfig(nranks=1, experiment="vquanda_experiment_triples_formated")):
        config = ColBERTConfig(
            nbits=2,
            root="/home/m2khoda/dual_retriever/experiments",
        )

        indexer = Indexer(
            checkpoint="colbertv2.0",
            config=config
        )

        indexer.index(
            name="vquanda_triples.nbits=2",
            collection=collection_tsv,
            overwrite="force_silent_overwrite"
        )

        # ----------------------------------------------------------------------
        # 4. Create a Searcher and rank the triples for each question
        # ----------------------------------------------------------------------
        experiment = "vquanda_experiment_triples_formated"
        index_name = "vquanda_triples.nbits=2"

        index_path = os.path.join(config.root, experiment, "indexes", index_name)

        searcher = Searcher(
            index=index_path,
            collection=collection_tsv,
            config=config
        )

        # For each question, rank its candidate triples
        for entry in data:
            q_text = entry["question"]
            q_id = entry["id"]
            # print(f"Processing question {q_id}: {q_text}")

            top_k = 200
            # ColBERT v2 typically returns (doc_ids, ranks, scores)
            doc_ids, ranks, scores = searcher.search(q_text, k=top_k)

            # Zip them into (doc_id, original_rank, score) tuples
            paired_results = list(zip(doc_ids, ranks, scores))
            # print(f"Raw ColBERT results: {paired_results}")

            # Filter to keep only doc_ids that match current question
            # Convert did_int -> str because we stored doc_id as string
            relevant_results = []
            for (did_int, rank_int, score_float) in paired_results:
                did_str = str(did_int)
                if doc_id_to_question_id.get(did_str) == q_id:
                    # We'll store the raw rank_int and the score for further sorting
                    relevant_results.append((did_str, rank_int, score_float))

            # Now you can sort them by score descending
            # Notice that rank_int is the "original" rank from ColBERT’s retrieval
            relevant_results.sort(key=lambda x: x[2], reverse=True)

            # Build the final list of ranked triples, including both original rank and final rank
            ranked_triples = []
            for final_rank, (did, original_rank, score) in enumerate(relevant_results, start=1):
                triple = doc_id_to_triple[did]
                ranked_triples.append({
                    "triple": triple,
                    "score": score,
                    "original_colbert_rank": original_rank,
                    "final_rank": final_rank
                })

            # Attach the final ranked list to the entry
            entry["retrived_triples_ranked"] = ranked_triples

    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Done! Wrote ranked triples to {output_json}")


if __name__ == '__main__':
    main()