"""
GPT-3.5 Turbo finetuning script using OpenAI API.
Handles file upload, job creation, and monitoring.
"""

import os
import time
import json
import argparse
from pathlib import Path
from typing import Optional, Dict, Any
from openai import OpenAI

try:
    from .gpt_data_prep import prepare_training_jsonl
except ImportError:
    from gpt_data_prep import prepare_training_jsonl


def upload_training_file(client: OpenAI, file_path: str) -> str:
    """
    Upload a training file to OpenAI.
    
    Args:
        client: OpenAI client instance
        file_path: Path to the JSONL training file
    
    Returns:
        File ID from OpenAI
    """
    print(f"Uploading training file: {file_path}")
    with open(file_path, "rb") as f:
        file_obj = client.files.create(
            file=f,
            purpose="fine-tune"
        )
    
    print(f"Uploaded file: {file_obj.id}")
    print(f"File status: {file_obj.status}")
    
    # Wait for file to be processed
    while file_obj.status != "processed":
        if file_obj.status == "error":
            raise RuntimeError(f"File upload failed: {file_obj.status_details}")
        print(f"Waiting for file processing... Status: {file_obj.status}")
        time.sleep(5)
        file_obj = client.files.retrieve(file_obj.id)
    
    print(f"File processed successfully: {file_obj.id}")
    return file_obj.id


def create_finetuning_job(
    client: OpenAI,
    training_file_id: str,
    model: str = "gpt-3.5-turbo",
    validation_file_id: Optional[str] = None,
    hyperparameters: Optional[Dict[str, Any]] = None,
    suffix: Optional[str] = None
) -> str:
    """
    Create a finetuning job.
    
    Args:
        client: OpenAI client instance
        training_file_id: ID of the uploaded training file
        model: Base model to finetune (default: gpt-3.5-turbo)
        validation_file_id: Optional validation file ID
        hyperparameters: Optional hyperparameters dict (e.g., {"n_epochs": 3})
        suffix: Optional suffix for the finetuned model name
    
    Returns:
        Job ID
    """
    job_params = {
        "training_file": training_file_id,
        "model": model
    }
    
    if validation_file_id:
        job_params["validation_file"] = validation_file_id
    
    if hyperparameters:
        job_params["hyperparameters"] = hyperparameters
    
    if suffix:
        job_params["suffix"] = suffix
    
    print(f"Creating finetuning job with parameters: {json.dumps(job_params, indent=2)}")
    job = client.fine_tuning.jobs.create(**job_params)
    
    print(f"Created finetuning job: {job.id}")
    print(f"Job status: {job.status}")
    
    return job.id


def monitor_finetuning_job(
    client: OpenAI,
    job_id: str,
    poll_interval: int = 60,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Monitor a finetuning job until completion.
    
    Args:
        client: OpenAI client instance
        job_id: ID of the finetuning job
        poll_interval: Seconds between status checks
        verbose: Whether to print status updates
    
    Returns:
        Final job object as dict
    """
    print(f"Monitoring finetuning job: {job_id}")
    
    while True:
        job = client.fine_tuning.jobs.retrieve(job_id)
        
        if verbose:
            print(f"Status: {job.status}")
            if hasattr(job, 'trained_tokens') and job.trained_tokens:
                print(f"Trained tokens: {job.trained_tokens}")
            if hasattr(job, 'estimated_finish') and job.estimated_finish:
                print(f"Estimated finish: {job.estimated_finish}")
        
        if job.status == "succeeded":
            print(f"✅ Finetuning completed successfully!")
            print(f"Fine-tuned model: {job.fine_tuned_model}")
            return job.model_dump() if hasattr(job, 'model_dump') else job
        
        elif job.status == "failed":
            error_msg = job.error.message if hasattr(job, 'error') and job.error else "Unknown error"
            print(f"❌ Finetuning failed: {error_msg}")
            raise RuntimeError(f"Finetuning job failed: {error_msg}")
        
        elif job.status in {"validating_files", "queued", "running"}:
            time.sleep(poll_interval)
        
        else:
            print(f"Unknown status: {job.status}")
            time.sleep(poll_interval)


def finetune_gpt(
    input_json_path: str,
    output_jsonl_path: Optional[str] = None,
    model: str = "gpt-3.5-turbo",
    validation_json_path: Optional[str] = None,
    validation_jsonl_path: Optional[str] = None,
    triples_limit: int = 10,
    start_index: int = 0,
    hyperparameters: Optional[Dict[str, Any]] = None,
    suffix: Optional[str] = None,
    knowledge_base: str = "DBpedia",
    skip_preparation: bool = False,
    skip_upload: bool = False,
    training_file_id: Optional[str] = None,
    skip_job_creation: bool = False,
    job_id: Optional[str] = None,
    monitor: bool = True,
    api_key: Optional[str] = None
) -> Dict[str, Any]:
    """
    Complete GPT finetuning pipeline.
    
    Args:
        input_json_path: Path to input JSON with ranked triples
        output_jsonl_path: Path to output JSONL (auto-generated if None)
        model: Base model name
        validation_json_path: Optional validation JSON path
        validation_jsonl_path: Optional validation JSONL path
        triples_limit: Maximum triples per example
        start_index: Start index for training data
        hyperparameters: Finetuning hyperparameters
        suffix: Model name suffix
        knowledge_base: Knowledge base type
        skip_preparation: Skip JSONL preparation
        skip_upload: Skip file upload
        training_file_id: Use existing training file ID
        skip_job_creation: Skip job creation
        job_id: Use existing job ID for monitoring
        monitor: Whether to monitor job until completion
        api_key: OpenAI API key (uses OPENAI_API_KEY env var if None)
    
    Returns:
        Dictionary with job information and model ID
    """
    # Initialize OpenAI client
    if api_key:
        client = OpenAI(api_key=api_key)
    else:
        client = OpenAI()  # Uses OPENAI_API_KEY environment variable
    
    results = {}
    
    # Step 1: Prepare training data
    if not skip_preparation:
        if output_jsonl_path is None:
            input_path = Path(input_json_path)
            output_jsonl_path = str(input_path.parent / f"{input_path.stem}_finetuning.jsonl")
        
        print("=" * 60)
        print("Step 1: Preparing training data")
        print("=" * 60)
        num_examples = prepare_training_jsonl(
            input_path=input_json_path,
            output_path=output_jsonl_path,
            triples_limit=triples_limit,
            start_index=start_index,
            knowledge_base=knowledge_base
        )
        results["training_examples"] = num_examples
        results["training_jsonl_path"] = output_jsonl_path
        
        # Prepare validation data if provided
        if validation_json_path:
            if validation_jsonl_path is None:
                val_path = Path(validation_json_path)
                validation_jsonl_path = str(val_path.parent / f"{val_path.stem}_finetuning.jsonl")
            
            print("\nPreparing validation data...")
            num_val = prepare_training_jsonl(
                input_path=validation_json_path,
                output_path=validation_jsonl_path,
                triples_limit=triples_limit,
                start_index=0,
                knowledge_base=knowledge_base
            )
            results["validation_examples"] = num_val
            results["validation_jsonl_path"] = validation_jsonl_path
    
    # Step 2: Upload training file
    if not skip_upload:
        if training_file_id is None:
            print("\n" + "=" * 60)
            print("Step 2: Uploading training file")
            print("=" * 60)
            training_file_id = upload_training_file(client, output_jsonl_path)
            results["training_file_id"] = training_file_id
            
            # Upload validation file if provided
            if validation_jsonl_path:
                validation_file_id = upload_training_file(client, validation_jsonl_path)
                results["validation_file_id"] = validation_file_id
        else:
            print(f"Using existing training file ID: {training_file_id}")
            results["training_file_id"] = training_file_id
    
    # Step 3: Create finetuning job
    if not skip_job_creation:
        print("\n" + "=" * 60)
        print("Step 3: Creating finetuning job")
        print("=" * 60)
        
        validation_file_id = results.get("validation_file_id")
        job_id = create_finetuning_job(
            client=client,
            training_file_id=training_file_id or results["training_file_id"],
            model=model,
            validation_file_id=validation_file_id,
            hyperparameters=hyperparameters or {"n_epochs": 3},
            suffix=suffix
        )
        results["job_id"] = job_id
    
    # Step 4: Monitor job
    if monitor and (job_id or results.get("job_id")):
        print("\n" + "=" * 60)
        print("Step 4: Monitoring finetuning job")
        print("=" * 60)
        final_job = monitor_finetuning_job(client, job_id or results["job_id"])
        results["fine_tuned_model"] = final_job.get("fine_tuned_model")
        results["job_status"] = final_job.get("status")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Finetune GPT-3.5 Turbo for SPARQL generation")
    parser.add_argument("--input", type=str, required=True, help="Input JSON file with ranked triples")
    parser.add_argument("--output", type=str, default=None, help="Output JSONL path (auto-generated if not provided)")
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo", help="Base model name")
    parser.add_argument("--validation", type=str, default=None, help="Validation JSON file path")
    parser.add_argument("--triples_limit", type=int, default=10, help="Maximum triples per example")
    parser.add_argument("--start_index", type=int, default=0, help="Start index for training data")
    parser.add_argument("--n_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--suffix", type=str, default=None, help="Model name suffix")
    parser.add_argument("--kb", type=str, default="DBpedia", choices=["DBpedia", "Wikidata"], help="Knowledge base")
    parser.add_argument("--skip_prep", action="store_true", help="Skip JSONL preparation")
    parser.add_argument("--skip_upload", action="store_true", help="Skip file upload")
    parser.add_argument("--training_file_id", type=str, default=None, help="Use existing training file ID")
    parser.add_argument("--skip_job", action="store_true", help="Skip job creation")
    parser.add_argument("--job_id", type=str, default=None, help="Use existing job ID for monitoring")
    parser.add_argument("--no_monitor", action="store_true", help="Don't monitor job until completion")
    parser.add_argument("--api_key", type=str, default=None, help="OpenAI API key (or use OPENAI_API_KEY env var)")
    
    args = parser.parse_args()
    
    results = finetune_gpt(
        input_json_path=args.input,
        output_jsonl_path=args.output,
        model=args.model,
        validation_json_path=args.validation,
        triples_limit=args.triples_limit,
        start_index=args.start_index,
        hyperparameters={"n_epochs": args.n_epochs},
        suffix=args.suffix,
        knowledge_base=args.kb,
        skip_preparation=args.skip_prep,
        skip_upload=args.skip_upload,
        training_file_id=args.training_file_id,
        skip_job_creation=args.skip_job,
        job_id=args.job_id,
        monitor=not args.no_monitor,
        api_key=args.api_key
    )
    
    print("\n" + "=" * 60)
    print("Finetuning Summary")
    print("=" * 60)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
