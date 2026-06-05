import argparse
import sys
import json
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

from src.dataset_generation.pdf_processor import PDFProcessor
from src.dataset_generation.text_cleaner import TextCleaner
from src.dataset_generation.chunker import Chunker
from src.dataset_generation.gemma_generator import GemmaGenerator
from src.dataset_generation.json_validator import JSONValidator
from src.dataset_generation.deduplicator import Deduplicator
from src.dataset_generation.dataset_writer import DatasetWriter

def parse_args():
    parser = argparse.ArgumentParser(description="ML Teacher Assistant - Dataset Generation Module MVP")
    parser.add_argument(
        "--pdf-path",
        type=str,
        default=None,
        help="Path to textbook PDF. If not specified, the first PDF in Data/books/ will be used."
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="gemma4:e4b",
        help="Ollama model name to use (default: gemma4:e4b)"
    )
    parser.add_argument(
        "--limit-chunks",
        type=int,
        default=None,
        help="Limit the number of chunks to process (useful for MVP tests)"
    )
    parser.add_argument(
        "--output-version",
        type=str,
        default="v1",
        help="Version identifier for this run (default: v1)"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=750,
        help="Target chunk size in words (default: 750)"
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=75,
        help="Target chunk overlap in words (default: 75)"
    )
    parser.add_argument(
        "--margin-top",
        type=float,
        default=50.0,
        help="PDF top margin for headers removal (default: 50.0)"
    )
    parser.add_argument(
        "--margin-bottom",
        type=float,
        default=55.0,
        help="PDF bottom margin for footers removal (default: 55.0)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="Data/datasets",
        help="Base output directory for datasets (default: Data/datasets)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume dataset generation from the last saved checkpoint"
    )
    return parser.parse_args()

def find_default_pdf() -> str:
    books_dir = Path("Data/books")
    if not books_dir.exists():
        raise FileNotFoundError(f"Books directory not found at: {books_dir}")
        
    pdfs = list(books_dir.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No PDF books found in {books_dir}")
        
    return str(pdfs[0])

def log_generation(chunk_id: str, concept: str, status: str, retry_count: int, reason: str = ""):
    """Log generation progress to logs/generation.log."""
    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / "generation.log"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_line = f"[{timestamp}] [INFO] chunk_id: {chunk_id} | concept: {concept} | status: {status} | retries: {retry_count}"
    if reason:
        log_line += f" | reason: {reason}"
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(log_line + "\n")

def load_existing_examples(jsonl_path: Path) -> List[Dict[str, Any]]:
    """Load already generated examples from the JSONL output file."""
    examples = []
    if jsonl_path.exists():
        try:
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        examples.append(json.loads(line))
        except Exception as e:
            print(f"[Warning] Failed to read existing JSONL: {e}")
    return examples

def format_time(seconds: float) -> str:
    """Format seconds to h m s or m s."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}h {m}m {s}s"
    return f"{m}m {s}s"

def main():
    args = parse_args()
    
    # Resolve PDF path
    try:
        if args.pdf_path is None:
            pdf_path = find_default_pdf()
            print(f"No PDF path specified. Defaulting to first found book: {pdf_path}")
        else:
            pdf_path = args.pdf_path
    except Exception as e:
        print(f"[Fatal Error] {str(e)}")
        sys.exit(1)

    print(f"\n==================================================")
    print(f"   ML Teacher Assistant - Dataset Generator MVP   ")
    print(f"==================================================")
    print(f"PDF Path:       {pdf_path}")
    print(f"Ollama Model:   {args.model_name}")
    print(f"Output Version: {args.output_version}")
    print(f"Chunk Config:   Size={args.chunk_size}, Overlap={args.chunk_overlap}")
    print(f"PDF Margins:    Top={args.margin_top}, Bottom={args.margin_bottom}")
    print(f"Limit Chunks:   {args.limit_chunks if args.limit_chunks is not None else 'All'}")
    print(f"Resume Flag:    {args.resume}")
    print(f"==================================================\n")

    # Initialize components
    print("Initializing components...")
    processor = PDFProcessor(margin_top=args.margin_top, margin_bottom=args.margin_bottom)
    cleaner = TextCleaner()
    chunker = Chunker(chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
    generator = GemmaGenerator(model_name=args.model_name)
    validator = JSONValidator()
    deduplicator = Deduplicator()
    writer = DatasetWriter(output_base_dir=args.output_dir)

    # Step 1: PDF Processing
    print("\n--- Step 1: Processing PDF & Extracting Blocks ---")
    try:
        segments = processor.extract_segments(pdf_path)
        print(f"Extracted {len(segments)} blocks/paragraphs from PDF.")
    except Exception as e:
        print(f"[Fatal Error] PDF extraction failed: {str(e)}")
        sys.exit(1)

    # Step 2: Text Cleaning
    print("\n--- Step 2: Cleaning Text segments ---")
    cleaned_segments = []
    for idx, seg in enumerate(segments):
        cleaned_text = cleaner.clean_segment(seg["text"])
        if cleaned_text:
            seg["text"] = cleaned_text
            cleaned_segments.append(seg)
    print(f"Cleaned and normalized text. Kept {len(cleaned_segments)} non-empty segments.")

    # Step 3: Semantic Chunking
    print("\n--- Step 3: Chunking Segments ---")
    chunks = chunker.create_chunks(cleaned_segments)
    total_available_chunks = len(chunks)
    print(f"Generated {total_available_chunks} chunks.")

    # Subset chunks if limit specified
    if args.limit_chunks is not None:
        chunks = chunks[:args.limit_chunks]
        print(f"Limiting generation to the first {len(chunks)} chunks.")

    if not chunks:
        print("[Error] No chunks available for processing. Exiting.")
        sys.exit(1)

    # Load Resume State if requested
    completed_chunk_ids = []
    valid_examples = []
    failed_chunks_dict = {}

    version_dir = Path(args.output_dir) / args.output_version
    jsonl_path = version_dir / "ml_teacher_dataset.jsonl"

    if args.resume:
        print("\n--- Resuming Dataset Generation ---")
        completed_chunk_ids = writer.load_checkpoint(args.output_version)
        print(f"Found {len(completed_chunk_ids)} completed chunk IDs in checkpoint.")
        
        # Load existing failed chunks
        failed_records = writer.load_failed_chunks(args.output_version)
        for fail in failed_records:
            failed_chunks_dict[fail["chunk_id"]] = fail["reason"]
            
        # Re-populate examples and deduplicator
        valid_examples = load_existing_examples(jsonl_path)
        print(f"Loaded {len(valid_examples)} previously generated examples from JSONL.")
        for ex in valid_examples:
            deduplicator.is_duplicate(ex["instruction"]) # populate deduplicator seen hashes
        print("Pre-seeded deduplicator cache with existing instructions.")

    # Step 4: Gemma Generation Loop
    print("\n--- Step 4: Running Dataset Generation ---")
    total_chunks = len(chunks)
    
    # Track stats
    # Filter completed/failed from count
    processed_chunks = len(valid_examples)
    failed_chunks = len(failed_chunks_dict)
    duplicates_removed = 0
    generated_examples = len(valid_examples)

    # Metrics for progress reporting
    session_start_time = time.time()
    session_processed_count = 0

    for idx, chunk in enumerate(chunks):
        chunk_id = chunk["chunk_id"]
        source_chunk = chunk["source_chunk"]
        chapter = chunk["chapter"]
        source_book = chunk["source_book"]
        
        # Skip if already completed (succeeded or failed)
        if chunk_id in completed_chunk_ids:
            print(f"[{idx+1}/{total_chunks}] Skipping completed chunk {chunk_id}.")
            continue
            
        if chunk_id in failed_chunks_dict:
            print(f"[{idx+1}/{total_chunks}] Skipping previously failed chunk {chunk_id}.")
            continue

        print(f"[{idx+1}/{total_chunks}] Processing {chunk_id} (Pages {chunk['start_page']}-{chunk['end_page']})...")
        
        # 4a. Query Gemma via Ollama
        example = generator.generate_example(source_chunk)
        if not example:
            reason = "Ollama error or JSON parse failure"
            print(f"  ❌ Generation failed ({reason}). Skipping.")
            failed_chunks += 1
            failed_chunks_dict[chunk_id] = reason
            writer.write_failed_chunk(chunk_id, reason, args.output_version)
            log_generation(chunk_id, "Unknown", "FAILED", generator.num_retries, reason)
            
            # Checkpoint the failed chunk so we don't block
            completed_chunk_ids.append(chunk_id)
            writer.write_checkpoint(completed_chunk_ids, args.output_version)
            
            session_processed_count += 1
            continue

        concept = example.get("concept", "Unknown")

        # 4b. Validate output schema & difficulty
        validated = validator.validate_and_normalize(example)
        if not validated:
            reason = "Validation failed (missing fields or invalid difficulty)"
            print(f"  ❌ Validation failed ({reason}). Skipping.")
            failed_chunks += 1
            failed_chunks_dict[chunk_id] = reason
            writer.write_failed_chunk(chunk_id, reason, args.output_version)
            log_generation(chunk_id, concept, "INVALID", generator.num_retries, reason)
            
            completed_chunk_ids.append(chunk_id)
            writer.write_checkpoint(completed_chunk_ids, args.output_version)
            
            session_processed_count += 1
            continue

        generated_examples += 1

        # 4c. Hashing deduplication check
        if deduplicator.is_duplicate(validated["instruction"]):
            reason = f"Duplicate instruction detected for '{validated['concept']}'"
            print(f"  ⚠️ {reason}. Skipping.")
            duplicates_removed += 1
            log_generation(chunk_id, concept, "DUPLICATE", generator.num_retries, reason)
            
            completed_chunk_ids.append(chunk_id)
            writer.write_checkpoint(completed_chunk_ids, args.output_version)
            
            session_processed_count += 1
            continue

        # 4d. Add metadata context for auditing
        validated["source_book"] = source_book
        validated["chapter"] = chapter
        validated["chunk_id"] = chunk_id
        validated["source_chunk"] = source_chunk

        # Incremental Save (Append to JSONL and update checkpoint)
        writer.append_example(validated, args.output_version)
        completed_chunk_ids.append(chunk_id)
        writer.write_checkpoint(completed_chunk_ids, args.output_version)

        valid_examples.append(validated)
        processed_chunks += 1
        log_generation(chunk_id, concept, "SUCCESS", generator.num_retries)
        
        print(f"  ✅ Success: Generated example for concept '{validated['concept']}' [{validated['difficulty']}].")
        
        session_processed_count += 1
        
        # Display progress every 10 processed chunks in this session
        if session_processed_count > 0 and session_processed_count % 10 == 0:
            elapsed = time.time() - session_start_time
            # total processed including current session
            finished_total = len(completed_chunk_ids)
            percent = (finished_total / total_chunks) * 100
            
            avg_time = elapsed / session_processed_count
            remaining_chunks = total_chunks - finished_total
            eta = avg_time * remaining_chunks
            
            print(f"\n==================================================")
            print(f"   [PROGRESS REPORT] {percent:.1f}% Complete ({finished_total}/{total_chunks})")
            print(f"   Elapsed Time:     {format_time(elapsed)}")
            print(f"   ETA Remaining:    {format_time(eta)}")
            print(f"==================================================\n")

    # Step 5: Save Final Dataset and Report
    print("\n--- Step 5: Saving Dataset & Report ---")
    if valid_examples:
        # Write final consolidated JSON and report
        paths = writer.write_dataset(valid_examples, args.output_version)
        writer.write_statistics(valid_examples, args.output_version)
        
        book_name = Path(pdf_path).stem.replace("_", " ").replace("-", " ")
        report_data = {
            "book_name": book_name,
            "total_chunks": total_available_chunks,
            "processed_chunks": processed_chunks,
            "failed_chunks": failed_chunks,
            "generated_examples": generated_examples,
            "duplicates_removed": duplicates_removed,
            "final_examples": len(valid_examples),
            "model_used": args.model_name,
            "timestamp": datetime.now().isoformat()
        }
        writer.write_report(report_data, args.output_version)
        
        # Print run summary
        print(f"\n==================================================")
        print(f"                 RUN SUMMARY                      ")
        print(f"==================================================")
        print(f"Total Chunks Checked:    {total_chunks}")
        print(f"Successfully Processed:  {processed_chunks}")
        print(f"Failed Generations:      {failed_chunks}")
        print(f"Duplicates Removed:      {duplicates_removed}")
        print(f"Final Examples Written:  {len(valid_examples)}")
        print(f"Outputs written under:   {Path(args.output_dir) / args.output_version}")
        print(f"==================================================\n")
    else:
        print("[Error] No valid examples generated in this run. Dataset was not saved.")

if __name__ == "__main__":
    main()
