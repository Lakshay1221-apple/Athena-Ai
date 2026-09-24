"""Dataset generation orchestration CLI for Athena AI."""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from src.common.config import DATASET_GEN_CONFIG, GEN_MODEL_CONFIG, PATHS
from src.common.logger import get_logger
from src.dataset_generation.chunker import Chunker
from src.dataset_generation.dataset_writer import DatasetWriter
from src.dataset_generation.deduplicator import Deduplicator
from src.dataset_generation.gemma_generator import GemmaGenerator
from src.dataset_generation.json_validator import JSONValidator
from src.dataset_generation.pdf_processor import PDFProcessor
from src.dataset_generation.text_cleaner import TextCleaner

logger = get_logger("dataset_generation")


def parse_args():
    parser = argparse.ArgumentParser(description="ML Teacher Assistant - Dataset Generation Pipeline")
    parser.add_argument(
        "--pdf-path",
        type=str,
        default=None,
        help="Path to textbook PDF. If not specified, the first PDF in Data/books/ will be used.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=GEN_MODEL_CONFIG.ollama_model,
        help=f"Ollama model name to use (default: {GEN_MODEL_CONFIG.ollama_model})",
    )
    parser.add_argument(
        "--limit-chunks",
        type=int,
        default=None,
        help="Limit the number of chunks to process (useful for test runs)",
    )
    parser.add_argument(
        "--output-version",
        type=str,
        default=DATASET_GEN_CONFIG.output_version,
        help="Version identifier for this run (default: v1)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DATASET_GEN_CONFIG.chunk_size,
        help="Target chunk size in words (default: 750)",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=DATASET_GEN_CONFIG.chunk_overlap,
        help="Target chunk overlap in words (default: 75)",
    )
    parser.add_argument(
        "--margin-top",
        type=float,
        default=DATASET_GEN_CONFIG.margin_top,
        help="PDF top margin for headers removal (default: 50.0)",
    )
    parser.add_argument(
        "--margin-bottom",
        type=float,
        default=DATASET_GEN_CONFIG.margin_bottom,
        help="PDF bottom margin for footers removal (default: 55.0)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PATHS.datasets_dir),
        help="Base output directory for datasets (default: Data/datasets)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume dataset generation from the last saved checkpoint",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration, PDF extraction, and Ollama connection without running full generation",
    )
    return parser.parse_args()


def find_default_pdf() -> str:
    books_dir = PATHS.books_dir
    if not books_dir.exists():
        raise FileNotFoundError(f"Books directory not found at: {books_dir}")

    pdfs = list(books_dir.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No PDF books found in {books_dir}")

    # Prioritize default book
    for p in pdfs:
        if "HandsonMachine-Learning" in p.name:
            return str(p)
    return str(pdfs[0])


def load_existing_examples(jsonl_path: Path) -> List[Dict[str, Any]]:
    examples = []
    if jsonl_path.exists():
        try:
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        examples.append(json.loads(line))
        except Exception as e:
            logger.warning(f"Failed to read existing JSONL: {e}")
    return examples


def format_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}h {m}m {s}s"
    return f"{m}m {s}s"


def main():
    args = parse_args()

    try:
        pdf_path = args.pdf_path or find_default_pdf()
    except Exception as e:
        logger.error(f"Fatal Error locating source PDF: {str(e)}")
        sys.exit(1)

    print("\n" + "=" * 50)
    print("   ML Teacher Assistant - Dataset Generator")
    print("=" * 50)
    print(f"PDF Path:       {pdf_path}")
    print(f"Ollama Model:   {args.model_name}")
    print(f"Output Version: {args.output_version}")
    print(f"Chunk Config:   Size={args.chunk_size}, Overlap={args.chunk_overlap}")
    print(f"PDF Margins:    Top={args.margin_top}, Bottom={args.margin_bottom}")
    print(f"Limit Chunks:   {args.limit_chunks if args.limit_chunks is not None else 'All'}")
    print(f"Resume Flag:    {args.resume}")
    print(f"Dry Run:        {args.dry_run}")
    print("=" * 50 + "\n")

    # Dry run check
    if args.dry_run:
        logger.info("[DRY RUN] Validating prerequisites...")
        if not Path(pdf_path).exists():
            logger.error(f"[DRY RUN] PDF does not exist at {pdf_path}")
            sys.exit(1)
        logger.info(f"[DRY RUN] PDF verified: {pdf_path}")

        ollama_ok = GemmaGenerator.check_ollama_availability(args.model_name)
        if not ollama_ok:
            logger.warning(f"[DRY RUN] Ollama check returned false for model '{args.model_name}'")
        else:
            logger.info(f"[DRY RUN] Ollama connection and model '{args.model_name}' verified!")

        processor = PDFProcessor(margin_top=args.margin_top, margin_bottom=args.margin_bottom)
        cleaner = TextCleaner()
        chunker = Chunker(chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)

        segments = processor.extract_segments(pdf_path)
        logger.info(f"[DRY RUN] Extracted {len(segments)} segments from PDF.")
        chunks = chunker.create_chunks(segments)
        logger.info(f"[DRY RUN] Created {len(chunks)} chunks.")
        print("\n✅ DRY RUN SUCCESSFUL — Configuration, PDF extraction, and chunking verified.\n")
        return

    # Initialize components
    logger.info("Initializing dataset generation pipeline components...")
    processor = PDFProcessor(margin_top=args.margin_top, margin_bottom=args.margin_bottom)
    cleaner = TextCleaner()
    chunker = Chunker(chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
    generator = GemmaGenerator(model_name=args.model_name)
    validator = JSONValidator(min_response_len=DATASET_GEN_CONFIG.min_response_length)
    deduplicator = Deduplicator()
    writer = DatasetWriter(output_base_dir=args.output_dir)

    # Step 1: PDF Processing
    logger.info("Step 1: Extracting segments from PDF...")
    try:
        segments = processor.extract_segments(pdf_path)
        logger.info(f"Extracted {len(segments)} blocks/paragraphs from PDF.")
    except Exception as e:
        logger.error(f"PDF extraction failed: {str(e)}")
        sys.exit(1)

    # Step 2: Text Cleaning
    logger.info("Step 2: Cleaning and normalizing text segments...")
    cleaned_segments = []
    for seg in segments:
        cleaned_text = cleaner.clean_segment(seg["text"])
        if cleaned_text:
            seg["text"] = cleaned_text
            cleaned_segments.append(seg)
    logger.info(f"Cleaned and normalized text. Kept {len(cleaned_segments)} segments.")

    # Step 3: Semantic Chunking
    logger.info("Step 3: Chunking segments...")
    chunks = chunker.create_chunks(cleaned_segments)
    total_available_chunks = len(chunks)
    logger.info(f"Generated {total_available_chunks} total chunks.")

    if args.limit_chunks is not None:
        chunks = chunks[:args.limit_chunks]
        logger.info(f"Limiting generation to first {len(chunks)} chunks.")

    if not chunks:
        logger.error("No chunks available for processing. Exiting.")
        sys.exit(1)

    # Load Resume State if requested
    completed_chunk_ids = []
    valid_examples = []
    failed_chunks_dict = {}

    version_dir = Path(args.output_dir) / args.output_version
    jsonl_path = version_dir / "ml_teacher_dataset.jsonl"

    if args.resume:
        logger.info("Loading previous checkpoint...")
        completed_chunk_ids = writer.load_checkpoint(args.output_version)
        logger.info(f"Found {len(completed_chunk_ids)} completed chunk IDs in checkpoint.")

        failed_records = writer.load_failed_chunks(args.output_version)
        for fail in failed_records:
            failed_chunks_dict[fail["chunk_id"]] = fail["reason"]

        valid_examples = load_existing_examples(jsonl_path)
        logger.info(f"Loaded {len(valid_examples)} previously generated examples from JSONL.")
        for ex in valid_examples:
            deduplicator.is_duplicate(ex.get("instruction", ""))
        logger.info("Pre-seeded deduplicator cache with existing instructions.")

    # Step 4: Generation Loop
    logger.info("Step 4: Running LLM generation loop...")
    total_chunks = len(chunks)
    processed_chunks = len(valid_examples)
    failed_chunks = len(failed_chunks_dict)
    duplicates_removed = 0
    generated_examples = len(valid_examples)

    session_start_time = time.time()
    session_processed_count = 0

    for idx, chunk in enumerate(chunks):
        chunk_id = chunk["chunk_id"]
        source_chunk = chunk["source_chunk"]
        chapter = chunk["chapter"]
        source_book = chunk["source_book"]

        if chunk_id in completed_chunk_ids:
            continue

        if chunk_id in failed_chunks_dict:
            continue

        logger.info(f"[{idx+1}/{total_chunks}] Processing {chunk_id} (Pages {chunk['start_page']}-{chunk['end_page']})...")

        example = generator.generate_example(source_chunk)
        if not example:
            reason = "Ollama error or JSON parse failure"
            logger.warning(f"  ❌ Generation failed for {chunk_id}: {reason}")
            failed_chunks += 1
            failed_chunks_dict[chunk_id] = reason
            writer.write_failed_chunk(chunk_id, reason, args.output_version)
            completed_chunk_ids.append(chunk_id)
            writer.write_checkpoint(completed_chunk_ids, args.output_version)
            session_processed_count += 1
            continue

        concept = example.get("concept", "Unknown")
        validated = validator.validate_and_normalize(example)
        if not validated:
            reason = "Validation failed (missing fields or invalid difficulty)"
            logger.warning(f"  ❌ Validation failed for {chunk_id}: {reason}")
            failed_chunks += 1
            failed_chunks_dict[chunk_id] = reason
            writer.write_failed_chunk(chunk_id, reason, args.output_version)
            completed_chunk_ids.append(chunk_id)
            writer.write_checkpoint(completed_chunk_ids, args.output_version)
            session_processed_count += 1
            continue

        generated_examples += 1

        if deduplicator.is_duplicate(validated["instruction"]):
            reason = f"Duplicate instruction detected for '{validated['concept']}'"
            logger.warning(f"  ⚠️ {reason}. Skipping.")
            duplicates_removed += 1
            completed_chunk_ids.append(chunk_id)
            writer.write_checkpoint(completed_chunk_ids, args.output_version)
            session_processed_count += 1
            continue

        validated["source_book"] = source_book
        validated["chapter"] = chapter
        validated["chunk_id"] = chunk_id
        validated["source_chunk"] = source_chunk

        # Incremental Save
        writer.append_example(validated, args.output_version)
        completed_chunk_ids.append(chunk_id)
        writer.write_checkpoint(completed_chunk_ids, args.output_version)

        valid_examples.append(validated)
        processed_chunks += 1
        logger.info(f"  ✅ Success: Generated example for '{validated['concept']}' [{validated['difficulty']}].")

        session_processed_count += 1

    # Step 5: Save Final Dataset and Report
    logger.info("Step 5: Saving consolidated dataset & statistics...")
    if valid_examples:
        writer.write_dataset(valid_examples, args.output_version)
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
            "timestamp": datetime.now().isoformat(),
        }
        writer.write_report(report_data, args.output_version)
        logger.info(f"Pipeline complete! {len(valid_examples)} examples written under {Path(args.output_dir) / args.output_version}")
    else:
        logger.error("No valid examples generated. Dataset was not saved.")


if __name__ == "__main__":
    main()
