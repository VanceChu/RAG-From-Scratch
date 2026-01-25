"""
Script to test document parsing and chunking in isolation.
Usage: python scripts/test_parser.py /path/to/document
"""

import argparse
import sys
import json
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag_core.ragflow_pipeline import parse_and_split_document
from rag_core.config import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP

def main():
    parser = argparse.ArgumentParser(description="Test RAG document parsing.")
    parser.add_argument("path", help="Path to the document to parse")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE, help="Chunk token size")
    parser.add_argument("--overlap", type=int, default=DEFAULT_CHUNK_OVERLAP, help="Chunk overlap")
    parser.add_argument("--output", help="Path to save the parsed result as JSON")
    args = parser.parse_args()

    doc_path = Path(args.path).expanduser().resolve()
    if not doc_path.exists():
        print(f"Error: File not found: {doc_path}")
        sys.exit(1)

    print(f"Parsing file: {doc_path}")
    print(f"Config: chunk_size={args.chunk_size}, overlap={args.overlap}")
    print("-" * 60)

    try:
        chunks = parse_and_split_document(
            path=doc_path,
            chunk_token_size=args.chunk_size,
            overlap_tokens=args.overlap
        )
        
        print(f"Successfully parsed {len(chunks)} chunks.\n")
        
        # Prepare serializable result
        output_data = []
        for i, chunk in enumerate(chunks):
            metadata = chunk.get("metadata", {})
            text = chunk.get("text", "")
            
            chunk_data = {
                "chunk_index": i,
                "type": metadata.get('doc_type', 'unknown'),
                "positions": metadata.get('positions', []),
                "image_path": metadata.get('image_path'),
                "content": text
            }
            output_data.append(chunk_data)

            print(f"--- Chunk {i} ---")
            print(f"Type: {chunk_data['type']}")
            preview_len = 200
            preview = text[:preview_len].replace("\n", " ") + "..." if len(text) > preview_len else text
            print(f"Content Preview: {preview}")
            print("\n")

        # Save to file if requested
        if args.output:
            output_path = Path(args.output).resolve()
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            print(f"-" * 60)
            print(f"Full parsing result saved to: {output_path}")

    except Exception as e:
        print(f"Parsing failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
