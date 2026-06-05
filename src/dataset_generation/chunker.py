from typing import List, Dict, Any

class Chunker:
    def __init__(self, chunk_size: int = 750, chunk_overlap: int = 75):
        """
        Initialize Chunker.

        Args:
            chunk_size (int): Target word count per chunk.
            chunk_overlap (int): Target overlap word count.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def _get_word_count(self, text: str) -> int:
        return len(text.split())

    def create_chunks(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Group segments into chunks of chunk_size words with chunk_overlap words.
        Ensures segments are kept intact.

        Args:
            segments (List[Dict[str, Any]]): List of segments from PDFProcessor.

        Returns:
            List[Dict[str, Any]]: List of chunks with metadata.
        """
        if not segments:
            return []

        chunks = []
        buffer = []
        buffer_words = 0
        chunk_idx = 1

        i = 0
        while i < len(segments):
            seg = segments[i]
            seg_words = self._get_word_count(seg["text"])

            # Add to buffer
            buffer.append(seg)
            buffer_words += seg_words

            # Check if buffer has reached or exceeded the chunk size
            # If the next segment exists, we check if adding it would put us way over,
            # or if we are already in the target range (between chunk_size and chunk_size + max(overlap, 100))
            is_last = (i == len(segments) - 1)
            
            # If buffer size is large enough, or it's the last element
            if buffer_words >= self.chunk_size or is_last:
                # Emit chunk
                chunk_text = "\n\n".join(s["text"] for s in buffer)
                
                # Determine metadata from segments in buffer
                source_book = buffer[0]["source_book"]
                # Use the chapter of the last segment in the buffer as the primary chapter
                chapter = buffer[-1]["chapter"]
                start_page = buffer[0]["page"]
                end_page = buffer[-1]["page"]
                
                chunks.append({
                    "chunk_id": f"chunk_{chunk_idx:03d}",
                    "source_chunk": chunk_text,
                    "source_book": source_book,
                    "chapter": chapter,
                    "start_page": start_page,
                    "end_page": end_page
                })
                chunk_idx += 1

                if is_last:
                    break

                # Prepare buffer for next chunk using overlap
                # We want to backtrack and find the starting index for the next chunk
                # that gives us roughly chunk_overlap words of overlap.
                overlap_words = 0
                overlap_count = 0
                for j in range(len(buffer) - 1, -1, -1):
                    w = self._get_word_count(buffer[j]["text"])
                    if overlap_words + w <= self.chunk_overlap or overlap_count == 0:
                        overlap_words += w
                        overlap_count += 1
                    else:
                        break
                
                # The next chunk starts at the current index i minus the overlap_count plus 1
                # (so we repeat the last overlap_count segments).
                # But to avoid infinite loops, if overlap_count equals the entire buffer size,
                # we must advance by at least 1 segment.
                if overlap_count >= len(buffer):
                    overlap_count = len(buffer) - 1
                
                if overlap_count > 0:
                    # Clear buffer and re-initialize it with the overlapping segments
                    overlap_segs = buffer[-overlap_count:]
                    buffer = list(overlap_segs)
                    buffer_words = sum(self._get_word_count(s["text"]) for s in buffer)
                else:
                    buffer = []
                    buffer_words = 0

            i += 1

        return chunks
