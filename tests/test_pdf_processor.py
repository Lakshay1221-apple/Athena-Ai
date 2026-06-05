from src.dataset_generation.pdf_processor import PDFProcessor
processor = PDFProcessor()

text = processor.extract_text("Data/books/your_book.pdf")


print(text[:1000])

processor.save_text(text,"Data/processed/book_text.txt")
