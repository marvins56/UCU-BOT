# from langchain.document_loaders import (
#     PyPDFLoader,
#     UnstructuredFileLoader,
#     Docx2txtLoader
# )
# from typing import List, Dict
# import os

# class DocumentProcessor:
#     def __init__(self):
#         self.supported_types = {
#             '.pdf': PyPDFLoader,
#             '.txt': UnstructuredFileLoader,
#             '.docx': Docx2txtLoader
#         }
    
#     def process_file(self, file_path: str) -> Dict:
#         # Get file extension
#         ext = os.path.splitext(file_path)[1].lower()
#         if ext not in self.supported_types:
#             raise ValueError(f"Unsupported file type: {ext}")
            
#         # Load and process document
#         loader = self.supported_types[ext](file_path)
#         documents = loader.load()
        
#         return {
#     'documents': documents,
#     'metadata': {
#         'source': file_path,
#         'type': ext,
#         'pages': len(documents)
#     }
# }

from langchain.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredWordDocumentLoader,
)
from typing import List, Dict
import os

class DocumentProcessor:
    def __init__(self):
        self.supported_types = {
            '.pdf': PyPDFLoader,
            '.txt': TextLoader,
            '.docx': UnstructuredWordDocumentLoader,
        }

    def process_file(self, file_path: str) -> Dict:
        # Get file extension
        ext = os.path.splitext(file_path)[1].lower()
        if ext not in self.supported_types:
            raise ValueError(f"Unsupported file type: {ext}")

        # Load and process document
        loader = self.supported_types[ext](file_path)
        documents = loader.load()

        return {
            'documents': documents,
            'metadata': {
                'source': file_path,
                'type': ext,
                'pages': len(documents),
            },
        }