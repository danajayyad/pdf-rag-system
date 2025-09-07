import PyPDF2
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
from chromadb.utils import embedding_functions
import re
import chromadb


class PDFRAG:
    def __init__(self, pdf_files, embeddings_model_name, generation_model_name):
        self.pdf_files = pdf_files
        self.embeddings_model_name = embeddings_model_name
        self.generation_model_name = generation_model_name
        self.extracted_pages = []
        self.chunks = []
        self.collection = None
        self.generator = None
        
    
    def clean_text(self,text):
        text = re.sub(r"\n+", " " , text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()
    
    
    def extract_and_clean(self):
        '''extract text from pdf, clean text , then store as pafe-level dictionaries'''
        for pdf_file in self.pdf_files:
            with open(pdf_file, 'rb') as pdf:
                reader = PyPDF2.PdfReader(pdf, strict = False)
                for i, page in enumerate(reader.pages):
                    content = page.extract_text()
                    content = self.clean_text(content)
                    self.extracted_pages.append(
                        {
                            "file_name" : pdf_file,
                            "page_number": i+1,
                            "content" : content
                        }
                    )
                
    def chunk_text(self, chunk_size = 200, chunk_overlap = 25):
        '''split extracted pages into smaller chunks with metadata'''
        tokenizer = AutoTokenizer.from_pretrained(self.embeddings_model_name)
        text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            tokenizer=tokenizer,
            chunk_size = chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        all_chunks =[]
        for page in self.extracted_pages:
            page_chunks = text_splitter.split_text(page["content"])
            for chunk in page_chunks:
                all_chunks.append(
                    {
                        "file_name": page["file_name"],
                        "page_number" : page["page_number"],
                        "chunk_content": chunk
                    }
                )
            self.chunks = all_chunks
        
        
    def create_collection(self):
            '''define chroma client, embedding function, chroma collection'''
            chroma_client = chromadb.Client()
            senetence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name = self.embeddings_model_name
            )
            
            self.collection = chroma_client.get_or_create_collection(
                name = "pdf_chunks",
                embedding_function=senetence_transformer_ef
            )

            self.collection.add(
                documents=[chunk["chunk_content"] for chunk in self.chunks],
                metadatas=[  {"file_name" : chunk["file_name"], "page_number": chunk["page_number"]} for chunk in self.chunks],
                ids = [f"chunk_{i+1}" for i in range(len(self.chunks))]
            )
            
    def load_generator(self):
            '''load generator'''
            generation_model_tokenizer = AutoTokenizer.from_pretrained(self.generation_model_name)
            generation_model = AutoModelForSeq2SeqLM.from_pretrained(self.generation_model_name)

            self.generator  = pipeline(
                    "text2text-generation",
                    model = generation_model,
                    tokenizer = generation_model_tokenizer,
                    device_map = "auto"
                )
            
        
    def query_and_generate(self, query, n_results=2):
            '''query collection and generate answer'''
            results = self.collection.query(
                query_texts = query,
                n_results= n_results
            )
            
            '''retrieve content + metadata of the most similar chunks'''
            retrieved_chunks = results["documents"][0]
            retrieved_metadata =  results["metadatas"][0]
            
            
            model_context= " ".join(retrieved_chunks)   
            prompt = f"Answer the following question based on this context: {model_context}\nQuestion:{query}"
            output = self.generator(prompt, max_new_tokens = 400, do_sample = False)
            
            return {
                "answer" : output[0]["generated_text"],
                "chunks": retrieved_chunks,
                "metadata": retrieved_metadata
            }
             

if __name__== '__main__':
   
    file_input = input("Enter PDF files names ---comma separated---")
    pdf_files = [f.strip() for f in file_input.split(",")]
    query = input("Enter your question:")
    
    # intialize RAG system
    rag_system = PDFRAG(pdf_files, "sentence-transformers/all-MiniLM-L6-v2", "google/flan-t5-base")
    rag_system.extract_and_clean()
    rag_system.chunk_text()
    rag_system.create_collection()
    rag_system.load_generator()
    
    # retrieve results
    result = rag_system.query_and_generate(query)



    print("-----RESULTS-----")
    print(f"Q: {query}\nA: {result["answer"]}")


    print("-----REFERENCES-----")
    for chunk, metadata in zip(result["chunks"] , result["metadata"]):
        print(f"File: {metadata["file_name"]} Page: {metadata["page_number"]}")
        print(f"Text: {chunk[:200]}........\n")
