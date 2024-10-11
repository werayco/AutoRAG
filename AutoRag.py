from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain.vectorstores import FAISS
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.output_parsers import (
    StructuredOutputParser,
    ResponseSchema,
    CommaSeparatedListOutputParser,
)
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter

class AutoRAG:
    def __init__(self, path_to_doc, chunk_size: int, chunk_overlap: int,model,temp):
        self.path_to_doc = path_to_doc
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model = model
        self.temp = temp
        self.llm = ChatOllama(model=self.model, temperature=self.temp)


    def parser(self, country, gender, language, term):
        schemas = [
            ResponseSchema(
                name="name", description="this field is for the name of the entity"
            ),
            ResponseSchema(
                name="dept",
                description="this is the department of the entity",
                type="string",
            ),
            ResponseSchema(name="ID", description="this serves as the primary key"),
        ]

        # Structured Output Parser
        parser = StructuredOutputParser.from_response_schemas(schemas)
        format_instruction = parser.get_format_instructions()

        template = """ Kindly Help Summarize the history of richest {gender} in {Country},
        {format_instruction} """

        pmpt = ChatPromptTemplate(
            messages=[
                ("system", "You are an AI assistant who specialies in healthcare"),
                ("user", template),
            ],
            input_variables=["gender","country"],
            partial_variables={"format_instruction": format_instruction},
        )
        msg = pmpt.format_messages(gender=gender, country=country)
        response = msg | self.llm  # | means a chain
        structured_parsed_op = parser.parse(response)

        # CommaSeparatedList Output_Parser
        comma_parser = CommaSeparatedListOutputParser()

        template_02 = """ Explain the term: {term} in {language}
        {format_instructions}"""
        format_struc = (
            comma_parser.get_format_instructions()
        )  # this is an instruction telling the llm to return a list

        pmp = ChatPromptTemplate(
            messages=[
                ("system", "You are an AI assistant who knows stuff"),
                ("user", template_02),
            ],
            input_variables=["term", "language"],
            partial_variable={"format_instruction": format_struc},
        )

        messg = pmp.format_messages(term=term, language=language)
        comma_response_llm = comma_parser.parse(self.llm(messg))
        return {"StructuredOutputJson":structured_parsed_op, "ListOutput":comma_response_llm}

    def RAGProcess(self, input) :
        
        pdf = UnstructuredPDFLoader(self.path_to_doc)
        documents = pdf.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )
        text = splitter.split_documents(documents=documents)

        embedder = HuggingFaceEmbeddings()
        faiss_db = FAISS.from_documents(text, embedder)
        retriever = faiss_db.as_retriever(
            search_type="similarity_score_threshold", search_kwargs={"k": 6, "lambda_mult": 0.25}
        )

        query_temp = PromptTemplate.from_template("""You are an AI assistant. Your task is to generate 5 different version
                                                  of the user's query and make similary serach with the vector database provided {question}""")
        
        retriever_ = MultiQueryRetriever.from_llm(retriever=retriever,llm=self.llm,prompt=query_temp)

        template = """ Answer the question bases on the following context:
        {context}
        question: {question}
        """
        prompt = ChatPromptTemplate.from_template(template)
    
        chain = ({"context":retriever_, "question": RunnablePassthrough()} | prompt | self.llm | StrOutputParser() )

        response = chain.invoke(input)

        return {"response":response}

if __name__ == "__main__":
    autorag = AutoRAG(model="nemotron-mini",chunk_size=100, chunk_overlap=10,path_to_doc="your_path_file_path_goes_here.pdf",temp=0.7)
