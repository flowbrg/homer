"""
All agents wrappers
"""

from typing import Literal, List

from langchain_core.messages.human import HumanMessage
from langchain_core.messages import AnyMessage

from src.core.configuration import Configuration
from src.core.retrieval_graph import get_retrieval_graph

from typing import Any

class RetrievalAgent:
    """
    Wrapper class for the retrieval agent.
    """
    def __init__(self):
        self._graph = get_retrieval_graph()


    def get_messages(
        self,
        configuration: Configuration,
        thread_id: int
    ) -> list[AnyMessage]:
        config = {"configurable": configuration.asdict() | {"thread_id": str(thread_id)}}
        graph_state = self._graph.get_state(config=config) # Output of get_state is a snapshot state tuple
        messages = graph_state.values["messages"] if "messages" in graph_state.values.keys() else []
        return messages

    def stream(
        self,
        query: str,
        configuration: Configuration,
        thread_id: int
    ) -> str | Any:
        """Invoke the retrieval graph with a query and thread ID.

        Args:
            query (dict[str, Any] | Any): The query to process.
            thread_id (int): The ID of the thread for context.

        Yields:
            str | Any: message chunks of the 'response' node.
        """
        config = {"configurable": configuration.asdict() | {"thread_id": thread_id}}
        for message_chunk, metadata in self._graph.stream(
            input={"messages":[HumanMessage(content=query)]},
            stream_mode="messages",
            config=config,
        ):
            if message_chunk.content and metadata["langgraph_node"] == "respond":
                yield message_chunk.content

from src.core.index_graph import get_index_graph

class IndexAgent:
    """
    Wrapper Class for the index graph.
    """
    def __init__(self):
        self._graph = get_index_graph()

    def invoke(self, path: str):
        self._graph.invoke(input={"path": path})

from src.core.models import load_chat_model, load_embedding_model
from src.core.retrieval import make_retriever

class ReportAgent:
    """
    Wrapper Class for the index graph.
    """
    def __init__(self, configuration: Configuration):
        self._model = load_chat_model(model=configuration.response_model, host= configuration.ollama_host)
        self._embeddings = load_embedding_model(model=configuration.embedding_model, host= configuration.ollama_host)

    def generate_outline(self, main_query: str, max_parts: int = 6) -> List[str]:
        """
        Generate an expert-level outline for a technical report using hybrid retrieval.
        The LLM bases the outline on actual chunks retrieved for the main query.
        """
        # Step 1: Retrieve top document chunks related to the query
        with make_retriever(embedding_model=self._embeddings) as retriever:
                results = retriever.invoke(input=main_query)
        context_text = "\n\n".join([doc.page_content for doc, _ in results])

        # Step 2: Generate outline prompt using retrieved chunks
        outline_prompt = f"""
    You are assisting with the creation of a highly technical report.

    Topic: "{main_query}"

    Below is a corpus of technical context extracted from domain documents:

    {context_text}

    Based on this information, propose exactly {max_parts} section titles for the report.
    The report is for expert engineers, so avoid general sections like "Introduction" or "Overview".
    Each section must reflect a precise technical aspect.

    Return only the section titles. One per line. No bullet points or numbering. No explanations.
    """

        response = self.raw_model_call(outline_prompt)
        outline = self._parse_outline(response)
        return outline[:max_parts]

    def _parse_outline(self, text: str) -> List[str]:
        lines = text.splitlines()
        cleaned = [line.strip("0123456789.:-• ").strip() for line in lines if line.strip()]
        return cleaned

    def generate_report(self, main_query: str, writting: Literal['technical', 'generic'] = 'technical') -> str:
        outline = self.generate_outline(main_query)
        style_label = "Technical Report" if writting == "technical" else "Generic Report"
        full_report = f"{style_label.upper()}\nTITLE: {main_query}\n\n"

        synthesizer = _AnswerSynthesizer(self._model)

        for i, section_title in enumerate(outline, 1):
            print(f"[{i}/{len(outline)}] Writing section: {section_title}")

            if writting == 'generic':
                section_query = f"""
Write an informative and accessible report section titled: "{section_title}"
Main topic: "{main_query}"

Instructions:
- Make it understandable to a general or non-expert audience.
- Avoid excessive technical jargon or heavy formulas.
- Use everyday language, clear structure, and examples where helpful.
- Stay grounded in the provided source content below.
"""
            else:
                section_query = f"""
Assemble a highly technical and detailed report section titled: "{section_title}"
Topic: "{main_query}"

Instructions:
- Use only the retrieved domain-specific content (below) as source material.
- Write in a formal, scientific tone, suitable for expert engineers or researchers.
- Incorporate precise terminology, quantitative references, and technical explanations.
- Avoid generalizations, summaries, or non-technical fluff.

Start writing the section content now.
"""
            with make_retriever(embedding_model=self._embeddings) as retriever:
                raw_results = retriever.invoke(input=section_query)
            all_chunks = [doc.page_content for doc, _ in raw_results]

            raw_text = synthesizer.synthesize(section_query, all_chunks)
            section_text = synthesizer.review(section_title, raw_text, main_query)

            full_report += f"{section_title.upper()}\n\n{section_text}\n\n"

        return full_report
    

class _AnswerSynthesizer:
    def __init__(self, llm):
        self.llm = llm

    def synthesize(self, query: str, documents: list[str]) -> str:
        context = "\n\n".join(documents)
        prompt = f"""Use the context below to answer the question. Combine ideas, analyze, and write a clear, insightful response.

Context:
{context}

Question: {query}

Answer:"""
        return self.llm.invoke(prompt).content.strip()
    
    def review(self, section_title: str, section_text: str, main_query: str) -> str:
        prompt = f"""
    You are a professional technical editor reviewing a draft section titled '{section_title}' in a report on '{main_query}'.

    Your goals:
    - Write fluent, connected, insightful paragraphs
    - **Remove all subheadings, subtitles, and formatting artifacts** like "1.2 Section Title", "Section Title:", or "Title:"
    - Keep the section title (from the report) but remove all internal titles
    - Avoid bullet points or numbered lists
    - Combine short, disjointed sentences into flowing, analytical prose
    - Preserve all technical ideas and data from the draft

    ---

    DRAFT SECTION:
    {section_text}

    ---

    FINAL POLISHED SECTION:
    """
        return self.llm.invoke(prompt).content.strip()