from typing import Dict, List, TypedDict, Annotated, Sequence
from langgraph.graph import Graph, StateGraph
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI
import logging
import json

from config import (
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_API_VERSION,
    AZURE_OPENAI_CHAT_DEPLOYMENT
)
from document_processor import DocumentProcessor
from vector_store import VectorStoreManager

logger = logging.getLogger(__name__)

# Define the state for our graph
class AgentState(TypedDict):
    messages: Annotated[Sequence[HumanMessage | AIMessage], "The messages in the conversation"]
    context: Annotated[List[Dict], "The relevant context chunks"]
    question: Annotated[str, "The user's question"]
    search_query: Annotated[str, "The constructed search query"]

class RAGAgent:
    def __init__(self):
        self.llm = AzureChatOpenAI(
            azure_deployment=AZURE_OPENAI_CHAT_DEPLOYMENT,
            openai_api_version=AZURE_OPENAI_API_VERSION,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_API_KEY,
            model_kwargs={"max_completion_tokens": 1000}
        )
        self.document_processor = DocumentProcessor()
        self.vector_store = VectorStoreManager()
        
        # Initialize the graph
        self.workflow = self._create_workflow()
        
        # Save the graph visualization
        try:
            graph = self.workflow.get_graph(xray=True)
            png_data = graph.draw_mermaid_png()
            with open("graph.png", "wb") as f:
                f.write(png_data)
            logger.info("Graph visualization saved to graph.png")
        except Exception as e:
            logger.error(f"Error saving graph visualization: {e}")

    def _create_workflow(self) -> Graph:
        """Create the LangGraph workflow."""
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("analyze_query", self._analyze_query)
        workflow.add_node("retrieve_context", self._retrieve_context)
        workflow.add_node("check_for_tables", self._check_for_tables)
        workflow.add_node("generate_response", self._generate_response)

        # Add edges
        workflow.add_edge("analyze_query", "retrieve_context")
        workflow.add_edge("retrieve_context", "check_for_tables")
        workflow.add_edge("check_for_tables", "generate_response")

        # Set entry point
        workflow.set_entry_point("analyze_query")

        return workflow.compile()

    def _analyze_query(self, state: AgentState) -> AgentState:
        """Analyze the query to identify tariff names and construct search query."""
        try:
            question = state["question"]
            
            # Create the prompt for query analysis
            analysis_prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a query analysis assistant that helps identify tariff-related information in user questions.
                Your task is to:
                1. Identify if the question is about a specific tariff (e.g., anchorage dues, port charges, etc.)
                2. Extract the relevant tariff name if present
                3. Construct an appropriate search query
                
                Respond in JSON format with the following structure:
                {{
                    "is_tariff_related": boolean,
                    "tariff_name": string or null,
                    "search_query": string
                }}"""),
                ("human", "{question}")
            ])
            
            # Get the analysis
            chain = analysis_prompt | self.llm
            analysis = chain.invoke({"question": question})
            
            # Parse the response
            try:
                analysis_data = json.loads(analysis.content)
                state["search_query"] = analysis_data["search_query"]
                logger.info(f"Query analysis: {analysis_data}")
            except json.JSONDecodeError:
                logger.error("Failed to parse LLM response as JSON")
                state["search_query"] = question  # Fallback to original question
            
            return state
        except Exception as e:
            logger.error(f"Error analyzing query: {e}")
            state["search_query"] = question  # Fallback to original question
            return state

    def _retrieve_context(self, state: AgentState) -> AgentState:
        """Retrieve context from the vector store using the analyzed query."""
        try:
            search_query = state["search_query"]
            context = self.vector_store.search(search_query, k=3)
            state["context"] = context
            return state
        except Exception as e:
            logger.error(f"Error retrieving context: {e}")
            raise

    def _check_for_tables(self, state: AgentState) -> AgentState:
        """Check for table references and retrieve additional context."""
        try:
            context = state["context"]
            additional_context = []
            
            for chunk in context:
                table_refs = self.document_processor.find_table_reference(chunk, context)
                additional_context.extend(table_refs)
            
            # Add unique additional context
            for chunk in additional_context:
                if chunk not in context:
                    context.append(chunk)
            
            state["context"] = context
            return state
        except Exception as e:
            logger.error(f"Error checking for tables: {e}")
            raise

    def _generate_response(self, state: AgentState) -> AgentState:
        """Generate the final response using the LLM."""
        try:
            question = state["question"]
            context = state["context"]
            
            # Format the context
            formatted_context = "\n\n".join([chunk.page_content for chunk in context])
            
            # Dump context to a file for inspection
            with open("temp_context.txt", "w", encoding="utf-8") as f:
                f.write(f"Question: {question}\n\n")
                f.write("Context chunks:\n")
                f.write("-" * 80 + "\n")
                for i, chunk in enumerate(context, 1):
                    f.write(f"\nChunk {i}:\n")
                    f.write("-" * 40 + "\n")
                    f.write(chunk.page_content)
                    f.write("\n" + "-" * 40 + "\n")
                f.write(f"\nTotal context length: {len(formatted_context)} characters")
            
            logger.info(f"Context dumped to temp_context.txt")
            
            # Create the messages
            messages = [
                {"role": "developer", "content": "You are a helpful assistant that answers questions based on the provided context. If you cannot answer based on the context, say so. Always provide a response, even if it's to say you don't have enough information."},
                {"role": "user", "content": f"Context: {formatted_context}\n\nQuestion: {question}"}
            ]
            
            # Generate response using the llm directly
            logger.info("Sending request to Azure OpenAI...")
            response = self.llm.invoke(messages)
            logger.info(f"Received response type: {type(response)}")
            logger.info(f"Response content: {response.content if response else 'None'}")
            
            # Ensure we have a valid response
            if not response:
                raise ValueError("No response object received from the model")
            
            if not hasattr(response, 'content'):
                raise ValueError(f"Response object missing content attribute. Response type: {type(response)}")
            
            if not response.content:
                # If we get an empty response, provide a default message
                response_content = "I apologize, but I couldn't generate a response based on the available context. Please try rephrasing your question or provide more specific details."
            else:
                response_content = response.content
            
            # Update state with response
            state["messages"].append(AIMessage(content=response_content))
            return state
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            # Provide a fallback response in case of error
            state["messages"].append(AIMessage(content="I encountered an error while processing your question. Please try again or rephrase your question."))
            return state

    def process_question(self, question: str) -> str:
        """Process a user question through the workflow."""
        try:
            # Initialize state
            state = AgentState(
                messages=[],
                context=[],
                question=question,
                search_query=""
            )
            
            # Run the workflow
            final_state = self.workflow.invoke(state)
            
            # Return the last message
            return final_state["messages"][-1].content
        except Exception as e:
            logger.error(f"Error processing question: {e}")
            raise 