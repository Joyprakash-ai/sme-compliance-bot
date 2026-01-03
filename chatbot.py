
---

## **2. `chatbot.py`**
```python
#!/usr/bin/env python3
"""
Core Compliance Bot implementation.
Uses RAG (Retrieval-Augmented Generation) to answer EU regulation questions.
"""

import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import json
from datetime import datetime

from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

@dataclass
class BotConfig:
    """Configuration for the Compliance Bot."""
    model_name: str = "gpt-4-turbo-preview"
    temperature: float = 0.1  # Low temperature for factual accuracy
    max_tokens: int = 1000
    k_retrieval: int = 5  # Number of documents to retrieve
    persist_directory: str = "./chroma_db"
    use_streaming: bool = False

class ComplianceBot:
    """Main chatbot class for EU regulation queries."""
    
    def __init__(self, config: Optional[BotConfig] = None):
        self.config = config or BotConfig()
        self._init_components()
        self.conversation_history = []
        
    def _init_components(self):
        """Initialize LangChain components."""
        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-ada-002",
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Load vector store
        self.vectorstore = Chroma(
            persist_directory=self.config.persist_directory,
            embedding_function=self.embeddings
        )
        
        # Initialize retriever
        self.retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": self.config.k_retrieval}
        )
        
        # Initialize LLM
        callbacks = [StreamingStdOutCallbackHandler()] if self.config.use_streaming else []
        self.llm = ChatOpenAI(
            model_name=self.config.model_name,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            streaming=self.config.use_streaming,
            callbacks=callbacks,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Custom prompt for legal/regulation answers
        self.prompt_template = PromptTemplate(
            input_variables=["context", "question", "history"],
            template="""You are an expert EU compliance assistant for SMEs. Your answers must be:
1. ACCURATE: Based only on the provided context from EU regulations.
2. PRECISE: Cite specific articles and directives.
3. PRACTICAL: Provide actionable advice for small businesses.
4. CLEAR: Avoid legal jargon when possible.

Previous conversation:
{history}

Relevant EU regulation context:
{context}

Question: {question}

Format your response as:
**Answer:** [Clear, concise answer]
**Applicable Directives:** [List of EU directives/articles]
**Next Steps:** [Practical advice for the SME]
**Disclaimer:** [Standard legal disclaimer]

If the context doesn't contain enough information, say: "I don't have enough specific information about this regulation. Please consult the official [directive name] or a legal professional."

Now answer:"""
        )
        
        # Initialize QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            chain_type_kwargs={
                "prompt": self.prompt_template,
                "memory": ConversationBufferMemory(
                    memory_key="history",
                    input_key="question",
                    return_messages=True
                )
            },
            return_source_documents=True
        )
    
    def ask(self, question: str, country_context: Optional[str] = None) -> Dict[str, Any]:
        """
        Ask a question about EU regulations.
        
        Args:
            question: The regulation question
            country_context: Optional country code for jurisdiction-specific answers
        
        Returns:
            Dictionary with answer, sources, and metadata
        """
        # Enhance question with country context
        enhanced_question = question
        if country_context:
            enhanced_question = f"[Jurisdiction: {country_context.upper()}] {question}"
        
        # Get current timestamp
        timestamp = datetime.utcnow().isoformat()
        
        try:
            # Get answer from QA chain
            result = self.qa_chain({"query": enhanced_question})
            
            # Extract source documents
            sources = []
            if "source_documents" in result:
                for doc in result["source_documents"]:
                    source_info = {
                        "content": doc.page_content[:200] + "...",
                        "metadata": doc.metadata,
                        "directive": doc.metadata.get("directive", "Unknown"),
                        "article": doc.metadata.get("article", "N/A")
                    }
                    sources.append(source_info)
            
            # Parse the response
            answer_text = result["result"]
            
            # Extract sections (simple parsing)
            sections = self._parse_answer_sections(answer_text)
            
            # Prepare response
            response = {
                "question": question,
                "answer": answer_text,
                "sections": sections,
                "sources": sources,
                "metadata": {
                    "timestamp": timestamp,
                    "country_context": country_context,
                    "model": self.config.model_name,
                    "retrieved_docs": len(sources)
                }
            }
            
            # Store in conversation history
            self.conversation_history.append({
                "timestamp": timestamp,
                "question": question,
                "answer_summary": sections.get("Answer", "")[:100] + "...",
                "directives": [s.get("directive", "") for s in sources]
            })
            
            return response
            
        except Exception as e:
            # Fallback response
            return {
                "question": question,
                "answer": f"I encountered an error processing your question: {str(e)}. Please try rephrasing or contact support.",
                "sections": {
                    "Answer": "Error occurred",
                    "Applicable Directives": [],
                    "Next Steps": "Please try again or consult official EU sources."
                },
                "sources": [],
                "metadata": {
                    "timestamp": timestamp,
                    "error": str(e),
                    "model": self.config.model_name
                }
            }
    
    def _parse_answer_sections(self, answer: str) -> Dict[str, str]:
        """Parse the formatted answer into sections."""
        sections = {
            "Answer": "",
            "Applicable Directives": "",
            "Next Steps": "",
            "Disclaimer": ""
        }
        
        current_section = None
        lines = answer.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check for section headers
            if line.startswith("**Answer:**"):
                current_section = "Answer"
                sections[current_section] = line.replace("**Answer:**", "").strip()
            elif line.startswith("**Applicable Directives:**"):
                current_section = "Applicable Directives"
                sections[current_section] = line.replace("**Applicable Directives:**", "").strip()
            elif line.startswith("**Next Steps:**"):
                current_section = "Next Steps"
                sections[current_section] = line.replace("**Next Steps:**", "").strip()
            elif line.startswith("**Disclaimer:**"):
                current_section = "Disclaimer"
                sections[current_section] = line.replace("**Disclaimer:**", "").strip()
            elif current_section and line.startswith("**"):
                # New section found
                current_section = None
            elif current_section:
                # Append to current section
                sections[current_section] += " " + line
        
        # Clean up
        for key in sections:
            sections[key] = sections[key].strip()
        
        return sections
    
    def get_conversation_history(self, limit: int = 10) -> List[Dict]:
        """Get recent conversation history."""
        return self.conversation_history[-limit:]
    
    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []

def main():
    """Command-line interface for the Compliance Bot."""
    import argparse
    
    parser = argparse.ArgumentParser(description="SME Compliance Bot CLI")
    parser.add_argument("--question", "-q", required=True, help="Question about EU regulations")
    parser.add_argument("--country", "-c", help="Country code for jurisdiction context")
    parser.add_argument("--model", default="gpt-4-turbo-preview", help="LLM model to use")
    parser.add_argument("--output", "-o", help="Output file for results (JSON)")
    
    args = parser.parse_args()
    
    # Initialize bot
    config = BotConfig(model_name=args.model)
    bot = ComplianceBot(config)
    
    # Ask question
    print(f"\n🤖 SME Compliance Bot")
    print(f"Question: {args.question}")
    if args.country:
        print(f"Jurisdiction: {args.country.upper()}")
    print("-" * 50)
    
    response = bot.ask(args.question, args.country)
    
    # Print answer
    print("\n📝 Answer:")
    print(response["sections"].get("Answer", response["answer"]))
    
    # Print directives
    if response["sections"].get("Applicable Directives"):
        print(f"\n📚 Applicable Directives:")
        print(response["sections"]["Applicable Directives"])
    
    # Print next steps
    if response["sections"].get("Next Steps"):
        print(f"\n🚀 Next Steps:")
        print(response["sections"]["Next Steps"])
    
    # Print sources
    if response["sources"]:
        print(f"\n🔍 Sources ({len(response['sources'])} retrieved):")
        for i, source in enumerate(response["sources"], 1):
            print(f"{i}. {source['directive']} - Article {source['article']}")
    
    # Save to file if requested
    if args.output:
        import json
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(response, f, indent=2, ensure_ascii=False)
        print(f"\n💾 Results saved to {args.output}")

if __name__ == "__main__":
    main()
