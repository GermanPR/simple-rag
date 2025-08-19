"""Prompt templates for different query intents and generation tasks."""

from typing import List, Dict, Any
from app.logic.intent import QueryIntent


class PromptTemplate:
    """Base class for prompt templates."""
    
    def format(self, **kwargs) -> str:
        """Format the prompt template with provided arguments."""
        raise NotImplementedError


class QAPromptTemplate(PromptTemplate):
    """Standard Q&A prompt template."""
    
    def __init__(self):
        self.system_prompt = """You are a helpful assistant that answers questions based strictly on the provided context. 

IMPORTANT RULES:
1. Answer ONLY based on the provided context
2. Include citations in the format [filename p.X] for each claim
3. If you cannot answer from the context, respond with "Insufficient evidence to answer this question"
4. Do not use your general knowledge - only use the provided context
5. Be concise and accurate"""

        self.user_template = """Context:
{context}

Question: {query}

Answer with citations in the format [filename p.X]:"""
    
    def format(self, query: str, context: str) -> List[Dict[str, str]]:
        """Format as messages for chat completion."""
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.user_template.format(query=query, context=context)}
        ]


class StructuredPromptTemplate(PromptTemplate):
    """Template for structured responses (lists, steps, etc.)."""
    
    def __init__(self):
        self.system_prompt = """You are a helpful assistant that provides structured answers based strictly on the provided context.

IMPORTANT RULES:
1. Answer ONLY based on the provided context
2. Use bullet points, numbered lists, or structured format as appropriate
3. Include citations in the format [filename p.X] for each point
4. If you cannot answer from the context, respond with "Insufficient evidence to answer this question"
5. Do not use your general knowledge - only use the provided context"""

        self.user_template = """Context:
{context}

Question: {query}

Provide a structured answer with citations [filename p.X]:"""
    
    def format(self, query: str, context: str) -> List[Dict[str, str]]:
        """Format as messages for chat completion."""
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.user_template.format(query=query, context=context)}
        ]


class ComparisonPromptTemplate(PromptTemplate):
    """Template for comparison-focused responses."""
    
    def __init__(self):
        self.system_prompt = """You are a helpful assistant that provides comparative analysis based strictly on the provided context.

IMPORTANT RULES:
1. Answer ONLY based on the provided context
2. Structure your comparison clearly with distinct points
3. Include citations in the format [filename p.X] for each comparison point
4. If you cannot make the comparison from the context, respond with "Insufficient evidence to answer this question"
5. Do not use your general knowledge - only use the provided context"""

        self.user_template = """Context:
{context}

Comparison request: {query}

Provide a structured comparison with citations [filename p.X]:"""
    
    def format(self, query: str, context: str) -> List[Dict[str, str]]:
        """Format as messages for chat completion."""
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.user_template.format(query=query, context=context)}
        ]


class SummaryPromptTemplate(PromptTemplate):
    """Template for summary responses."""
    
    def __init__(self):
        self.system_prompt = """You are a helpful assistant that provides concise summaries based strictly on the provided context.

IMPORTANT RULES:
1. Summarize ONLY based on the provided context
2. Focus on the most important points
3. Include citations in the format [filename p.X] for key points
4. If you cannot summarize from the context, respond with "Insufficient evidence to answer this question"
5. Do not use your general knowledge - only use the provided context"""

        self.user_template = """Context:
{context}

Summary request: {query}

Provide a concise summary with citations [filename p.X]:"""
    
    def format(self, query: str, context: str) -> List[Dict[str, str]]:
        """Format as messages for chat completion."""
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.user_template.format(query=query, context=context)}
        ]


class PromptManager:
    """Manages prompt templates for different intents."""
    
    def __init__(self):
        self.templates = {
            QueryIntent.QA: QAPromptTemplate(),
            QueryIntent.STRUCTURED: StructuredPromptTemplate(),
            QueryIntent.COMPARISON: ComparisonPromptTemplate(),
            QueryIntent.SUMMARY: SummaryPromptTemplate(),
        }
        # Default to QA template for unknown intents
        self.default_template = QAPromptTemplate()
    
    def get_prompt(self, intent: QueryIntent, query: str, context: str) -> List[Dict[str, str]]:
        """
        Get formatted prompt messages for the given intent.
        
        Args:
            intent: Detected query intent
            query: User query
            context: Retrieved context text
            
        Returns:
            List of message dictionaries for chat completion
        """
        template = self.templates.get(intent, self.default_template)
        return template.format(query=query, context=context)
    
    def format_context(self, chunks_info: List[Dict[str, Any]]) -> str:
        """
        Format retrieved chunks into context string.
        
        Args:
            chunks_info: List of chunk dictionaries with text, filename, page
            
        Returns:
            Formatted context string
        """
        if not chunks_info:
            return ""
        
        context_parts = []
        for i, chunk in enumerate(chunks_info, 1):
            text = chunk.get("text", "")
            filename = chunk.get("filename", "unknown")
            page = chunk.get("page", 0)
            
            # Limit chunk text length for context
            if len(text) > 500:
                text = text[:497] + "..."
            
            context_parts.append(f"[{i}] From {filename} p.{page}: {text}")
        
        return "\n\n".join(context_parts)


# Global prompt manager instance
prompt_manager = PromptManager()