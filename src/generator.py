"""
Generation module using BioMistral-7B (4-bit quantized) for RAG answer generation.
Takes retrieved context and produces grounded yes/no/maybe answers with explanations.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from typing import List, Optional


SYSTEM_PROMPT = """You are a medical QA assistant. Given a medical question and relevant PubMed abstracts as context, provide:
1. A final answer: yes, no, or maybe
2. A brief explanation grounded in the provided evidence

Only use information from the provided context. If the context is insufficient, say "maybe" and explain what information is missing."""

RAG_PROMPT_TEMPLATE = """Context (retrieved PubMed abstracts):
{context}

Question: {question}

Based on the context above, answer the question. Provide:
- Final answer (yes/no/maybe):
- Explanation:"""


class RAGGenerator:
    """Answer generator using quantized BioMistral-7B with retrieved context."""

    def __init__(
        self,
        model_name: str = "BioMistral/BioMistral-7B",
        device: Optional[str] = None,
        load_in_4bit: bool = True,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # 4-bit quantization config for Colab T4
        quant_config = None
        if load_in_4bit and self.device == "cuda":
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quant_config,
            device_map="auto" if self.device == "cuda" else None,
            torch_dtype=torch.float16,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def format_context(self, retrieved_docs: List[str], max_docs: int = 3) -> str:
        """Format retrieved documents into a context string."""
        context_parts = []
        for i, doc in enumerate(retrieved_docs[:max_docs], 1):
            # Truncate very long abstracts
            truncated = doc[:1500] if len(doc) > 1500 else doc
            context_parts.append(f"[Abstract {i}]: {truncated}")
        return "\n\n".join(context_parts)

    def generate_answer(
        self,
        question: str,
        retrieved_docs: List[str],
        max_new_tokens: int = 300,
        temperature: float = 0.3,
        max_context_docs: int = 3,
    ) -> dict:
        """
        Generate an answer given a question and retrieved documents.

        Args:
            question: The medical question
            retrieved_docs: List of retrieved abstract texts
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            max_context_docs: Maximum number of context documents to include

        Returns:
            Dict with 'answer', 'decision', and 'full_response' keys
        """
        context = self.format_context(retrieved_docs, max_docs=max_context_docs)
        prompt = RAG_PROMPT_TEMPLATE.format(context=context, question=question)

        # Prepend system prompt
        full_prompt = f"{SYSTEM_PROMPT}\n\n{prompt}"

        inputs = self.tokenizer(
            full_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.1,
            )

        # Decode only the generated part
        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Parse decision from response
        decision = self._parse_decision(response)

        return {
            "decision": decision,
            "answer": response.strip(),
            "full_response": response,
            "question": question,
            "num_context_docs": min(len(retrieved_docs), max_context_docs),
        }

    def _parse_decision(self, response: str) -> str:
        """Extract yes/no/maybe decision from generated response."""
        response_lower = response.lower()

        # Look for explicit decision patterns
        for label in ["yes", "no", "maybe"]:
            # Check common patterns
            if f"final answer: {label}" in response_lower:
                return label
            if f"answer: {label}" in response_lower:
                return label
            if f"answer ({label})" in response_lower:
                return label

        # Fallback: check first line
        first_line = response_lower.split("\n")[0].strip()
        for label in ["yes", "no", "maybe"]:
            if label in first_line:
                return label

        return "maybe"  # Default when parsing fails
