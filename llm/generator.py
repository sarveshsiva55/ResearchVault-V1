from openai import OpenAI
from config.settings import LLM_TEMPERATURE, LLM_TOP_P, LLM_MAX_NEW_TOKENS

class LocalLLMGenerator:
    def __init__(self, api_base: str = "http://localhost:8000/v1", api_key: str = "sk-no-key-required"):
        # Connects to a local llama.cpp server instance
        self.client = OpenAI(base_url=api_base, api_key=api_key)

    def generate_answer(self, query: str, evidence_chunks: list[str]) -> str:
        
        # Grounded Answer Prompt from Section 8.5
        system_prompt = (
            "You are an offline research knowledge assistant.\n"
            "Use only the supplied evidence.\n"
            "Do not invent facts, values, papers, authors, citations or references.\n"
            "Every factual claim must be supported by evidence.\n"
            "Numerical claims must match evidence exactly.\n"
            "If evidence is insufficient, state that the stored collection does not contain sufficient evidence.\n"
        )
        
        evidence_text = "\n\n".join([f"EVIDENCE [{i}]:\n{chunk}" for i, chunk in enumerate(evidence_chunks)])
        
        user_prompt = f"EVIDENCE:\n{evidence_text}\n\nQUESTION:\n{query}"

        try:
            response = self.client.chat.completions.create(
                model="local-model", # The model name doesn't matter for local llama.cpp
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=LLM_TEMPERATURE,
                top_p=LLM_TOP_P,
                max_tokens=LLM_MAX_NEW_TOKENS
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error connecting to local LLM: {str(e)}"

if __name__ == "__main__":
    llm = LocalLLMGenerator()
    print("Local LLM Generator Ready.")
