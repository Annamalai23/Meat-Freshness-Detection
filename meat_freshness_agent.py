import os
import json
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class MeatFreshnessAgent:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if self.api_key:
            self.client = OpenAI(api_key=self.api_key)
        else:
            self.client = None
        
        self.memory = {
            "analysis_history": []
        }

    async def analyze_with_reasoning(self, analysis_data):
        """Perform multi-step reasoning on detection results using LLM"""
        if not self.client:
            return {
                "analysis": "Agentic reasoning unavailable (No API Key).",
                "reasoning_steps": 0,
                "tools_used": []
            }

        prompt = f"""
        As a Food Safety Expert AI, analyze the following meat freshness detection data:
        Data: {json.dumps(analysis_data)}
        
        Provide:
        1. A brief technical explanation of why the meat was classified this way.
        2. Specific food safety advice for the consumer.
        3. Commercial recommendations for the retailer (e.g., pricing, storage).
        
        Be concise but professional.
        """

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a professional meat quality and food safety agent."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7
            )
            
            analysis_text = response.choices[0].message.content
            
            # Store in memory
            self.memory["analysis_history"].append({
                "timestamp": datetime.now().isoformat(),
                "result": analysis_text
            })

            return {
                "analysis": analysis_text,
                "reasoning_steps": 2,
                "tools_used": ["LLM-Reasoning"]
            }
        except Exception as e:
            return {
                "analysis": f"Reasoning failed: {str(e)}",
                "reasoning_steps": 1,
                "tools_used": []
            }

    def get_memory_summary(self):
        count = len(self.memory["analysis_history"])
        return f"Total analyses in memory: {count}"

def create_agent():
    return MeatFreshnessAgent()
