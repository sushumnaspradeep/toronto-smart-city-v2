import os
import json
import logging
from openai import OpenAI
from dotenv import load_dotenv
from resolver import TorontoUrbanLocationResolver # Import the resolver from previous steps

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

class NeighborhoodUrbanAgent:
    def __init__(self, data_dir: str = "../../data_payloads"):
        """
        Initializes the agent, the local spatial resolver, and the connection 
        to the Nemotron model hosted on the ASUS DGX.
        """
        self.resolver = TorontoUrbanLocationResolver(data_dir=data_dir)
        
        # Point the standard OpenAI client to your local DGX server
        dgx_base_url = os.getenv("DGX_API_BASE_URL", "http://localhost:8000/v1")
        dgx_api_key = os.getenv("DGX_API_KEY", "empty-key-for-local")
        
        # The specific Nemotron model loaded on your DGX (update if needed)
        self.model_name = os.getenv("NEMOTRON_MODEL_NAME", "nvidia/nemotron-4-340b-instruct")
        
        logger.info(f"Connecting to Nemotron DGX Endpoint at: {dgx_base_url}")
        self.llm_client = OpenAI(
            base_url=dgx_base_url,
            api_key=dgx_api_key
        )

    def analyze_neighborhood(self, user_query: str, target_address: str) -> str:
        """
        Resolves the address to exact city metrics, then prompts Nemotron for an analysis.
        """
        logger.info(f"Resolving context for address: {target_address}")
        
        # 1. Get the deterministic JSON data using our local pipeline
        resolution_result = self.resolver.resolve_profile_by_address(target_address)
        
        if "error" in resolution_result:
            return f"System Error: {resolution_result['error']}"
            
        ward_name = resolution_result["matched_neighborhood"]
        metric_data = resolution_result["data_payload"]
        
        # 2. Construct the Agent's Context and Prompt
        system_prompt = (
            "You are an expert urban planning advisor for the City of Toronto. "
            "You provide highly analytical, data-driven advice to government agencies regarding "
            "neighborhood revitalization. Speak professionally, concisely, and ground all your "
            "recommendations explicitly in the provided data payload. Do not invent metrics."
        )
        
        user_prompt = (
            f"**Target Location:** {target_address} (Located in: {ward_name})\n\n"
            f"**City Data Payload for {ward_name}:**\n"
            f"{json.dumps(metric_data, indent=2)}\n\n"
            f"**Agency Request:** {user_query}"
        )

        # 3. Call the DGX Nemotron Endpoint
        logger.info(f"Sending context and query to Nemotron model ({self.model_name})...")
        try:
            response = self.llm_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2, # Keep temperature low for analytical accuracy
                max_tokens=1024
            )
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Failed to communicate with DGX endpoint: {e}")
            return f"LLM Connection Error: {str(e)}"

# --- Execution Demo ---
if __name__ == "__main__":
    # Note: Ensure you run this from within the 'src' directory so the relative paths match
    agent = NeighborhoodUrbanAgent()
    
    query = "I want to open a youth community center. Does this area have enough transit and recreation, or is it lacking?"
    address = "100 Queen St W"
    
    print("\n" + "="*50)
    print(f"QUERY: {query}\nLOCATION: {address}")
    print("="*50 + "\n")
    
    answer = agent.analyze_neighborhood(user_query=query, target_address=address)
    
    print("🤖 NEMOTRON RESPONSE:\n")
    print(answer)