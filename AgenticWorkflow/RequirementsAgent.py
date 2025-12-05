import json
from typing import Optional
from openai import AsyncOpenAI
from agents import Agent, Runner, OpenAIChatCompletionsModel
from pydantic import BaseModel


# ============================================
# Response Schema
# ============================================

class RequirementsOutput(BaseModel):
    """Structured output for requirements analysis."""
    Features: list[str]
    Techstack: dict[str, list[str]]


# ============================================
# Requirements Analysis Agent
# ============================================

SYSTEM_INSTRUCTIONS = """You are a Requirements Analysis Agent. Your task is to analyze user prompts describing their website/application requirements and extract structured information.

When a user describes what they want to build, you must:

1. **Extract Features**: Identify ALL features mentioned or implied in the user's description. Be comprehensive and include:
   - Core functionality features
   - User authentication/authorization features
   - UI/UX features
   - Data management features
   - Integration features
   - Any other features mentioned or reasonably implied

2. **Recommend Techstack**: Based on the requirements, suggest appropriate technologies for:
   - **frontend**: UI frameworks, CSS libraries, state management tools
   - **backend**: Server frameworks, API technologies, authentication libraries
   - **database**: Database systems (SQL, NoSQL, caching, etc.)
   - **ai**: AI/ML tools, APIs, or libraries if applicable (leave empty if not needed)
   - **devops**: Deployment, CI/CD, containerization tools if applicable

IMPORTANT RULES:
1. You MUST respond with ONLY valid JSON - no additional text, explanations, or markdown
2. The JSON must have exactly two keys: "Features" and "Techstack"
3. "Features" must be an array of strings
4. "Techstack" must be an object with keys: "frontend", "backend", "database", "ai", "devops"
5. Each techstack category must be an array of strings
6. Be specific with technology choices (e.g., "React 18" not just "React")
7. Consider modern, production-ready technologies
8. If AI is not needed for the project, set "ai" to an empty array []

Example output format:
{
    "Features": [
        "User registration and login",
        "Product catalog with search",
        "Shopping cart functionality",
        "Payment processing",
        "Order tracking",
        "Admin dashboard"
    ],
    "Techstack": {
        "frontend": ["Next.js 14", "Tailwind CSS", "Zustand"],
        "backend": ["Node.js", "Express.js", "JWT Authentication"],
        "database": ["PostgreSQL", "Redis"],
        "ai": [],
        "devops": ["Docker", "GitHub Actions", "Vercel"]
    }
}

Remember: Output ONLY the JSON object, nothing else. Do not put the json in any code block.
"""

# Configure Ollama client (OpenAI-compatible API)
ollama_client = AsyncOpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"  # Ollama doesn't require a real API key
)

# Create the model using Ollama's qwen3:8b
ollama_model = OpenAIChatCompletionsModel(
    model="qwen3:8b",
    openai_client=ollama_client
)

# Create the Requirements Analysis Agent
requirements_agent = Agent(
    name="Requirements Analysis Agent",
    instructions=SYSTEM_INSTRUCTIONS,
    tools=[],  # No tools needed - pure analysis agent
    model=ollama_model
)


def analyze_requirements(prompt: str) -> dict:
    """
    Analyze user requirements and return structured JSON output.
    
    Args:
        prompt: User's description of their website/application requirements
        
    Returns:
        Dictionary containing Features and Techstack
    """
    result = Runner.run_sync(requirements_agent, prompt)
    
    # Parse the JSON response
    try:
        # Try to extract JSON from the response
        response_text = result.final_output.strip()
        
        # Handle case where response might have markdown code blocks
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()
        
        parsed_output = json.loads(response_text)
        
        # Validate structure
        if "Features" not in parsed_output:
            parsed_output["Features"] = []
        if "Techstack" not in parsed_output:
            parsed_output["Techstack"] = {
                "frontend": [],
                "backend": [],
                "database": [],
                "ai": [],
                "devops": []
            }
        
        return parsed_output
    except json.JSONDecodeError as e:
        # Return raw output with error info if parsing fails
        return {
            "error": f"Failed to parse JSON: {str(e)}",
            "raw_output": result.final_output,
            "Features": [],
            "Techstack": {
                "frontend": [],
                "backend": [],
                "database": [],
                "ai": [],
                "devops": []
            }
        }


def chat(prompt: str) -> str:
    """Send a single prompt to the agent and get a formatted response."""
    result = analyze_requirements(prompt)
    return json.dumps(result, indent=2)


def run_interactive_cli():
    """Run an interactive CLI session with the Requirements Agent."""
    print("=" * 60)
    print("📋 Requirements Analysis Agent")
    print("   Powered by Ollama (qwen3:8b)")
    print("=" * 60)
    print("\nI analyze your project requirements and provide:")
    print("  - A comprehensive list of features")
    print("  - Recommended technology stack")
    print("\nDescribe your project idea and I'll break it down!")
    print("\nExamples:")
    print("  - 'I want to build an e-commerce website with user login,'")
    print("    'product search, cart, and payment integration'")
    print("  - 'Create a social media platform with posts, comments,'")
    print("    'likes, and real-time notifications'")
    print("\nType 'exit' or 'quit' to end the session.\n")
    
    while True:
        try:
            user_input = input("\n📝 Describe your project: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit", "q"]:
                print("\n👋 Goodbye!")
                break
            
            print("\n🔍 Analyzing requirements...\n")
            response = chat(user_input)
            print("📊 Analysis Result:")
            print("-" * 40)
            print(response)
            print("-" * 40)
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")


# Export for use by other modules
def get_requirements(user_prompt: str) -> dict:
    """
    Main function to get requirements from user prompt.
    
    Args:
        user_prompt: User's project description
        
    Returns:
        Dictionary with 'Features' and 'Techstack' keys
    """
    return analyze_requirements(user_prompt)


if __name__ == "__main__":
    run_interactive_cli()

