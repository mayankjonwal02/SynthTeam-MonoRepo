import json
from typing import Optional
from openai import AsyncOpenAI
from agents import Agent, Runner, OpenAIChatCompletionsModel


# ============================================
# Configure Ollama Client
# ============================================

ollama_client = AsyncOpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

ollama_model = OpenAIChatCompletionsModel(
    model="qwen3:8b",
    openai_client=ollama_client
)


# ============================================
# Agent 1: File Schema Generator
# ============================================

FILE_SCHEMA_INSTRUCTIONS = """You are a File Schema Generator Agent for FastAPI-based AI Services.

Your task is to analyze the given AI service requirements and techstack, then generate a comprehensive file structure schema.

You will receive:
1. **Features**: List of AI features to implement (vision, LLM, external AI services, etc.)
2. **Techstack**: AI technologies to use (OpenAI, HuggingFace, LangChain, etc.)

Generate a file schema that follows FastAPI best practices:

IMPORTANT RULES:
1. Output ONLY valid JSON - no markdown, no explanations
2. The schema should include ALL necessary files for a production-ready FastAPI AI service
3. Include proper folder structure with services, routes, models, utils, config
4. Each file entry should have: "path", "description", "dependencies" (list of other files it depends on)

Standard structure for FastAPI AI Service:
- main.py (FastAPI app entry point)
- config.py (Configuration and environment variables)
- requirements.txt (Python dependencies)
- routes/ (API endpoints)
- services/ (AI service logic)
- models/ (Pydantic models for request/response)
- utils/ (Helper functions)

Output format:
{
    "project_name": "ai_service_name",
    "description": "Brief description of the AI service",
    "files": [
        {
            "path": "main.py",
            "description": "FastAPI application entry point",
            "dependencies": ["config.py", "routes/"]
        },
        {
            "path": "config.py", 
            "description": "Configuration and settings",
            "dependencies": []
        },
        {
            "path": "requirements.txt",
            "description": "Python package dependencies",
            "dependencies": []
        },
        {
            "path": "models/schemas.py",
            "description": "Pydantic request/response models",
            "dependencies": []
        },
        {
            "path": "services/ai_service.py",
            "description": "Core AI service implementation",
            "dependencies": ["config.py", "models/schemas.py"]
        },
        {
            "path": "routes/ai_routes.py",
            "description": "API route definitions",
            "dependencies": ["services/ai_service.py", "models/schemas.py"]
        },
        {
            "path": "utils/helpers.py",
            "description": "Utility functions",
            "dependencies": []
        }
    ],
    "ai_capabilities": ["list of AI capabilities this service will have"]
}

Remember: Output ONLY the JSON object, nothing else.
"""

file_schema_agent = Agent(
    name="File Schema Generator",
    instructions=FILE_SCHEMA_INSTRUCTIONS,
    tools=[],
    model=ollama_model
)


# ============================================
# Agent 2: Coding Agent
# ============================================

CODING_AGENT_INSTRUCTIONS = """You are an Expert AI Coding Agent specializing in FastAPI-based AI Services.

Your task is to generate production-ready Python code for FastAPI AI services.

You will receive:
1. **File Schema**: The file structure to implement
2. **Requirements**: Features to implement
3. **Techstack**: Technologies to use
4. **Current Code**: Existing code (may be empty or have previous code to improve)
5. **Review Feedback**: Feedback from code reviewer (if any)

IMPORTANT RULES:
1. Output ONLY valid JSON - no markdown, no explanations
2. Generate complete, working code for each file
3. Follow Python best practices and PEP 8 style
4. Include proper error handling and validation
5. Use async/await for I/O operations
6. Include proper type hints
7. Include docstrings for functions and classes
8. If review feedback is provided, address ALL issues mentioned

Code should include:
- Proper imports
- Environment variable handling
- Error handling with try/except
- Input validation with Pydantic
- Async endpoints where appropriate
- Proper HTTP status codes
- CORS middleware setup
- Health check endpoint

Output format:
{
    "files": {
        "main.py": "# Full code for main.py\\nimport ...\\n...",
        "config.py": "# Full code for config.py\\n...",
        "requirements.txt": "fastapi\\nuvicorn\\n...",
        "models/schemas.py": "# Full code...\\n...",
        "services/ai_service.py": "# Full code...\\n...",
        "routes/ai_routes.py": "# Full code...\\n..."
    },
    "status": "completed",
    "notes": ["Any notes about the implementation"]
}

Remember: Output ONLY the JSON object with complete working code for ALL files.
"""

coding_agent = Agent(
    name="AI Coding Agent",
    instructions=CODING_AGENT_INSTRUCTIONS,
    tools=[],
    model=ollama_model
)


# ============================================
# Agent 3: Code Reviewer Agent
# ============================================

CODE_REVIEWER_INSTRUCTIONS = """You are an Expert Code Reviewer Agent for FastAPI-based AI Services.

Your task is to thoroughly review the generated code and provide feedback.

You will receive:
1. **File Schema**: Expected file structure
2. **Requirements**: Features that should be implemented
3. **Generated Code**: The code to review

Review criteria:
1. **Completeness**: Are all required files present with complete code?
2. **Correctness**: Is the code syntactically correct and will it run?
3. **Imports**: Are all necessary imports included?
4. **Error Handling**: Is there proper try/except handling?
5. **Type Hints**: Are type hints used properly?
6. **Async/Await**: Are async operations handled correctly?
7. **Security**: Are there any security issues (exposed secrets, etc.)?
8. **Best Practices**: Does the code follow FastAPI best practices?
9. **API Design**: Are the endpoints properly designed?
10. **Documentation**: Are there docstrings and comments?

IMPORTANT RULES:
1. Output ONLY valid JSON - no markdown, no explanations
2. Be thorough but fair in your review
3. If code is acceptable, set "approved" to true
4. If issues found, set "approved" to false and list all issues
5. Provide specific, actionable feedback for each issue

Output format:
{
    "approved": true/false,
    "overall_score": 8.5,
    "review_summary": "Brief summary of the review",
    "issues": [
        {
            "file": "filename.py",
            "severity": "critical/major/minor",
            "issue": "Description of the issue",
            "suggestion": "How to fix it"
        }
    ],
    "improvements": [
        "Optional improvements that could be made"
    ],
    "feedback_for_coder": "Detailed feedback message for the coding agent"
}

Set "approved" to true ONLY if:
- All files are complete and have working code
- No critical or major issues exist
- Code will run without errors
- All requirements are implemented

Remember: Output ONLY the JSON object, nothing else.
"""

code_reviewer_agent = Agent(
    name="Code Reviewer Agent",
    instructions=CODE_REVIEWER_INSTRUCTIONS,
    tools=[],
    model=ollama_model
)


# ============================================
# Agent 4: Documentation Generator
# ============================================

DOCUMENTATION_INSTRUCTIONS = """You are a Technical Documentation Agent for FastAPI AI Services.

Your task is to generate comprehensive documentation in Markdown format.

You will receive:
1. **Project Info**: Project name and description
2. **File Schema**: The file structure
3. **Generated Code**: The complete codebase
4. **Requirements**: Original requirements

Generate documentation that includes:
1. Project Overview
2. Features
3. Installation instructions
4. Configuration (environment variables)
5. API Endpoints documentation
6. Usage examples with curl/Python
7. File structure explanation
8. Dependencies

IMPORTANT RULES:
1. Output ONLY the raw markdown content - no JSON wrapper, no code blocks around it
2. Make it comprehensive and professional
3. Include code examples
4. Document all API endpoints with request/response examples

Remember: Output ONLY the raw markdown documentation content.
"""

documentation_agent = Agent(
    name="Documentation Generator",
    instructions=DOCUMENTATION_INSTRUCTIONS,
    tools=[],
    model=ollama_model
)


# ============================================
# Helper Functions
# ============================================

def parse_json_response(response_text: str) -> dict:
    """Parse JSON from agent response, handling markdown code blocks."""
    text = response_text.strip()
    
    # Remove thinking tags if present (qwen3 sometimes includes these)
    if "<think>" in text:
        # Remove everything between <think> and </think>
        import re
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    
    # Handle markdown code blocks
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()
    
    return json.loads(text)


def generate_file_schema(features: list[str], ai_techstack: list[str]) -> dict:
    """Generate file schema for the AI service."""
    prompt = f"""Generate a file schema for a FastAPI AI Service with:

**Features to implement:**
{json.dumps(features, indent=2)}

**AI Technologies to use:**
{json.dumps(ai_techstack, indent=2)}

Create a comprehensive file structure that supports all these features."""

    result = Runner.run_sync(file_schema_agent, prompt)
    
    try:
        return parse_json_response(result.final_output)
    except json.JSONDecodeError as e:
        return {
            "error": f"Failed to parse schema: {str(e)}",
            "raw": result.final_output,
            "files": []
        }


def generate_code(file_schema: dict, features: list[str], techstack: list[str], 
                  current_code: dict, review_feedback: str = "") -> dict:
    """Generate code based on schema and feedback."""
    prompt = f"""Generate FastAPI AI Service code:

**File Schema:**
{json.dumps(file_schema, indent=2)}

**Features to implement:**
{json.dumps(features, indent=2)}

**AI Technologies:**
{json.dumps(techstack, indent=2)}

**Current Code (to improve upon):**
{json.dumps(current_code, indent=2) if current_code else "Empty - generate fresh code"}

**Review Feedback to address:**
{review_feedback if review_feedback else "No feedback yet - generate initial implementation"}

Generate complete, production-ready code for ALL files in the schema."""

    result = Runner.run_sync(coding_agent, prompt)
    
    try:
        return parse_json_response(result.final_output)
    except json.JSONDecodeError as e:
        return {
            "error": f"Failed to parse code: {str(e)}",
            "raw": result.final_output,
            "files": {},
            "status": "error"
        }


def review_code(file_schema: dict, features: list[str], generated_code: dict) -> dict:
    """Review the generated code."""
    prompt = f"""Review this FastAPI AI Service code:

**Expected File Schema:**
{json.dumps(file_schema, indent=2)}

**Required Features:**
{json.dumps(features, indent=2)}

**Generated Code to Review:**
{json.dumps(generated_code, indent=2)}

Thoroughly review the code and determine if it's ready for production."""

    result = Runner.run_sync(code_reviewer_agent, prompt)
    
    try:
        return parse_json_response(result.final_output)
    except json.JSONDecodeError as e:
        return {
            "error": f"Failed to parse review: {str(e)}",
            "raw": result.final_output,
            "approved": False,
            "feedback_for_coder": "Review parsing failed, please regenerate code"
        }


def generate_documentation(project_info: dict, file_schema: dict, 
                           code: dict, features: list[str]) -> str:
    """Generate documentation for the AI service."""
    prompt = f"""Generate comprehensive documentation for this FastAPI AI Service:

**Project Info:**
{json.dumps(project_info, indent=2)}

**File Structure:**
{json.dumps(file_schema, indent=2)}

**Complete Codebase:**
{json.dumps(code, indent=2)}

**Features:**
{json.dumps(features, indent=2)}

Generate professional Markdown documentation."""

    result = Runner.run_sync(documentation_agent, prompt)
    
    # Return raw markdown (may have code blocks but that's fine for docs)
    text = result.final_output.strip()
    
    # Remove thinking tags if present
    if "<think>" in text:
        import re
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    
    return text


# ============================================
# Main AI Engineer Agent Function
# ============================================

def generate_ai_service(features: list[str], ai_techstack: list[str], 
                        max_iterations: int = 5, verbose: bool = True) -> dict:
    """
    Main function to generate a complete FastAPI AI Service.
    
    Args:
        features: List of AI features to implement
        ai_techstack: List of AI technologies to use
        max_iterations: Maximum number of coding/review iterations
        verbose: Whether to print progress
        
    Returns:
        Dictionary with file paths as keys and code as values,
        including documentation.md
    """
    
    if verbose:
        print("=" * 60)
        print("🤖 AI Engineer Agent - Starting Code Generation")
        print("=" * 60)
    
    # Step 1: Generate File Schema
    if verbose:
        print("\n📁 Step 1: Generating file schema...")
    
    file_schema = generate_file_schema(features, ai_techstack)
    
    if "error" in file_schema:
        if verbose:
            print(f"❌ Schema generation failed: {file_schema['error']}")
        return {"error": "Schema generation failed", "details": file_schema}
    
    if verbose:
        print(f"✅ Schema generated: {len(file_schema.get('files', []))} files planned")
        for f in file_schema.get('files', []):
            print(f"   - {f.get('path', 'unknown')}")
    
    # Step 2: Coding and Review Loop
    current_code = {}
    review_feedback = ""
    iteration = 0
    approved = False
    
    while not approved and iteration < max_iterations:
        iteration += 1
        
        if verbose:
            print(f"\n🔄 Iteration {iteration}/{max_iterations}")
            print("-" * 40)
        
        # Generate/Update Code
        if verbose:
            print("💻 Coding Agent: Generating code...")
        
        code_result = generate_code(
            file_schema=file_schema,
            features=features,
            techstack=ai_techstack,
            current_code=current_code,
            review_feedback=review_feedback
        )
        
        if "error" in code_result and code_result.get("status") == "error":
            if verbose:
                print(f"⚠️  Code generation had issues, retrying...")
            continue
        
        current_code = code_result.get("files", {})
        
        if verbose:
            print(f"✅ Generated {len(current_code)} files")
        
        # Review Code
        if verbose:
            print("🔍 Reviewer Agent: Reviewing code...")
        
        review_result = review_code(
            file_schema=file_schema,
            features=features,
            generated_code=current_code
        )
        
        approved = review_result.get("approved", False)
        review_feedback = review_result.get("feedback_for_coder", "")
        
        if verbose:
            score = review_result.get("overall_score", "N/A")
            print(f"   Score: {score}/10")
            print(f"   Approved: {'✅ Yes' if approved else '❌ No'}")
            
            if not approved:
                issues = review_result.get("issues", [])
                if issues:
                    print(f"   Issues found: {len(issues)}")
                    for issue in issues[:3]:  # Show first 3 issues
                        print(f"      - [{issue.get('severity', 'unknown')}] {issue.get('file', '')}: {issue.get('issue', '')[:50]}...")
    
    if verbose:
        if approved:
            print(f"\n✅ Code approved after {iteration} iteration(s)!")
        else:
            print(f"\n⚠️  Max iterations reached. Using best available code.")
    
    # Step 3: Generate Documentation
    if verbose:
        print("\n📝 Step 3: Generating documentation...")
    
    project_info = {
        "name": file_schema.get("project_name", "ai_service"),
        "description": file_schema.get("description", "FastAPI AI Service")
    }
    
    documentation = generate_documentation(
        project_info=project_info,
        file_schema=file_schema,
        code=current_code,
        features=features
    )
    
    # Add documentation to output
    current_code["documentation.md"] = documentation
    
    if verbose:
        print("✅ Documentation generated!")
        print("\n" + "=" * 60)
        print("🎉 AI Service Generation Complete!")
        print(f"   Total files: {len(current_code)}")
        print("=" * 60)
    
    # Return final result
    return {
        "files": current_code,
        "schema": file_schema,
        "iterations": iteration,
        "approved": approved,
        "project_name": project_info["name"]
    }


# ============================================
# Export Functions
# ============================================

def get_ai_service_code(features: list[str], ai_techstack: list[str], 
                        max_iterations: int = 5) -> dict:
    """
    Main export function to generate AI service code.
    
    Args:
        features: List of AI features (e.g., ["image classification", "text generation"])
        ai_techstack: List of AI technologies (e.g., ["OpenAI", "HuggingFace"])
        max_iterations: Maximum coding/review iterations
        
    Returns:
        Dictionary with:
        - "files": Dict of filepath -> code content
        - "schema": The file schema used
        - "iterations": Number of iterations taken
        - "approved": Whether code was approved by reviewer
        - "project_name": Name of the generated project
    """
    return generate_ai_service(features, ai_techstack, max_iterations, verbose=False)


def chat(features: list[str], ai_techstack: list[str]) -> str:
    """Generate AI service and return formatted JSON string."""
    result = generate_ai_service(features, ai_techstack, verbose=False)
    return json.dumps(result, indent=2)


# ============================================
# Interactive CLI
# ============================================

def run_interactive_cli():
    """Run an interactive CLI session with the AI Engineer Agent."""
    print("=" * 60)
    print("🤖 AI Engineer Agent")
    print("   FastAPI AI Service Generator")
    print("   Powered by Ollama (qwen3:8b)")
    print("=" * 60)
    print("\nI generate complete FastAPI-based AI services with:")
    print("  - Multi-agent code generation and review")
    print("  - Production-ready code")
    print("  - Comprehensive documentation")
    print("\nSupported AI capabilities:")
    print("  - Vision (image classification, object detection)")
    print("  - LLM (text generation, summarization, Q&A)")
    print("  - External AI APIs (OpenAI, HuggingFace, etc.)")
    print("\nType 'exit' or 'quit' to end the session.\n")
    
    while True:
        try:
            # Get features
            print("\n📋 Enter AI features to implement (comma-separated):")
            print("   Example: image classification, text generation, sentiment analysis")
            features_input = input("   > ").strip()
            
            if features_input.lower() in ["exit", "quit", "q"]:
                print("\n👋 Goodbye!")
                break
            
            if not features_input:
                print("❌ Please enter at least one feature.")
                continue
            
            features = [f.strip() for f in features_input.split(",")]
            
            # Get AI techstack
            print("\n🛠️  Enter AI technologies to use (comma-separated):")
            print("   Example: OpenAI, HuggingFace Transformers, LangChain")
            tech_input = input("   > ").strip()
            
            if tech_input.lower() in ["exit", "quit", "q"]:
                print("\n👋 Goodbye!")
                break
            
            if not tech_input:
                print("❌ Please enter at least one AI technology.")
                continue
            
            ai_techstack = [t.strip() for t in tech_input.split(",")]
            
            # Generate AI service
            print("\n" + "=" * 60)
            result = generate_ai_service(features, ai_techstack, verbose=True)
            
            if "error" not in result:
                print("\n📊 Generated Files:")
                print("-" * 40)
                for filepath in result.get("files", {}).keys():
                    print(f"   📄 {filepath}")
                
                # Ask if user wants to see the code
                print("\n💾 Would you like to save the generated code? (yes/no)")
                save_input = input("   > ").strip().lower()
                
                if save_input in ["yes", "y"]:
                    print("\n📁 Enter output directory path:")
                    output_dir = input("   > ").strip()
                    
                    if output_dir:
                        import os
                        os.makedirs(output_dir, exist_ok=True)
                        
                        for filepath, code in result.get("files", {}).items():
                            full_path = os.path.join(output_dir, filepath)
                            os.makedirs(os.path.dirname(full_path) if os.path.dirname(full_path) else output_dir, exist_ok=True)
                            with open(full_path, "w") as f:
                                f.write(code)
                            print(f"   ✅ Saved: {full_path}")
                        
                        print(f"\n🎉 All files saved to: {output_dir}")
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    run_interactive_cli()

