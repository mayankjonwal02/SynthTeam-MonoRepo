import json
from typing import Optional
from openai import AsyncOpenAI
from agents import Agent, Runner, OpenAIChatCompletionsModel


# ============================================
# Database Schema Generation Agent
# ============================================

SYSTEM_INSTRUCTIONS = """You are a Database Schema Design Agent. Your task is to analyze product requirements and generate comprehensive database schemas based on the specified database technology.

You will receive:
1. **Product Requirements**: A list of features that the application needs
2. **Database Techstack**: The database technologies to design schemas for (e.g., PostgreSQL, MongoDB, MySQL, Redis)

Your job is to:
1. Analyze all the features and identify the data entities needed
2. Design appropriate schemas for each specified database technology
3. Include all necessary fields, data types, relationships, and indexes
4. Consider best practices for each database type

IMPORTANT RULES:
1. You MUST respond with ONLY valid JSON - no additional text, explanations, or markdown code blocks
2. The JSON must have a "schemas" key containing an object
3. Each database in the techstack should have its own schema definition
4. For SQL databases (PostgreSQL, MySQL, etc.), provide table definitions with columns, types, constraints, and relationships
5. For NoSQL databases (MongoDB, etc.), provide collection schemas with field definitions
6. For cache databases (Redis), provide key patterns and data structures
7. Include indexes for frequently queried fields
8. Include foreign key relationships where applicable
9. Use appropriate data types for each database system

Output format:
{
    "schemas": {
        "PostgreSQL": {
            "tables": {
                "users": {
                    "columns": {
                        "id": {"type": "UUID", "primary_key": true, "default": "gen_random_uuid()"},
                        "email": {"type": "VARCHAR(255)", "unique": true, "not_null": true},
                        "password_hash": {"type": "VARCHAR(255)", "not_null": true},
                        "created_at": {"type": "TIMESTAMP", "default": "CURRENT_TIMESTAMP"},
                        "updated_at": {"type": "TIMESTAMP", "default": "CURRENT_TIMESTAMP"}
                    },
                    "indexes": ["email"],
                    "constraints": []
                },
                "products": {
                    "columns": {
                        "id": {"type": "UUID", "primary_key": true},
                        "name": {"type": "VARCHAR(255)", "not_null": true},
                        "price": {"type": "DECIMAL(10,2)", "not_null": true},
                        "user_id": {"type": "UUID", "foreign_key": "users.id"}
                    },
                    "indexes": ["name", "user_id"],
                    "constraints": ["FOREIGN KEY (user_id) REFERENCES users(id)"]
                }
            }
        },
        "MongoDB": {
            "collections": {
                "users": {
                    "schema": {
                        "_id": "ObjectId",
                        "email": {"type": "String", "required": true, "unique": true},
                        "passwordHash": {"type": "String", "required": true},
                        "profile": {
                            "name": "String",
                            "avatar": "String"
                        },
                        "createdAt": "Date",
                        "updatedAt": "Date"
                    },
                    "indexes": [{"email": 1}]
                }
            }
        },
        "Redis": {
            "key_patterns": {
                "session": {
                    "pattern": "session:{userId}",
                    "type": "HASH",
                    "fields": ["token", "expiresAt", "deviceInfo"],
                    "ttl": 86400
                },
                "cache": {
                    "pattern": "cache:product:{productId}",
                    "type": "STRING",
                    "ttl": 3600
                }
            }
        }
    },
    "relationships": [
        {"from": "products.user_id", "to": "users.id", "type": "many-to-one"},
        {"from": "orders.user_id", "to": "users.id", "type": "many-to-one"}
    ],
    "notes": [
        "Use UUID for primary keys for better distribution",
        "Add indexes on frequently queried columns"
    ]
}

Remember: Output ONLY the JSON object, nothing else. No markdown code blocks.
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

# Create the Database Schema Agent
database_agent = Agent(
    name="Database Schema Agent",
    instructions=SYSTEM_INSTRUCTIONS,
    tools=[],  # No tools needed - pure schema generation agent
    model=ollama_model
)


def generate_schemas(requirements: list[str], db_techstack: list[str]) -> dict:
    """
    Generate database schemas based on requirements and techstack.
    
    Args:
        requirements: List of product features/requirements
        db_techstack: List of database technologies (e.g., ["PostgreSQL", "MongoDB", "Redis"])
        
    Returns:
        Dictionary containing database schemas for each technology
    """
    # Format the prompt for the agent
    prompt = f"""Generate database schemas for the following:

**Product Requirements/Features:**
{json.dumps(requirements, indent=2)}

**Database Technologies to use:**
{json.dumps(db_techstack, indent=2)}

Analyze all the features and create comprehensive schemas for each database technology listed. 
Include all necessary tables/collections, fields, relationships, indexes, and constraints.
Consider the data needs of each feature and how they relate to each other."""

    result = Runner.run_sync(database_agent, prompt)
    
    # Parse the JSON response
    try:
        response_text = result.final_output.strip()
        
        # Handle case where response might have markdown code blocks
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()
        
        parsed_output = json.loads(response_text)
        
        # Validate structure
        if "schemas" not in parsed_output:
            parsed_output = {"schemas": parsed_output}
        
        return parsed_output
        
    except json.JSONDecodeError as e:
        return {
            "error": f"Failed to parse JSON: {str(e)}",
            "raw_output": result.final_output,
            "schemas": {}
        }


def chat(requirements: list[str], db_techstack: list[str]) -> str:
    """Generate schemas and return formatted JSON string."""
    result = generate_schemas(requirements, db_techstack)
    return json.dumps(result, indent=2)


def run_interactive_cli():
    """Run an interactive CLI session with the Database Agent."""
    print("=" * 60)
    print("🗄️  Database Schema Agent")
    print("   Powered by Ollama (qwen3:8b)")
    print("=" * 60)
    print("\nI generate database schemas based on your requirements.")
    print("I support: PostgreSQL, MySQL, MongoDB, Redis, and more!")
    print("\nType 'exit' or 'quit' to end the session.\n")
    
    while True:
        try:
            # Get requirements
            print("\n📋 Enter your product features (comma-separated):")
            print("   Example: user auth, product catalog, shopping cart, orders")
            features_input = input("   > ").strip()
            
            if features_input.lower() in ["exit", "quit", "q"]:
                print("\n👋 Goodbye!")
                break
            
            if not features_input:
                print("❌ Please enter at least one feature.")
                continue
            
            requirements = [f.strip() for f in features_input.split(",")]
            
            # Get database techstack
            print("\n🛢️  Enter database technologies (comma-separated):")
            print("   Example: PostgreSQL, MongoDB, Redis")
            db_input = input("   > ").strip()
            
            if db_input.lower() in ["exit", "quit", "q"]:
                print("\n👋 Goodbye!")
                break
            
            if not db_input:
                print("❌ Please enter at least one database technology.")
                continue
            
            db_techstack = [db.strip() for db in db_input.split(",")]
            
            print("\n🔄 Generating schemas...\n")
            response = chat(requirements, db_techstack)
            print("📊 Generated Schemas:")
            print("-" * 40)
            print(response)
            print("-" * 40)
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")


# Export for use by other modules
def get_database_schemas(requirements: list[str], db_techstack: list[str]) -> dict:
    """
    Main function to get database schemas from requirements.
    
    Args:
        requirements: List of product features/requirements
        db_techstack: List of database technologies
        
    Returns:
        Dictionary with 'schemas', 'relationships', and 'notes' keys
    """
    return generate_schemas(requirements, db_techstack)


if __name__ == "__main__":
    run_interactive_cli()

