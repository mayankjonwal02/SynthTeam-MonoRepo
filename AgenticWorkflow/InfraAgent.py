import docker
import uuid
import time
import json
from typing import Optional
from openai import AsyncOpenAI
from agents import Agent, Runner, function_tool, OpenAIChatCompletionsModel

# ============================================
# Docker Container Management Tools
# ============================================

@function_tool
def run_container(
    image_name: str,
    project_id: str,
    ports: Optional[str] = None,
    environment: Optional[str] = None,
    command: Optional[str] = None
) -> str:
    """
    Run a Docker container under a specific project ID. Always pulls the latest image.
    
    Args:
        image_name: Docker image name (e.g., "mongo:latest", "postgres:latest", "redis:latest")
        project_id: Project identifier to group containers
        ports: JSON string of ports to expose (e.g., '{"27017/tcp": null}' for auto-assign)
        environment: JSON string of environment variables (e.g., '{"POSTGRES_PASSWORD": "secret"}')
        command: Optional command to run in the container
    
    Returns:
        JSON string with container details including container_id, name, and port_mappings
    """
    client = docker.from_env()
    
    # Parse JSON strings
    ports_dict = json.loads(ports) if ports else None
    env_dict = json.loads(environment) if environment else None

    # Always pull the latest image
    print(f"[INFO] Pulling latest image {image_name}...")
    try:
        for line in client.api.pull(image_name, stream=True, decode=True):
            if 'status' in line:
                layer = line.get('id', '')
                status = line.get('status', '')
                progress = line.get('progress', '')
                print(f"\r  {layer}: {status} {progress}", end='', flush=True)
        print()
    except Exception as e:
        print(f"[WARN] Failed to pull image: {e}. Trying cached image...")

    # Create a unique container name
    unique_id = uuid.uuid4().hex[:8]
    service_name = image_name.split(":")[0].split("/")[-1]
    container_name = f"{project_id}-{service_name}-{unique_id}"

    # Run the container
    print(f"[INFO] Starting container: {container_name}")

    container = client.containers.run(
        image_name,
        name=container_name,
        detach=True,
        ports=ports_dict,
        environment=env_dict,
        command=command,
        labels={
            "project_id": project_id,
            "service": service_name
        }
    )

    # Fetch port mappings with retry
    port_mappings = {}
    if ports_dict:
        for _ in range(10):
            container.reload()
            container_ports = container.attrs["NetworkSettings"]["Ports"]
            all_ports_assigned = True
            
            for port_key in ports_dict.keys():
                port_info = container_ports.get(port_key)
                if port_info and len(port_info) > 0:
                    port_mappings[port_key] = port_info[0]["HostPort"]
                else:
                    all_ports_assigned = False
            
            if all_ports_assigned:
                break
            time.sleep(0.5)

    print(f"[INFO] Container running: {container_name}")
    if port_mappings:
        print(f"[INFO] Port mappings: {port_mappings}")

    return json.dumps({
        "status": "success",
        "container_id": container.id[:12],
        "container_name": container_name,
        "project_id": project_id,
        "image": image_name,
        "port_mappings": port_mappings
    })


@function_tool
def list_project_containers(project_id: str) -> str:
    """
    List all containers belonging to a specific project.
    
    Args:
        project_id: Project identifier to filter containers
    
    Returns:
        JSON string with list of containers including their id, name, status, and image
    """
    client = docker.from_env()
    containers = client.containers.list(
        all=True,
        filters={"label": f"project_id={project_id}"}
    )
    result = [
        {
            "container_id": c.id[:12],
            "name": c.name,
            "status": c.status,
            "image": c.image.tags[0] if c.image.tags else "unknown"
        }
        for c in containers
    ]
    return json.dumps({"project_id": project_id, "containers": result})


@function_tool
def stop_container(container_id: str) -> str:
    """
    Stop a running container.
    
    Args:
        container_id: Container ID or name to stop
    
    Returns:
        JSON string with status of the operation
    """
    client = docker.from_env()
    try:
        container = client.containers.get(container_id)
        container.stop()
        return json.dumps({"status": "stopped", "container_id": container_id})
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})


@function_tool
def remove_container(container_id: str) -> str:
    """
    Remove a container (stops it first if running).
    
    Args:
        container_id: Container ID or name to remove
    
    Returns:
        JSON string with status of the operation
    """
    client = docker.from_env()
    try:
        container = client.containers.get(container_id)
        container.remove(force=True)
        return json.dumps({"status": "removed", "container_id": container_id})
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})


# ============================================
# Microservice Infrastructure Agent
# ============================================

SYSTEM_INSTRUCTIONS = """You are a Microservice Infrastructure Agent. Your task is to help users set up and manage microservices using Docker containers.

You can deploy:
- **Databases**: MongoDB, PostgreSQL, MySQL, MariaDB, Redis, Cassandra, CouchDB
- **Message Brokers**: RabbitMQ, Apache Kafka, Redis (pub/sub), NATS
- **Search**: Elasticsearch, OpenSearch, Meilisearch
- **Cache**: Redis, Memcached
- **Other**: Nginx, Traefik, Consul, Vault, MinIO

Common configurations (use these as reference):

MongoDB:
- image: mongo:latest
- ports: {"27017/tcp": null}
- environment: {"MONGO_INITDB_ROOT_USERNAME": "admin", "MONGO_INITDB_ROOT_PASSWORD": "<password>"}

PostgreSQL:
- image: postgres:latest
- ports: {"5432/tcp": null}
- environment: {"POSTGRES_PASSWORD": "<password>", "POSTGRES_USER": "admin", "POSTGRES_DB": "mydb"}

MySQL:
- image: mysql:latest
- ports: {"3306/tcp": null}
- environment: {"MYSQL_ROOT_PASSWORD": "<password>", "MYSQL_DATABASE": "mydb"}

Redis:
- image: redis:latest
- ports: {"6379/tcp": null}

RabbitMQ (with management UI):
- image: rabbitmq:management
- ports: {"5672/tcp": null, "15672/tcp": null}
- environment: {"RABBITMQ_DEFAULT_USER": "admin", "RABBITMQ_DEFAULT_PASS": "<password>"}

Elasticsearch:
- image: elasticsearch:8.11.0
- ports: {"9200/tcp": null, "9300/tcp": null}
- environment: {"discovery.type": "single-node", "xpack.security.enabled": "false"}

IMPORTANT:
1. Ask for project_id if not provided
2. Generate secure random passwords for services that require them
3. Explain what you're deploying and provide connection details after deployment
4. Always use null (not None) for auto-assigned ports in JSON
5. The ports and environment parameters must be valid JSON strings
6. use this command command="tail -f /dev/null" when setting up python or nodejs packages
"""

# Configure Ollama client (OpenAI-compatible API)
ollama_client = AsyncOpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"  # Ollama doesn't require a real API key
)

# Create the model using Ollama's qwen3:4b
ollama_model = OpenAIChatCompletionsModel(
    model="qwen3:8b",
    openai_client=ollama_client
)

# Create the agent
microservice_agent = Agent(
    name="Microservice Infrastructure Agent",
    instructions=SYSTEM_INSTRUCTIONS,
    tools=[run_container, list_project_containers, stop_container, remove_container],
    model=ollama_model
)


def chat(prompt: str) -> str:
    """Send a single prompt to the agent and get a response."""
    result = Runner.run_sync(microservice_agent, prompt)
    return result.final_output


def run_interactive_cli():
    """Run an interactive CLI session with the agent."""
    print("=" * 60)
    print("🚀 Microservice Infrastructure Agent")
    print("   Powered by Ollama (qwen3:4b)")
    print("=" * 60)
    print("\nI can help you set up databases, message brokers, and other")
    print("microservices using Docker. Just tell me what you need!")
    print("\nExamples:")
    print("  - 'Set up a MongoDB database for project myapp'")
    print("  - 'I need PostgreSQL and Redis for my backend'")
    print("  - 'List all containers in project myapp'")
    print("\nType 'exit' or 'quit' to end the session.\n")
    
    while True:
        try:
            user_input = input("\n📝 You: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit", "q"]:
                print("\n👋 Goodbye!")
                break
            
            print("\n🤖 Agent: ")
            response = chat(user_input)
            print(response)
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    run_interactive_cli()
