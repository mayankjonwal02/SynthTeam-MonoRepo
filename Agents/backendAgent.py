from crewai import Agent
from crewai import Task
from crewai import Crew
import asyncio

backend_Agent = Agent(
    role="Backend Engineer",
    goal="Build scalable backend systems using {language}",
    backstory=(
        "You are {user_name}, a top-tier engineer working at {company}, "
        "known for solving complex backend challenges in {language}."
    ),
    verbose=True,
    allow_delegation=False
)



task = Task(
    description="Design a microservice architecture using {language} for the company {company}.",
    agent=backend_Agent,
    expected_output="A proposed architecture with justification."
)



crew = Crew(agents=[custom_agent], tasks=[task])


async def main():
    result = await crew.kickoff(inputs={
        "user_name": "Mayank Jonwal",
        "company": "Karuna AI",
        "language": "Go"
    })
    print(result)

asyncio.run(main())
