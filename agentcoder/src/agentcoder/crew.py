import os
from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from dotenv import load_dotenv

load_dotenv()

@CrewBase
class AgentCoderCrew():

    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    def __init__(self) -> None:
        self.openrouter_llm = LLM(
            model=os.getenv("OPENAI_MODEL_NAME"),
            base_url=os.getenv("OPENAI_API_BASE"),
            api_key=os.getenv("OPENROUTER_API_KEY")
        )

    # --- AGENTS ---
    @agent
    def programmer(self) -> Agent:
        return Agent(
            config=self.agents_config['programmer'],
            llm=self.openrouter_llm,
            verbose=True
        )

    @agent
    def test_designer(self) -> Agent:
        return Agent(
            config=self.agents_config['test_designer'],
            llm=self.openrouter_llm,
            verbose=True
        )

    # --- TASKS ---
    @task
    def coding_task(self) -> Task:
        return Task(config=self.tasks_config['coding_task'])

    @task
    def test_generation_task(self) -> Task:
        return Task(config=self.tasks_config['test_generation_task'])

    # --- ISOLATED MICRO-CREWS ---
    @crew
    def coding_crew(self) -> Crew:
        return Crew(
            agents=[self.programmer()],
            tasks=[self.coding_task()],
            process=Process.sequential,
            verbose=True
        )

    @crew
    def testing_crew(self) -> Crew:
        return Crew(
            agents=[self.test_designer()],
            tasks=[self.test_generation_task()],
            process=Process.sequential,
            verbose=True
        )