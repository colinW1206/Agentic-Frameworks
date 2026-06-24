from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
import os
from dotenv import load_dotenv

load_dotenv()

@CrewBase
class AgileCoderCrew():

    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    def __init__(self) -> None:
        self.openrouter_llm = LLM(
            model=os.getenv("OPENAI_MODEL_NAME"),
            base_url=os.getenv("OPENAI_API_BASE"),
            api_key=os.getenv("OPENROUTER_API_KEY")
        )

    @agent
    def product_manager(self) -> Agent: return Agent(config=self.agents_config['product_manager'], llm=self.openrouter_llm, verbose=True)

    @agent
    def scrum_master(self) -> Agent: return Agent(config=self.agents_config['scrum_master'], llm=self.openrouter_llm, verbose=True)

    @agent
    def developer(self) -> Agent: return Agent(config=self.agents_config['developer'], llm=self.openrouter_llm, verbose=True)

    @agent
    def senior_developer(self) -> Agent: return Agent(config=self.agents_config['senior_developer'], llm=self.openrouter_llm, verbose=True)

    @agent
    def tester(self) -> Agent: return Agent(config=self.agents_config['tester'], llm=self.openrouter_llm, verbose=True)

    @task
    def backlog_creation_task(self) -> Task: return Task(config=self.tasks_config['backlog_creation'])

    @task
    def sprint_planning_task(self) -> Task: return Task(config=self.tasks_config['sprint_planning'])

    @task
    def implementation_task(self) -> Task: return Task(config=self.tasks_config['implementation'])

    @task
    def code_review_task(self) -> Task: return Task(config=self.tasks_config['code_review'])

    @task
    def verification_task(self) -> Task: return Task(config=self.tasks_config['verification'])

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True
        )