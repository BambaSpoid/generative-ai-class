import os
from dotenv import load_dotenv
from crewai import Agent, LLM
from tools import tool

load_dotenv()

MODEL = os.getenv("MODEL", "ollama/llama3")

llm = LLM(
    model=MODEL,
    temperature=0.3,
    max_tokens=800,
    use_native=not MODEL.startswith("ollama/"),
)

news_researcher = Agent(
    role="Chercheur d’actualité",
    goal="Trouver des infos récentes et fiables sur {topic}, avec sources et points clés.",
    backstory="Tu es rigoureux, tu compares plusieurs sources et tu évites les rumeurs.",
    tools=[tool],
    llm=llm,
    verbose=True,
    memory=False,
    allow_delegation=False,
    max_iter=3,
)

news_writer = Agent(
    role="Journaliste / Rédacteur",
    goal="Rédiger un article clair et structuré sur {topic} à partir de la recherche.",
    backstory="Tu écris simple, tu hiérarchises l’info, et tu termines par une synthèse.",
    tools=[tool],
    llm=llm,
    verbose=True,
    memory=True,
    allow_delegation=False,
)
