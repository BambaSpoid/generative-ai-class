import os
from dotenv import load_dotenv
from crewai import Crew, Process
from agents import news_researcher, news_writer
from tasks import research_task, write_task

load_dotenv()

topic = os.getenv("TOPIC", "IA au Sénégal")

crew = Crew(
    agents=[news_researcher, news_writer],
    tasks=[research_task, write_task],
    process=Process.sequential,
    verbose=True,
)

result = crew.kickoff(inputs={"topic": topic})
print("\n\n=== RESULT ===\n")
print(result)
print("\nFichier généré: outputs/nouvel-article-de-blog.md\n")
