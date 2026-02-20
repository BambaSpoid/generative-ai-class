from crewai import Task
from agents import news_researcher, news_writer

research_task = Task(
    description=(
        "Recherche l’actualité sur le thème: {topic}. "
        "Donne 5 faits récents, 5 liens/sources, et 5 bullet points de contexte."
    ),
    expected_output="Une note de recherche structurée avec faits + sources + contexte.",
    agent=news_researcher,
)

write_task = Task(
    description=(
        "Écris un article en français basé sur la recherche. Format:\n"
        "- Titre\n- Intro (3-4 lignes)\n"
        "- 3 sections (avec sous-titres)\n"
        "- Conclusion + 'À retenir' (5 bullets)\n"
        "- Sources (liens)\n"
    ),
    expected_output="Un article complet en markdown.",
    agent=news_writer,
    output_file="outputs/nouvel-article-de-blog.md",
)
