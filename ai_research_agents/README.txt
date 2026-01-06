AI Research Agents

This project, AI Research Agents, demonstrates a multi-agent workflow where AI agents autonomously search the web, summarize content, and provide structured insights, mimicking a small research team.

Features:

Web Search Agent: Uses DuckDuckGo to search for information online.

Insight Researcher Agent: Analyzes summaries, extracts key topics, and produces detailed insights.

Supervisor Agent: Coordinates the workflow, deciding which agent acts next or when to finish.

Automatic Logging: Saves the output of the Insight Researcher to insights.txt.

LangGraph Workflow: Agents are connected via a state graph, ensuring sequential and conditional execution.


Installation:

Clone the repository:

	git clone https://github.com/yourusername/ai-llm-projects.git
	cd ai-llm-projects


Create a virtual environment (recommended):

	python -m venv venv
	source venv/bin/activate  # Linux/macOS
	venv\Scripts\activate     # Windows


Install dependencies:

	pip install -r requirements.txt


Set your OpenAI API key:

	export OPENAI_API_KEY="your_api_key_here"   # Linux/macOS
	setx OPENAI_API_KEY "your_api_key_here"     # Windows

Usage:


Run the main agent workflow:

	python ai_research_agents.py

You can enter any topic in the console for the agents to research (e.g., “Write an article about the latest AI trends”).

Output will appear in the console.

Insights from the Insight_Researcher are automatically appended to insights.txt.

File Structure:
ai-llm-projects/
├── ai_research_agents.py   # Main multi-agent workflow
├── requirements.txt        # Exact dependency versions
├── insights.txt            # Auto-generated insights by the Insight Researcher
└── README.txt               # Project description and instructions


Workflow:

User provides an initial query.

Web Searcher fetches search results and summarizes them.

Insight Researcher analyzes the summary, identifies key topics, and generates structured insights.

Supervisor decides which agent should act next until the workflow finishes.

The final insights are saved automatically to a text file for easy reference.