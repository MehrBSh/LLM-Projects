import os
import operator
from typing import Annotated, Sequence, TypedDict, List, Dict, Any

import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langgraph.graph import StateGraph, END

# -----------------------------
# 0. Environment & Model Setup
# -----------------------------
# Disable LangChain tracing for simpler output
os.environ["LANGCHAIN_TRACING_V2"] = "false"

# Initialize OpenAI LLM (GPT-4-mini variant)
llm = ChatOpenAI(model="gpt-4o-mini")


# =====================================================
# File writer for Insight_Researcher output
# =====================================================
# Saves the insights produced by the Insight_Researcher agent to a text file
def save_insights_to_txt(text: str, filename="insights.txt"):
    with open(filename, "a", encoding="utf-8") as f:
        f.write(text)
        f.write("\n\n" + "-" * 80 + "\n\n")  # Separate multiple entries clearly


# -----------------------------
# 1. Utility functions (tools)
# -----------------------------
# Functions that fetch data 

def internet_search_raw(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """Search the internet using DuckDuckGo and return raw results as a list of dictionaries."""
    with DDGS() as ddgs:
        results = [r for r in ddgs.text(query, max_results=max_results)]
    return results


def process_content_raw(url: str) -> str:
    """Fetch and extract all text from a given webpage URL."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")
        text = soup.get_text(separator="\n")
        return text
    except Exception as e:
        return f"Error fetching {url}: {e}"


# -----------------------------
# 2. Define Agent State
# -----------------------------
# State used in LangGraph workflow to keep track of messages and next action
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]  # Stores all messages exchanged
    next: str  # Next agent to act (Web_Searcher, Insight_Researcher, or FINISH)


# -----------------------------
# 3. Worker nodes
# -----------------------------
def web_searcher_node(state: AgentState) -> Dict[str, Any]:
    """Node for Web_Searcher: searches the web and summarizes content for the user."""
    messages = list(state["messages"])
    
    # Take the last human message as the search query
    user_texts = [m.content for m in messages if isinstance(m, HumanMessage)]
    query = user_texts[-1] if user_texts else "latest AI technology trends in 2024"

    # Fetch search results
    results = internet_search_raw(query, max_results=5)

    # Convert results into readable snippets for LLM
    snippets = []
    for r in results:
        title = r.get("title", "")
        body = r.get("body", "")
        href = r.get("href", "")
        snippets.append(f"Title: {title}\nURL: {href}\nSnippet: {body}\n")

    combined = "\n\n".join(snippets) if snippets else "No search results found."

    # Prepare system & human messages for LLM
    system_msg = SystemMessage(
        content="You are a Web Searcher. Summarize the provided search results clearly for the user."
    )
    prompt_msg = HumanMessage(
        content=(
            f"User request:\n{query}\n\n"
            f"Search results:\n{combined}\n\n"
            "Summarize the key AI technology trends in 2024."
        )
    )

    # Generate response
    ai_response = llm.invoke([system_msg, prompt_msg])

    # Return as AIMessage for workflow
    worker_msg = AIMessage(content=ai_response.content, name="Web_Searcher")
    return {"messages": [worker_msg]}


def insight_researcher_node(state: AgentState) -> Dict[str, Any]:
    """Node for Insight_Researcher: analyzes summaries, extracts topics, and provides insights."""
    messages = list(state["messages"])
    
    # Use the last AI message (from Web_Searcher) as input
    ai_texts = [m.content for m in messages if isinstance(m, AIMessage)]
    base_text = ai_texts[-1] if ai_texts else "No prior summary available."

    # System message guiding the LLM to perform structured analysis
    system_msg = SystemMessage(
        content=(
            "You are an Insight Researcher.\n"
            "Given the content, you will:\n"
            "1) Identify key topics.\n"
            "2) Provide detailed insights for each topic.\n"
            "3) Include implications, opportunities, and risks.\n"
            "Return a clear, structured analysis."
        )
    )
    prompt_msg = HumanMessage(
        content=(
            "Content to analyze:\n\n"
            f"{base_text}\n\n"
            "Now perform your analysis as described."
        )
    )

    # Generate insights
    ai_response = llm.invoke([system_msg, prompt_msg])
    worker_msg = AIMessage(content=ai_response.content, name="Insight_Researcher")
    
    # -----------------------------
    # Save output to TXT file
    # -----------------------------
    save_insights_to_txt(ai_response.content)

    return {"messages": [worker_msg]}


# -----------------------------
# 4. Supervisor node
# -----------------------------
def supervisor_node(state: AgentState) -> Dict[str, Any]:
    """
    Supervisor decides which worker acts next or if the workflow should finish.
    Based on messages, outputs one of: Web_Searcher, Insight_Researcher, FINISH.
    """
    members = ["Web_Searcher", "Insight_Researcher"]
    options = ["FINISH"] + members

    messages = list(state["messages"])
    system_prompt = (
        "You are a supervisor overseeing workers: Web_Searcher, Insight_Researcher.\n"
        "Rules:\n"
        "- If web research hasn't been done, choose Web_Searcher.\n"
        "- If research is done but insights are not, choose Insight_Researcher.\n"
        "- If both steps are done, choose FINISH.\n"
        f"Respond with ONLY one of: {options}. No explanations."
    )

    system_msg = SystemMessage(content=system_prompt)
    convo = [system_msg] + messages
    decision_msg = llm.invoke(convo)

    # Normalize LLM choice
    raw_choice = decision_msg.content.strip().upper()
    if "WEB" in raw_choice:
        choice = "Web_Searcher"
    elif "INSIGHT" in raw_choice:
        choice = "Insight_Researcher"
    elif "FINISH" in raw_choice:
        choice = "FINISH"
    else:
        choice = "FINISH"  # Default to avoid infinite loops

    return {"next": choice}


# -----------------------------
# 5. Build LangGraph workflow
# -----------------------------
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("Web_Searcher", web_searcher_node)
workflow.add_node("Insight_Researcher", insight_researcher_node)
workflow.add_node("supervisor", supervisor_node)

# Worker → Supervisor edges
for member in ["Web_Searcher", "Insight_Researcher"]:
    workflow.add_edge(member, "supervisor")

# Supervisor → next node or END
conditional_map = {
    "Web_Searcher": "Web_Searcher",
    "Insight_Researcher": "Insight_Researcher",
    "FINISH": END,
}
workflow.add_conditional_edges("supervisor", lambda x: x["next"], conditional_map)

# Entry point of the workflow
workflow.set_entry_point("supervisor")

graph = workflow.compile()

"""
# This block demonstrates how to run the workflow with a fixed, predefined topic.
# You can replace the initial message with other topics as needed, or select for  6. Interactive console input .
# -----------------------------
# 6. Run the workflow
# -----------------------------
if __name__ == "__main__":
    initial_message = HumanMessage(
        content=(
            "Search for the latest AI technology trends in 2024, "
            "summarize the content, and have the insight researcher "
            "provide structured insights for each topic."
        )
    )

    for state_update in graph.stream({"messages": [initial_message], "next": ""}):
        if "__end__" in state_update:
            continue
        # Print intermediate workflow updates
        print(state_update)
        print("----")
"""

#  Interactive console version: enter a topic and see the workflow step by step.
# -----------------------------
# 6. Interactive console input 
# -----------------------------
if __name__ == "__main__":
    print("=== AI Research Agents Interactive ===")  
    topic = input("Enter the topic you want to research: ")  

    initial_message = HumanMessage(
        content=f"Please research this topic: {topic}"  
    )

    # Start streaming the workflow
    for state_update in graph.stream({"messages": [initial_message], "next": ""}):
        if "__end__" in state_update:
            continue
        
        print("\n[Workflow Update]")  
        print(state_update)

        for msg in state_update.get("messages", []):  
            print(f"{msg.name}: {msg.content}\n")  
        print("----")
