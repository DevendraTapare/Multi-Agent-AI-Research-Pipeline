````markdown
# Multi-Agent AI Research Pipeline

🚀 A fully autonomous AI-powered research system inspired by ChatGPT's Deep Research capability.

This project demonstrates how multiple AI agents can collaborate to perform end-to-end research automatically: searching the web, reading sources, generating structured reports, and critically evaluating the final output.

The goal is to build an intelligent research workflow with no usage limits, leveraging modern LangChain agents, LCEL pipelines, web search tools, and LLM reasoning.

---

## Features

- Autonomous multi-agent workflow
- Live web search using Tavily Search API
- Automated webpage reading and content extraction
- AI-generated structured research reports
- AI-powered report evaluation and critique
- Streamlit-based user interface
- Modern LangChain `create_agent` architecture
- LCEL (LangChain Expression Language) pipelines
- Shared state management between agents
- Modular and extensible design

---

## Architecture Overview

The system consists of four sequential AI components:

```text
User Topic
    │
    ▼
Search Agent
    │
    ▼
Reader Agent
    │
    ▼
Writer Chain
    │
    ▼
Critic Chain
    │
    ▼
Final Research Report + Feedback
````

---

## Agent 1 — Search Agent

### Purpose

Find relevant information from the internet.

### Technologies

* LangChain `create_agent`
* Tavily Search API

### Responsibilities

* Searches the web for the given topic
* Retrieves:

  * URLs
  * Titles
  * Snippets
* Returns up to 5 relevant sources

### Output

```python
state["search_results"]
```

---

## Agent 2 — Reader Agent

### Purpose

Visit and read the discovered sources.

### Technologies

* LangChain `create_agent`
* BeautifulSoup4

### Responsibilities

* Opens URLs returned by Search Agent
* Extracts clean article text
* Removes unnecessary HTML content
* Limits extraction to ~3000 words per source

### Output

```python
state["scraped_content"]
```

---

## Chain 3 — Writer Chain

### Purpose

Generate a professional research report.

### Technologies

* ChatPromptTemplate
* OpenAI LLM
* StructuredOutputParser
* LCEL Pipeline

### Workflow

```python
prompt | llm | parser
```

### Report Structure

* Introduction
* Key Findings
* Conclusion
* Sources

### Output

```python
state["report"]
```

---

## Chain 4 — Critic Chain

### Purpose

Evaluate the quality of the generated report.

### Technologies

* ChatPromptTemplate
* OpenAI LLM
* StructuredOutputParser
* LCEL Pipeline

### Evaluation Criteria

* Research Quality
* Completeness
* Clarity
* Source Usage

### Output Format

* Score (/10)
* Strengths
* Areas for Improvement
* Final Verdict

### Output

```python
state["feedback"]
```

---

## Shared State Flow

```text
topic
    │
    ▼
state["search_results"]
    │
    ▼
state["scraped_content"]
    │
    ▼
state["report"]
    │
    ▼
state["feedback"]
```

The shared state dictionary enables seamless communication between agents and chains without recomputation.

---

## Project Structure

```text
multi_agent_system/
│
├── app.py
├── agents.py
├── pipeline.py
├── tools.py
├── requirements.txt
├── .env
└── README.md
```

### File Descriptions

| File             | Purpose                         |
| ---------------- | ------------------------------- |
| app.py           | Streamlit frontend              |
| agents.py        | Agent creation and LCEL chains  |
| tools.py         | Search and scraping tools       |
| pipeline.py      | Research orchestration workflow |
| requirements.txt | Python dependencies             |
| .env             | API keys and configuration      |

---

## Technology Stack

### AI Frameworks

* LangChain
* LCEL
* OpenAI API

### Search

* Tavily Search API

### Data Extraction

* BeautifulSoup4
* Requests

### Frontend

* Streamlit

### Environment Management

* Python Dotenv

---

## Why `create_agent`?

This project intentionally uses:

```python
create_agent()
```

instead of:

```python
create_react_agent()
```

### Reasons

* Aligns with modern LangChain architecture
* Better long-term compatibility
* Cleaner implementation
* Future-proof against deprecations
* Recommended for new agent-based applications

---

## Installation

### Clone Repository

```bash
git clone https://github.com/DevendraTapare/Multi-Agent-AI-Research-Pipeline.git

cd Multi-Agent-AI-Research-Pipeline
```

### Create Virtual Environment

```bash
python -m venv .venv
```

### Activate Environment

**Windows**

```bash
.venv\Scripts\activate
```

**Mac/Linux**

```bash
source .venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Environment Variables

Create a `.env` file:

```env
OPENAI_API_KEY=your_openai_api_key
TAVILY_API_KEY=your_tavily_api_key
```

---

## Run the Application

```bash
streamlit run app.py
```

Application will launch locally:

```text
http://localhost:8501
```

---

## Example Workflow

### Input

```text
Impact of Agentic AI on Enterprise Software
```

### Generated Output

* Web Search Results
* Scraped Research Content
* Structured Research Report
* AI Quality Assessment

---

## Future Enhancements

* LangGraph-based orchestration
* Multi-agent parallel execution
* Source citation validation
* PDF export functionality
* Research memory and vector storage
* Multi-model support (OpenAI, Claude, Gemini)
* Real-time streaming responses
* Research report versioning

---

## Learning Outcomes

This project demonstrates practical experience with:

* Agentic AI Systems
* Multi-Agent Architectures
* LangChain
* LCEL Pipelines
* Prompt Engineering
* Web Search Automation
* Web Scraping
* LLM Application Development
* Streamlit Deployment
* AI Workflow Orchestration

---

## Author

**Devendra Tapare**

AI Engineer | Agentic AI Developer | LLM Applications | Multi-Agent Systems

Currently focused on building production-grade AI agents, autonomous research systems, and enterprise AI solutions.

---

## License

MIT License

Feel free to use, modify, and extend this project for educational and commercial purposes.

```
```
