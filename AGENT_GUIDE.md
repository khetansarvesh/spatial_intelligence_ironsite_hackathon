# SiteIQ Agent - Natural Language Query Guide

The SiteIQ Agent allows you to query productivity reports using natural language. It uses LLM function calling to provide accurate, data-driven answers.

## Setup

### 1. Set API Key

Choose either OpenAI or Anthropic:

**OpenAI (GPT-4o):**
```bash
export OPENAI_API_KEY=your_openai_api_key
```

**Anthropic (Claude):**
```bash
export ANTHROPIC_API_KEY=your_anthropic_api_key
```

### 2. Generate a Report

First, process a video to generate a productivity report:

```bash
python main.py --video construction_video.mp4 --output report.json
```

## Usage

### Single Query Mode

Ask a single question:

```bash
# Using OpenAI (default)
python query_agent.py --report report.json "What tools were used the most?"

# Using Anthropic/Claude
python query_agent.py --report report.json --provider anthropic "How can we improve productivity?"
```

### Interactive Mode

Have a conversation with the agent:

```bash
python query_agent.py --report report.json --interactive
```

Example session:
```
You: What was the overall productivity score?
SiteIQ: The overall productivity score was 75%, which is rated as "Good". The worker spent
2h 15m (75%) in productive activities and 45m (25%) in low-productivity activities including
15% idle time and 10% searching for tools.

You: Which tool was used the most?
SiteIQ: The drill was the most used tool, accounting for 1h 30m (68%) of total tool time.
It was used 12 times with an average duration of 7m 30s per use.

You: How can productivity be improved?
SiteIQ: Based on the analysis, here are the key recommendations:

1. Reduce idle time (currently 15%, target <10%)
2. Improve tool organization to minimize searching (currently 10% of time)
3. Batch similar tasks to reduce tool switching (8 switches detected)
4. Maintain the high productivity pattern from the first hour (85%)

You: quit
Goodbye!
```

### View Summary Without LLM

Get a quick text summary without using the LLM:

```bash
python query_agent.py --report report.json --summary
```

## Example Questions

### Productivity Metrics
- "What was the overall productivity score?"
- "How much time was spent being productive?"
- "What percentage of time was idle?"
- "How does this session compare to average?"

### Tool Usage
- "What tools were used the most?"
- "How long was the drill used?"
- "How many times did the worker switch tools?"
- "What activities were performed with the hammer?"

### Activity Analysis
- "What activities took the most time?"
- "How much time was spent searching?"
- "When was the worker most productive?"
- "Were there any long idle periods?"

### Comparisons
- "Compare the first half to the second half"
- "Compare the first 5 minutes to the last 5 minutes"
- "Compare 0-300 to 300-600" (time ranges in seconds)

### Insights & Recommendations
- "How can productivity be improved?"
- "What were the main issues?"
- "What recommendations do you have?"
- "What insights were generated?"

## Python API

### Basic Usage

```python
from src.agent import ProductivityAgent
from query_agent import load_report_from_json

# Load report
report = load_report_from_json("report.json")

# Initialize agent
agent = ProductivityAgent(
    report=report,
    provider="openai",  # or "anthropic"
)

# Ask questions
response = agent.chat("What was the overall productivity?")
print(response)

response = agent.chat("Which tool was used most?")
print(response)
```

### Using Agent Tools Directly

You can also use the agent tools directly without the LLM:

```python
from src.agent import AgentTools
from query_agent import load_report_from_json

report = load_report_from_json("report.json")
tools = AgentTools(report)

# Get productivity score
score_data = tools.get_productivity_score()
print(f"Score: {score_data['overall_score_percentage']:.1f}%")

# Get tool usage
tool_data = tools.get_tool_usage()
for tool in tool_data['tools']:
    print(f"{tool['name']}: {tool['total_time_formatted']}")

# Find idle periods
idle_data = tools.find_idle_periods(min_duration_seconds=30)
print(f"Idle periods: {idle_data['idle_period_count']}")
for period in idle_data['idle_periods']:
    print(f"  {period['start_time_formatted']} - {period['end_time_formatted']}")

# Get insights
insights = tools.get_insights_and_recommendations()
print("Insights:", insights['insights'])
print("Recommendations:", insights['recommendations'])
```

## Available Tools

The agent has access to 7 tools for querying data:

1. **get_activity_summary** - Summary of activities over time
2. **get_tool_usage** - Tool usage statistics
3. **find_idle_periods** - Find idle/unproductive periods
4. **get_productivity_score** - Overall productivity metrics
5. **compare_periods** - Compare two time periods
6. **get_insights_and_recommendations** - AI-generated insights
7. **get_time_breakdown** - Detailed time breakdown by activity

## Model Selection

### OpenAI Models
```bash
# GPT-4o (default, recommended)
python query_agent.py --report report.json --provider openai "your question"

# GPT-4o-mini (faster, cheaper)
python query_agent.py --report report.json --provider openai --model gpt-4o-mini "your question"
```

### Anthropic Models
```bash
# Claude 3.5 Sonnet (default, recommended)
python query_agent.py --report report.json --provider anthropic "your question"

# Claude 3 Haiku (faster, cheaper)
python query_agent.py --report report.json --provider anthropic --model claude-3-haiku-20240307 "your question"
```

## Tips for Better Results

1. **Be specific**: "How long was the drill used in the first 10 minutes?" is better than "Tell me about tools"

2. **Ask for numbers**: The agent provides precise times, percentages, and counts

3. **Request comparisons**: "Compare period X to period Y" for trend analysis

4. **Multiple questions**: In interactive mode, build on previous answers

5. **Use time ranges**: "Compare 0-300 to 300-600" for specific period comparisons

## Troubleshooting

### API Key Issues
```
Error: API key not provided
```
Solution: Set the environment variable:
- OpenAI: `export OPENAI_API_KEY=your_key`
- Anthropic: `export ANTHROPIC_API_KEY=your_key`

### Model Not Found
```
Error: Model not found
```
Solution: Check available models or use default by omitting --model

### Timeout Errors
If queries timeout, try:
- Using a smaller model (gpt-4o-mini or claude-haiku)
- Asking more specific questions
- Breaking complex queries into multiple simpler ones

## Cost Optimization

- **OpenAI GPT-4o-mini**: ~10x cheaper than GPT-4o, good for most queries
- **Anthropic Claude Haiku**: Fastest and cheapest option
- **Use --summary mode**: Free, instant text report without LLM
- **Batch questions**: Ask multiple questions in interactive mode (reuses context)

## Examples

### Complete Workflow

```bash
# 1. Process video
python main.py --video site_video.mp4 --output report.json

# 2. Quick summary (no LLM cost)
python query_agent.py --report report.json --summary

# 3. Interactive analysis (detailed)
python query_agent.py --report report.json --interactive --provider anthropic

# In interactive mode:
# - Ask specific questions
# - Get detailed insights
# - Request recommendations
# - Compare time periods
```

### Script Integration

```python
#!/usr/bin/env python3
"""Automated productivity analysis script."""

from main import SiteIQPipeline
from src.agent import ProductivityAgent

# Process video
pipeline = SiteIQPipeline(detector_backend="yolo")
report = pipeline.process_video("construction.mp4")

# Save report
pipeline._save_report(report, "report.json")

# Initialize agent
agent = ProductivityAgent(report, provider="anthropic")

# Automated queries
questions = [
    "What was the overall productivity score?",
    "Which tools were used the most?",
    "How can productivity be improved?",
]

print("Automated Productivity Analysis")
print("=" * 60)

for question in questions:
    print(f"\n{question}")
    answer = agent.chat(question)
    print(answer)
```

## Next Steps

- Try the interactive mode for conversational analysis
- Integrate the agent into your workflow automation
- Build custom dashboards using the agent tools
- Create scheduled reports with automated insights
