# Building a Reliable AI Agent from Scratch with a Small Local LLM

This repository contains the companion notebook for a two-part tutorial series on Medium:

- **[Part 1: Prompts, reflection and constrained decoding](PART1_URL)**
- **[Part 2: Tools, parallel calls and LangGraph](PART2_URL)**

The posts explain the ideas and show the key code. The notebook has everything: the full code, all outputs, and the evaluation harness.

## What's in the notebook

[`agent_from_scratch.ipynb`](agent_from_scratch.ipynb) builds a ReAct agent from scratch on top of [Phi-4-mini-instruct](https://huggingface.co/microsoft/Phi-4-mini-instruct) (3.8B parameters) and improves it one stage at a time. The agent is a stock-market assistant working in a **fictional market** with frozen prices and deliberate traps, so every run is reproducible and every answer can be scored automatically.

After every stage, the agent runs the same 12-question test set, and the results go into a scoreboard:

| Stage | What we add | Correct (of 12) |
|---|---|---|
| 0 | Baseline ReAct loop with a text format | 5 |
| 1 | A finish rule in the prompt | 4 |
| 2 | Self-correction (reflection) | 6 |
| 3 | Constrained JSON decoding with [Outlines](https://github.com/dottxt-ai/outlines) | 5 |
| 4 | A company search tool | 11 |
| 5 | Parallel tool calls | 12 |
| Bonus | Stage 5 rebuilt with [LangGraph](https://github.com/langchain-ai/langgraph) | 12 |

The notebook also looks at how the prompt grows with every improvement, and it includes an optional appendix that connects the agent to live data from Yahoo Finance.

## Running it

**Requirements:** Python 3.12 and a GPU with about 8 GB of memory (Phi-4-mini in bfloat16). The code was tested with `transformers` 5.x, `outlines` 1.3, `langgraph` 1.2 and `langchain-core` 1.6.

```bash
pip install torch "transformers>=4.56" accelerate "outlines>=1.0,<2" "pydantic>=2" pandas matplotlib
pip install langgraph langchain-core    # for the LangGraph section
pip install "yfinance>=0.2.54"          # optional, only for the live-data appendix
```

Then open the notebook and run all cells from top to bottom in one session. The LangGraph section compares its results with the earlier stages, so it needs them in memory. The full evaluation runs every question at every stage, about 84 agent runs, so expect it to take a while.

All generation is greedy (`do_sample=False`), so with the same library versions you should get the same results as in the saved outputs.
