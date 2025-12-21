# Domain Content Generation System

A hierarchical content generation system using LangGraph for creating structured educational content with grounded and ungrounded QA pairs for machine unlearning research.

## Structure

```
src/
├── config.py           # Configuration parameters
├── models.py           # Pydantic data models
├── state.py            # LangGraph state classes
├── utils.py            # Utility functions (logging, LLM init)
├── main.py             # Main entry point
└── graphs/             # LangGraph graph definitions
    ├── book_graph.py   # Book generation subgraph
    ├── article_graph.py # Article generation subgraph
    └── domain_graph.py  # Main domain orchestration graph
```

## Usage

### Run from command line:

```bash
python -m src.main
```

### Configuration

Parameters can be adjusted in `src/config.py` or via environment variables with the `GEN_` prefix:

```bash
export GEN_TOPICS_MIN_ITEMS=3
export GEN_TOPICS_MAX_ITEMS=5
export GEN_MODEL_NAME="gpt-4"
```

### Customize Domain

Edit `src/main.py` to change the domain:

```python
domain_name = "Machine Learning"
domain_description = "Artificial intelligence, neural networks, and ML algorithms"
```

## Output

Generated content is saved to `output/domain_<name>.json` with the following structure:

```json
{
  "name": "Brazil",
  "description": "Brazilian culture, history, geography, and society",
  "topics": [...],
  "books": [
    {
      "title": "...",
      "chapters": [...],
      "grounded_questions": [...],
      "ungrounded_questions": [...]
    }
  ],
  "articles": [...]
}
```

## Architecture

### Hierarchical Generation

1. **Domain Level** (Main Graph)
   - Generates topics for a domain
   - Orchestrates book and article generation
   - Compiles final `Domain` object

2. **Book Subgraph**
   - Plans book structure (TOC)
   - Writes chapters in parallel
   - Generates **grounded QA** (answerable from book)
   - Generates **ungrounded QA** (not answerable, tests spillover)

3. **Article Subgraph**
   - Plans article structure
   - Writes sections in parallel
   - Generates grounded and ungrounded QA pairs

### Key Features

✅ **Grounded vs Ungrounded QA**: Test both retention (grounded) and spillover effects (ungrounded)  
✅ **Parallel Generation**: Chapters/sections written simultaneously  
✅ **Modular Subgraphs**: Easy to add new content types  
✅ **Fully Structured**: Pydantic models with validation  
✅ **Reference Tracking**: QA pairs linked to chapters/sections

## Development

### Testing in Notebook

Use the Jupyter notebook at `notebook/domain_unlearn/01_generation.ipynb` for interactive testing and experimentation.

### Adding New Content Types

1. Create new models in `models.py`
2. Add state classes in `state.py`
3. Create new subgraph in `graphs/`
4. Integrate into `domain_graph.py`
