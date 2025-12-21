"""Main entry point for domain content generation.

This script generates a complete domain with topics, books, and articles
using the hierarchical LangGraph-based generation system.

Usage:
    python -m src.main
"""

import json
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from src.config import config
from src.graphs import build_domain_graph
from src.utils import logger

# Load environment variables
load_dotenv()


def main():
    """Run domain generation for Brazil."""
    # Configuration
    domain_name = "Brazil"
    domain_description = "Brazilian culture, history, geography, and society"
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / timestamp
    run_dir.mkdir(exist_ok=True)

    logger.info("=" * 80)
    logger.info("Domain Content Generation")
    logger.info("=" * 80)
    logger.info(f"Domain: {domain_name}")
    logger.info(f"Description: {domain_description}")
    logger.info(f"Model: {config.model_name}")
    logger.info(f"Topics: {config.topics_min_items}-{config.topics_max_items}")
    logger.info(f"Chapters per book: {config.toc_min_items}-{config.toc_max_items}")
    logger.info(f"Articles per topic: {config.articles_min_per_topic}")
    logger.info(
        f"Grounded QA: {config.grounded_qa_min_items}-{config.grounded_qa_max_items}"
    )
    logger.info(
        f"Ungrounded QA: {config.ungrounded_qa_min_items}-{config.ungrounded_qa_max_items}"
    )
    logger.info(f"Run output directory: {run_dir}")
    logger.info("=" * 80)

    # Build the domain graph
    logger.info("Building domain generation graph...")
    domain_graph = build_domain_graph()

    # Run domain generation
    logger.info("Starting domain generation...")
    result = domain_graph.invoke(
        {
            "name": domain_name,
            "description": domain_description,
        }
    )

    # Extract domain
    domain = result["domain"]

    # Log results
    logger.info("=" * 80)
    logger.info("Generation Complete!")
    logger.info("=" * 80)
    logger.info(f"Topics generated: {len(domain.topics)}")
    for topic in domain.topics:
        logger.info(f"  - {topic.name}: {topic.description}")

    logger.info(f"\nBooks generated: {len(domain.books)}")
    for book in domain.books:
        logger.info(
            f"  - {book.title} ({len(book.chapters)} chapters, "
            f"{len(book.grounded_questions)} grounded QA, "
            f"{len(book.ungrounded_questions)} ungrounded QA)"
        )

    logger.info(f"\nArticles generated: {len(domain.articles)}")
    for article in domain.articles:
        logger.info(
            f"  - {article.title} ({len(article.sections)} sections, "
            f"{len(article.grounded_questions)} grounded QA, "
            f"{len(article.ungrounded_questions)} ungrounded QA)"
        )

    # Save outputs
    output_file = run_dir / "domain.json"
    logger.opt(colors=True).info(
        "\n<green>Saving domain JSON</green> to <yellow>{}</yellow>...", output_file
    )

    domain_dict = domain.model_dump()
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(domain_dict, f, indent=2, ensure_ascii=False)

    # Save graph visualizations (multiple formats for compatibility)
    try:
        # Try PNG export with pygraphviz (requires GraphViz installed)
        try:
            graph_png_path = run_dir / "agent_graph.png"
            domain_graph.get_graph().draw_png(str(graph_png_path))
            logger.opt(colors=True).info(
                "<green>Saved PNG graph visualization</green> to <yellow>{}</yellow>",
                graph_png_path,
            )
        except Exception as png_exc:
            logger.opt(colors=True).debug(
                "<dim>PNG export not available: {}</dim>", png_exc
            )

        # Always save Mermaid diagram (works without external dependencies)
        mermaid_path = run_dir / "agent_graph.mmd"
        mermaid_diagram = domain_graph.get_graph().draw_mermaid()
        with open(mermaid_path, "w", encoding="utf-8") as f:
            f.write(mermaid_diagram)
        logger.opt(colors=True).info(
            "<green>Saved Mermaid graph diagram</green> to <yellow>{}</yellow>",
            mermaid_path,
        )

        # Also save as HTML for easy viewing
        html_path = run_dir / "agent_graph.html"
        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Domain Generation Graph - {domain_name}</title>
    <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
    <script>mermaid.initialize({{ startOnLoad: true, theme: 'default' }});</script>
</head>
<body>
    <h1>Domain Generation Graph: {domain_name}</h1>
    <div class="mermaid">
{mermaid_diagram}
    </div>
</body>
</html>"""
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        logger.opt(colors=True).info(
            "<green>Saved HTML graph visualization</green> to <yellow>{}</yellow> (open in browser)",
            html_path,
        )
    except Exception as exc:  # pragma: no cover - visualization optional
        logger.opt(colors=True).warning(
            "<yellow>Could not generate graph visualizations</yellow>: {}", exc
        )

    file_size = output_file.stat().st_size / 1024  # KB
    logger.opt(colors=True).info(
        "<green>Saved domain JSON!</green> File size: <yellow>{:.2f} KB</yellow>",
        file_size,
    )
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
