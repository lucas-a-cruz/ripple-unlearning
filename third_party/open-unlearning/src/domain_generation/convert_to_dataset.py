"""Convert domain generation output to HuggingFace dataset format for unlearning.

This script takes the domain.json output from domain generation and converts it
into a format compatible with the open-unlearning framework:
- Grounded QA pairs for the "forget" dataset
- Full text content for pretraining-style unlearning
"""

import json
from pathlib import Path
from typing import List, Dict, Any
import argparse

from datasets import Dataset, DatasetDict
from loguru import logger


def extract_qa_pairs(domain_data: Dict[str, Any]) -> List[Dict[str, str]]:
    """Extract all grounded QA pairs from domain data.

    Args:
        domain_data: Parsed domain.json dictionary

    Returns:
        List of {"question": str, "answer": str} dictionaries
    """
    qa_pairs = []

    # Extract from books
    for book in domain_data.get("books", []):
        for qa in book.get("grounded_questions", []):
            qa_pairs.append({
                "question": qa["question"],
                "answer": qa["answer"],
                "source": f"Book: {book['title']}",
                "topic": book["topic"]
            })

    # Extract from articles
    for article in domain_data.get("articles", []):
        for qa in article.get("grounded_questions", []):
            qa_pairs.append({
                "question": qa["question"],
                "answer": qa["answer"],
                "source": f"Article: {article['title']}",
                "topic": article["topic"]
            })

    logger.info(f"Extracted {len(qa_pairs)} grounded QA pairs")
    return qa_pairs


def extract_full_text_content(domain_data: Dict[str, Any]) -> List[Dict[str, str]]:
    """Extract full text content from books and articles.

    Args:
        domain_data: Parsed domain.json dictionary

    Returns:
        List of {"text": str, "source": str} dictionaries
    """
    text_samples = []

    # Extract book content (chapter by chapter)
    for book in domain_data.get("books", []):
        for chapter in book.get("chapters", []):
            # Combine all sections of a chapter into one text
            chapter_text_parts = [f"# {chapter['title']}\n\n"]
            for section in chapter.get("sections", []):
                chapter_text_parts.append(f"## {section['name']}\n\n{section['content']}\n\n")

            full_chapter_text = "".join(chapter_text_parts)
            text_samples.append({
                "text": full_chapter_text,
                "source": f"Book: {book['title']} - Chapter {chapter['idx']}: {chapter['title']}",
                "topic": book["topic"]
            })

    # Extract article content
    for article in domain_data.get("articles", []):
        article_text_parts = [
            f"# {article['title']}\n\n",
            f"**Abstract:** {article['abstract']}\n\n"
        ]

        for section in article.get("sections", []):
            article_text_parts.append(f"## {section['name']}\n\n{section['content']}\n\n")

        full_article_text = "".join(article_text_parts)
        text_samples.append({
            "text": full_article_text,
            "source": f"Article: {article['title']}",
            "topic": article["topic"]
        })

    logger.info(f"Extracted {len(text_samples)} full text samples")
    return text_samples


def create_datasets(
    domain_json_path: Path,
    output_dir: Path,
    dataset_name: str,
    split_ratio: float = 0.8
):
    """Create HuggingFace datasets from domain.json.

    Args:
        domain_json_path: Path to domain.json file
        output_dir: Output directory for datasets
        dataset_name: Name of the dataset (e.g., "brazil", "usa_history")
        split_ratio: Ratio to split QA pairs into forget/retain (default 0.8)
    """
    logger.info(f"Loading domain data from {domain_json_path}")
    with open(domain_json_path, "r", encoding="utf-8") as f:
        domain_data = json.load(f)

    domain_name = domain_data["name"]
    logger.info(f"Processing domain: {domain_name}")

    # Create output directory
    output_dir = output_dir / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract QA pairs
    qa_pairs = extract_qa_pairs(domain_data)

    # Split QA pairs into forget and retain
    split_idx = int(len(qa_pairs) * split_ratio)
    forget_qa = qa_pairs[:split_idx]
    retain_qa = qa_pairs[split_idx:]

    logger.info(f"Split: {len(forget_qa)} forget, {len(retain_qa)} retain")

    # Create QA datasets
    forget_dataset = Dataset.from_list(forget_qa)
    retain_dataset = Dataset.from_list(retain_qa)

    qa_dataset_dict = DatasetDict({
        "forget": forget_dataset,
        "retain": retain_dataset,
    })

    # Save QA datasets
    qa_output_path = output_dir / "qa_dataset"
    qa_dataset_dict.save_to_disk(str(qa_output_path))
    logger.info(f"Saved QA dataset to {qa_output_path}")

    # Extract full text content for pretraining-style unlearning
    text_samples = extract_full_text_content(domain_data)

    # Split text samples into forget and retain
    text_split_idx = int(len(text_samples) * split_ratio)
    forget_text = text_samples[:text_split_idx]
    retain_text = text_samples[text_split_idx:]

    logger.info(f"Text split: {len(forget_text)} forget, {len(retain_text)} retain")

    # Create text datasets
    forget_text_dataset = Dataset.from_list(forget_text)
    retain_text_dataset = Dataset.from_list(retain_text)

    text_dataset_dict = DatasetDict({
        "forget": forget_text_dataset,
        "retain": retain_text_dataset,
    })

    # Save text datasets
    text_output_path = output_dir / "text_dataset"
    text_dataset_dict.save_to_disk(str(text_output_path))
    logger.info(f"Saved text dataset to {text_output_path}")

    # Save metadata
    metadata = {
        "domain_name": domain_name,
        "dataset_name": dataset_name,
        "num_topics": len(domain_data.get("topics", [])),
        "num_books": len(domain_data.get("books", [])),
        "num_articles": len(domain_data.get("articles", [])),
        "qa_forget_size": len(forget_qa),
        "qa_retain_size": len(retain_qa),
        "text_forget_size": len(forget_text),
        "text_retain_size": len(retain_text),
        "split_ratio": split_ratio,
    }

    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved metadata to {metadata_path}")

    logger.success(f"✅ Dataset creation complete for '{dataset_name}'!")
    logger.info(f"   - QA dataset: {qa_output_path}")
    logger.info(f"   - Text dataset: {text_output_path}")
    logger.info(f"   - Metadata: {metadata_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert domain generation output to HuggingFace dataset format"
    )
    parser.add_argument(
        "domain_json",
        type=Path,
        help="Path to domain.json file"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/domain_datasets"),
        help="Output directory for datasets (default: data/domain_datasets)"
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        required=True,
        help="Name of the dataset (e.g., 'brazil', 'usa_history')"
    )
    parser.add_argument(
        "--split-ratio",
        type=float,
        default=0.8,
        help="Ratio to split into forget/retain (default: 0.8)"
    )

    args = parser.parse_args()

    create_datasets(
        domain_json_path=args.domain_json,
        output_dir=args.output_dir,
        dataset_name=args.dataset_name,
        split_ratio=args.split_ratio
    )


if __name__ == "__main__":
    main()
