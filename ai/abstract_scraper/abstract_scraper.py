import json
import logging
from datetime import datetime
from typing import Dict, List, Union

import arxiv
from tqdm import tqdm

from categories_to_scrape import CATEGORIES

NOW = datetime.now()
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[
        logging.FileHandler(f"abstract_scraper/abstract_scraper_{NOW}.log"),
        logging.StreamHandler(),
    ],
)

OUTPUT_FILEPATH = f"abstract_scraper/dataset_{NOW}.json"


def create_entry(result: arxiv.Result) -> Dict[str, Union[str, int]]:
    authors = [author.name for author in result.authors]
    published_timestamp = result.published.timestamp()
    return {
        "id": result.entry_id,
        "primary_category": result.primary_category,
        "title": result.title,
        "authors": authors,
        "published": published_timestamp,
        "abstract": result.summary,
        "pdf_url": result.pdf_url,
    }


def get_papers_details_for_category(
    category: str,
) -> List[Dict[str, Union[str, int]]]:
    papers = []
    search = arxiv.Search(
        query=f"cat:{category}",
        max_results=float("inf"),
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )

    for result in tqdm(search.results()):
        entry = create_entry(result)
        papers.append(entry)

    return papers


def main() -> None:
    logging.info("Starting to scrape arxiv.org")
    dataset = []

    for category, subcategories in CATEGORIES.items():
        for subcategory in subcategories:
            logging.info(f"Scraping category {category}, subcategory: {subcategory}")
            papers = get_papers_details_for_category(subcategory)
            dataset.extend(papers)

    logging.info("Deduplicating dataset")
    deduplicated_dataset = list({paper["id"]: paper for paper in dataset}.values())
    logging.info(f"{len(deduplicated_dataset)} abstracts retrieved")

    logging.info(f"Saving dataset to {OUTPUT_FILEPATH}")
    with open(OUTPUT_FILEPATH, "w") as file:
        json.dump(deduplicated_dataset, file)


if __name__ == "__main__":
    main()
