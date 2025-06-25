import json
import argparse
import logging
from pathlib import Path
from typing import Set, Dict, Any, Optional

from datasets import load_dataset

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_keywords(filepath: Path) -> Set[str]:
    """Loads keywords from a file, one keyword per line."""
    if not filepath.is_file():
        logger.error(f"Keyword file not found: {filepath}")
        raise FileNotFoundError(f"Keyword file not found: {filepath}")
    
    keywords: Set[str] = set()
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            keyword = line.strip().lower()
            if keyword: # Avoid adding empty lines
                keywords.add(keyword)
    logger.info(f"Loaded {len(keywords)} unique keywords from {filepath}")
    return keywords

def _save_json(data: Dict[str, Dict[str, Any]], output_path: Path) -> None:
    """Helper function to save data dictionary to JSON file."""
    temp_output_path = output_path.with_suffix(output_path.suffix + '.tmp')
    try:
        with open(temp_output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
        # Rename temporary file to the final destination file atomically
        temp_output_path.rename(output_path)
    except IOError as e:
        logger.error(f"Failed to write output file {output_path}: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred during saving to {output_path}: {e}")


def filter_and_save_pubmed(
    keywords: Set[str],
    output_path: Path,
    max_abstracts: Optional[int] = None,
    save_interval: int = 1000 # New parameter for save frequency
) -> None:
    """
    Streams PubMed, filters abstracts based on keywords, saves them incrementally,
    and performs a final save.

    Args:
        keywords: A set of lowercase keywords to filter by.
        output_path: The path to save the filtered JSON data.
        max_abstracts: The maximum number of matching abstracts to save. 
                       If None or <= 0, saves all matches.
        save_interval: How often (in terms of found abstracts) to save the file.
    """
    logger.info(f"Starting PubMed stream and filtering process. Saving every {save_interval} abstracts.")
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    filtered_data: Dict[str, Dict[str, Any]] = {}
    if output_path.exists():
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                filtered_data = json.load(f)
            logger.info(f"Loaded {len(filtered_data)} existing abstracts from {output_path}. Will add new finds.")
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Could not load existing data from {output_path}: {e}. Starting fresh.")
            filtered_data = {} 

    pubmed_stream = load_dataset('pubmed', streaming=True, split='train')
    
    processed_count = 0
    # Adjust found_count based on potentially loaded data
    initial_found_count = len(filtered_data)
    found_count = initial_found_count 

    try:
        for entry in pubmed_stream:
            processed_count += 1
            try:
                # Extract required fields
                medline_citation = entry.get('MedlineCitation', {})
                pmid_obj = medline_citation.get('PMID')
                
                if not pmid_obj:
                    continue 

                # PMID can sometimes be an object with text content
                pmid = str(pmid_obj) if not isinstance(pmid_obj, str) else pmid_obj
                
                if pmid in filtered_data:
                    # logger.debug(f"PMID {pmid} already exists in loaded data. Skipping.")
                    if processed_count % 50000 == 0:
                         logger.info(f"Processed {processed_count} entries (skipping many existing)...")
                    continue

                article = medline_citation.get('Article', {})
                abstract = article.get('Abstract', {})
                
                year_info = medline_citation.get('DateCompleted') or medline_citation.get('DateRevised') or article.get('Journal', {}).get('JournalIssue', {}).get('PubDate')
                year = None
                if year_info and 'Year' in year_info:
                    year = year_info.get('Year')

                if year < 2010:
                    continue

                abstract_text = abstract.get('AbstractText')
                abstract_title = article.get('ArticleTitle')

                if not abstract_text or not abstract_title:
                    continue
                    
                search_content = (abstract_title.lower() if abstract_title else "") + " " + (abstract_text.lower() if abstract_text else "")

                if any(keyword in search_content for keyword in keywords):
                    
                    # Add the new abstract
                    filtered_data[pmid] = {
                        "year": year,
                        "abstract_text": abstract_text,
                        "abstract_title": abstract_title
                    }
                    found_count += 1 # Increment count *only* for newly added abstracts

                    # Log progress for newly found items
                    newly_found_count = found_count - initial_found_count
                    if newly_found_count % 100 == 0 and newly_found_count > 0:
                        logger.info(f"Processed: {processed_count}, Total Found: {found_count} (Added {newly_found_count} new)")

                    if newly_found_count > 0 and newly_found_count % save_interval == 0:
                        logger.info(f"Reached {newly_found_count} new abstracts. Performing intermediate save...")
                        _save_json(filtered_data, output_path)
                        logger.info(f"Intermediate save complete. Total abstracts saved: {len(filtered_data)}")

                    # Check if max_abstracts limit is reached (based on total found)
                    if max_abstracts is not None and max_abstracts > 0 and found_count >= max_abstracts:
                        logger.info(f"Reached target abstract limit of {max_abstracts} total abstracts. Stopping.")
                        break 
            
            except Exception as e:
                logger.error(f"Error processing entry near count {processed_count}: {e}. Entry snippet: {str(entry)[:200]}...", exc_info=False)
                continue 

            if processed_count % 10000 == 0: 
                logger.info(f"Processed {processed_count} entries from stream...")

    except Exception as e:
        logger.exception(f"A critical error occurred during stream processing: {e}")
    finally:
        logger.info("Processing loop finished or interrupted. Performing final save...")
        _save_json(filtered_data, output_path)
        logger.info(f"Final save complete. Total abstracts in file: {len(filtered_data)}")
        logger.info(f"Finished processing. Total entries scanned from stream: {processed_count}. Total abstracts in file: {len(filtered_data)}.")

def main() -> None:
    """Main function to parse arguments and run the filtering process."""
    parser = argparse.ArgumentParser(description="Filter PubMed abstracts based on keywords and save incrementally.")
    parser.add_argument(
        "--keywords_file",
        type=Path,
        required=True,
        help="Path to the text file containing keywords (one per line)."
    )
    parser.add_argument(
        "--output_json",
        type=Path,
        required=True,
        help="Path to save the filtered PubMed abstracts in JSON format."
    )
    parser.add_argument(
        "--max_abstracts",
        type=int,
        default=None, # Default to None (no limit)
        help="Maximum number of *total* matching abstracts to save (including previously saved). Default is no limit."
    )
    parser.add_argument(
        "--save_interval",
        type=int,
        default=1000,
        help="Save the JSON file every N *newly found* abstracts. Default is 1000."
    )

    args = parser.parse_args()

    save_interval = args.save_interval
    if save_interval <= 0:
        logger.warning("save_interval must be positive. Using default value of 1000.")
        save_interval = 1000

    max_abstracts_limit = args.max_abstracts
    if max_abstracts_limit is not None and max_abstracts_limit <= 0:
        logger.warning("max_abstracts is non-positive, setting to no limit.")
        max_abstracts_limit = None

    try:
        keywords = load_keywords(args.keywords_file)
        if not keywords:
            logger.warning("No keywords loaded. Exiting.")
            return
        filter_and_save_pubmed(
            keywords, 
            args.output_json, 
            max_abstracts_limit, 
            save_interval
        )
    except FileNotFoundError:
        pass
    except Exception as e:
        logger.exception(f"An unexpected error occurred: {e}") 

if __name__ == "__main__":
    main()
