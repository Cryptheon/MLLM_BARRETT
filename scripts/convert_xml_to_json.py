import xml.etree.ElementTree as ET
import json
import argparse
from typing import List, Dict, Any

def convert_xml(input_filepath: str, output_filepath: str, output_format: str) -> None:
    """
    Reads an XML file and converts it to either JSON or JSONL format.

    Args:
        input_filepath (str): Path to the input XML file.
        output_filepath (str): Path to the output file.
        output_format (str): The desired format ('json' or 'jsonl').
    """
    try:
        # Read the content from the input XML file
        with open(input_filepath, 'r', encoding='utf-8') as file:
            xml_data = file.read()

        root = ET.fromstring(xml_data)
        
        # Container for the extracted data
        cases_list: List[Dict[str, Any]] = []
        
        # Define the specific tags to be extracted
        fields_to_extract: List[str] = [
            "Analysis", "An_Number", "Version", "Examination", "KlinischeVraagstelling",
            "KlinischeGegevens", "Macroscopie", "Microscopie", "Conclusie", "Diagnose"
        ]

        # Parse the XML rows
        for row in root.findall('row'):
            case_details: Dict[str, str] = {}
            for field in fields_to_extract:
                element = row.find(field)
                if element is not None and element.text is not None:
                    case_details[field] = element.text.strip()
                else:
                    case_details[field] = ""
            
            cases_list.append({"case": case_details})

        # Write to the output file based on the selected format
        with open(output_filepath, 'w', encoding='utf-8') as f:
            if output_format == 'json':
                # Write as a standard JSON array
                json.dump(cases_list, f, ensure_ascii=False, indent=4)
            elif output_format == 'jsonl':
                # Write line by line
                for entry in cases_list:
                    f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            
        print(f"Successfully converted '{input_filepath}' to '{output_filepath}' in {output_format.upper()} format.")
        print(f"Processed {len(cases_list)} records.")

    except FileNotFoundError:
        print(f"Error: The file '{input_filepath}' was not found.")
    except ET.ParseError as e:
        print(f"Error parsing XML file '{input_filepath}': {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert an XML file to JSON or JSONL.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path to the input XML file."
    )
    
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to the output file (e.g., data.json or data.jsonl)."
    )

    parser.add_argument(
        "--format",
        type=str,
        choices=['json', 'jsonl'],
        default='json',
        help="The output format: 'json' for a single array, 'jsonl' for newline-delimited JSON. (default: json)"
    )

    args = parser.parse_args()

    convert_xml(args.input_file, args.output_file, args.format)

if __name__ == "__main__":
    main()
