import xml.etree.ElementTree as ET
import json
import argparse
from typing import List, Dict, Any

def convert_xml_to_json(input_filepath: str, output_filepath: str) -> None:
    """
    Reads an XML file, parses it to extract specified fields from each <row> 
    element, and saves the structured data as a JSON file.

    Args:
        input_filepath (str): The path to the input XML file.
        output_filepath (str): The path to the output JSON file.
    """
    try:
        # Read the content from the input XML file
        with open(input_filepath, 'r', encoding='utf-8') as file:
            xml_data = file.read()

        root = ET.fromstring(xml_data)
        
        # Using type hints for the list of dictionaries
        cases_list: List[Dict[str, Any]] = []
        
        # Define the specific tags to be extracted from each <row>
        fields_to_extract: List[str] = [
            "Analysis", "An_Number", "Version", "Examination", "KlinischeVraagstelling",
            "KlinischeGegevens", "Macroscopie", "Microscopie", "Conclusie", "Diagnose"
        ]

        for row in root.findall('row'):
            case_details: Dict[str, str] = {}
            for field in fields_to_extract:
                element = row.find(field)
                # Check if the element exists and has text
                if element is not None and element.text is not None:
                    # Use strip() to remove leading/trailing whitespace and newlines
                    case_details[field] = element.text.strip()
                else:
                    # Assign an empty string if the tag is empty or not present
                    case_details[field] = ""
            
            cases_list.append({"case": case_details})

        # Write the structured data to a JSON file with pretty printing
        with open(output_filepath, 'w', encoding='utf-8') as json_file:
            json.dump(cases_list, json_file, ensure_ascii=False, indent=4)
            
        print(f"Successfully converted '{input_filepath}' to '{output_filepath}'")

    except FileNotFoundError:
        print(f"Error: The file '{input_filepath}' was not found.")
    except ET.ParseError as e:
        print(f"Error parsing XML file '{input_filepath}': {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def main() -> None:
    """
    Main function to parse command-line arguments and run the conversion.
    """
    parser = argparse.ArgumentParser(
        description="Convert a specific XML format to a JSON file.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument(
        "--input_file",
        type=str,
        help="Path to the input XML file."
    )
    
    parser.add_argument(
        "--output_file",
        type=str,
        nargs='?', # Makes the argument optional
        default="output.json",
        help="Path to the output JSON file. (default: output.json)"
    )

    args = parser.parse_args()

    convert_xml_to_json(args.input_file, args.output_file)

if __name__ == "__main__":
    main()
