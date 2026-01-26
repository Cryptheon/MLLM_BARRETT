# transfer_labels.py

import json
import argparse

def main():
    parser = argparse.ArgumentParser(
        description="Transfers a label from a source JSON to a target JSON by matching records on a unique key."
    )
    
    parser.add_argument("--source_json", type=str, required=True, help="Path to the JSON file containing the ground truth labels.")
    parser.add_argument("--target_json", type=str, required=True, help="Path to the JSON file that needs the labels.")
    parser.add_argument("--output_json", type=str, required=True, help="Path to save the updated target JSON file.")
    parser.add_argument("--match_key", type=str, default="case_id", help="The key used to match records between files (e.g., 'case_id').")
    parser.add_argument("--label_key", type=str, default="report_extracted_label", help="The key of the label to transfer (e.g., 'report_extracted_label').")

    args = parser.parse_args()
    
    print(f"Loading source labels from '{args.source_json}'...")
    try:
        with open(args.source_json, 'r', encoding='utf-8') as f:
            source_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Source file not found at {args.source_json}")
        return

    # Create a lookup map: {case_id: report_extracted_label} for efficient searching.
    label_map = {}
    for item in source_data:
        case_id = item.get(args.match_key)
        label = item.get(args.label_key)
        if case_id and label is not None:
            label_map[case_id] = label
        else:
            print(f"Warning: Skipping record in source file due to missing '{args.match_key}' or '{args.label_key}'.")

    print(f"Created a lookup map with {len(label_map)} entries.")

    print(f"Loading target data from '{args.target_json}' to update...")
    try:
        with open(args.target_json, 'r', encoding='utf-8') as f:
            target_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Target file not found at {args.target_json}")
        return

    # Iterate through target data and transfer the labels using the map.
    transferred_count = 0
    missing_count = 0
    for item in target_data:
        case_id = item.get(args.match_key)
        if case_id in label_map:
            item[args.label_key] = label_map[case_id]
            transferred_count += 1
        else:
            print(f"Warning: No matching label found for {args.match_key} '{case_id}' in the source file.")
            item[args.label_key] = None  # Explicitly set to null if no match is found
            missing_count += 1
            
    print(f"\nTransferred labels for {transferred_count} records.")
    if missing_count > 0:
        print(f"Could not find a match for {missing_count} records.")

    # Save the updated data.
    with open(args.output_json, 'w', encoding='utf-8') as f:
        json.dump(target_data, f, ensure_ascii=False, indent=4)

    print(f"Successfully saved updated data to {args.output_json}")

if __name__ == "__main__":
    main()