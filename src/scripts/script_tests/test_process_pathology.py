import pandas as pd
import argparse
import os

def read_test_files():
    """Reads test_report.txt and test_prompt.txt for testing."""
    
    # Read the test pathology report
    with open("test_report.txt", "r") as f:
        test_report = f.read().strip()
    
    # Read the test prompt
    with open("test_prompt.txt", "r") as f:
        test_prompt = f.read().strip()
    
    # Create a test CSV
    test_csv_path = "test_input.csv"
    df = pd.DataFrame({"report_text": [test_report]})
    df.to_csv(test_csv_path, index=False)
    
    print(f"Test files read successfully: test_report.txt, test_prompt.txt")
    return test_csv_path, test_prompt

def run_test():
    """Runs the main script with test input and verifies output."""
    
    test_csv, test_prompt_path = read_test_files()
    output_csv = "test_output.csv"
    
    # Run the main script with test files
    os.system(
        f"python process_pathology_reports.py --config test_config.yaml \
        --prompt_path test_prompt.txt \
        --input_csv {test_csv} \
        --output_csv {output_csv} \
        --column report_text \
        --batch_size 1 \
        --num_variations 1"
    )
    
    # Verify output
    if os.path.exists(output_csv):
        df_out = pd.read_csv(output_csv)
        print("Test output file created successfully!")
        print(df_out.head())
    else:
        print("Test failed: Output CSV was not created.")

if __name__ == "__main__":
    run_test()
