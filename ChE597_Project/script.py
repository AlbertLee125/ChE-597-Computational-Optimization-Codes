import argparse
import json

def convert_to_json(input_file, output_file):
    # Read the contents of the Python file
    with open(input_file, "r") as file:
        python_code = file.read()

    # Create a dictionary with the "code" field and the Python code as the value
    data = {"code": python_code}

    # Write the dictionary to the JSON file
    with open(output_file, "w") as file:
        json.dump(data, file, indent=4)

    print("JSON file created successfully.")

def main():
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description="Convert a Python file to JSON")

    # Add command line arguments
    parser.add_argument("input_file", help="Path to the input Python file")
    parser.add_argument("output_file", help="Path to the output JSON file")

    # Parse the command line arguments
    args = parser.parse_args()

    # Call the convert_to_json function with the provided arguments
    convert_to_json(args.input_file, args.output_file)

if __name__ == "__main__":
    main()