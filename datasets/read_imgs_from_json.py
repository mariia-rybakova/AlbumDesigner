import json

try:
    with open("json_files/57448844.json", "r") as f:
        data = json.load(f)
        print(data)
        print("File is in valid JSON format.")
except json.JSONDecodeError as e:
    print("Error: File is not in valid JSON format:", e)
