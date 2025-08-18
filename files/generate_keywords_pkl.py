import requests
import pickle
from urllib.parse import quote
from manual_queries import Parts


def calculate_and_save_embeddings(api_url: str, embed_version: str, output_path: str):
    """
    Calculates text embeddings via an API and saves them to a PKL file.

    Args:
        queries (dict): A dictionary where keys are part identifiers (e.g., 'Part1')
                        and values are lists of text queries.
        api_url (str): The base URL of the embedding API endpoint.
        embed_version (str): The version of the embedding model to use.
        output_path (str): The path to save the output PKL file.
    """
    queries = Parts
    embeddings_data = {}
    print("Starting embedding calculation...")

    for part_key, text_list in queries.items():
        part_embeddings = {}
        print(f"Processing {part_key}...")

        for text in text_list:
            # URL encode the text to handle special characters
            encoded_text = quote(text)
            full_url = f"{api_url}/text-embedding?text={encoded_text}&modelVersion={embed_version}"

            try:
                response = requests.get(full_url, timeout=30)  # 30-second timeout
                response.raise_for_status()  # Raises an HTTPError for bad responses (4XX or 5XX)

                # Assuming the API returns a JSON with an 'embedding' key
                embedding = response.json().get('results')['text_embeddings'][0]

                if embedding:
                    if text not in part_embeddings:
                        part_embeddings[text] = []
                    part_embeddings[text].extend(embedding)
                else:
                    print(f"Warning: No embedding found for text: '{text}'")

            except requests.exceptions.RequestException as e:
                print(f"Error fetching embedding for '{text}': {e}")
                # Optionally, you could skip this text or retry
                continue

        embeddings_data[part_key] = part_embeddings
        print(f"Finished processing {part_key}.")

    # Save the dictionary to a PKL file
    with open(output_path, 'wb') as pkl_file:
        pickle.dump(embeddings_data, pkl_file)

    print(f"\nSuccessfully saved all embeddings to {output_path}")


if __name__ == '__main__':
    # 2. Configure your API details
    # IMPORTANT: Replace with your actual API endpoint URL
    #API_ENDPOINT_URL = 'http://text-embedding.dev.pictimenet.pic-time.com:8080'
    API_ENDPOINT_URL =  'http://10.0.28.21:8080/'
    EMBEDDING_MODEL_VERSION = "1"

    # 3. Specify the output file path
    OUTPUT_FILE = "queries_embeddings_v1.pkl"


    # 4. Run the function
    # Note: This will make real API calls. For testing without an API,
    # you would mock the `requests.get` call.
    calculate_and_save_embeddings(API_ENDPOINT_URL, EMBEDDING_MODEL_VERSION, OUTPUT_FILE)

    # --- How to Load and Use the File (as per your example) ---
    def load_and_verify_embeddings(file_path: str):
        """Loads the PKL file and prints its structure."""
        try:
            with open(file_path, 'rb') as pkl_file:
                loaded_data = pickle.load(pkl_file)
                print("\n--- Verification ---")
                print(f"Successfully loaded data from {file_path}")
                print("Data structure:")
                for part, embeddings in loaded_data.items():
                    print(f"  - {part}: Contains {len(embeddings)} unique text(s)")
                    for text, embedding_list in embeddings.items():
                        # Displaying the length of the first embedding as an example
                        if embedding_list:
                            print(
                                f"    - Text: '{text}', Found {len(embedding_list)} embedding(s), Vector length: {len(embedding_list[0])}")
                return loaded_data
        except FileNotFoundError:
            print(f"Error: The file {file_path} was not found.")
        except Exception as e:
            print(f"An error occurred while loading the file: {e}")

    # To run this part, you first need to generate the PKL file.
    # loaded_tags_features = load_and_verify_embeddings(OUTPUT_FILE)

    # Your subsequent code would then use 'loaded_tags_features' as planned.