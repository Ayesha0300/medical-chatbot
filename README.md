# MediAI Assistant

## Overview
MediAI Assistant is an intelligent medical knowledge assistant that helps manage and query medical documentation.

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/medical-chatbot.git
    cd medical-chatbot
    ```

2. Create and activate a virtual environment using `uv`:
    ```sh
    uv venv
    ```

3. Synchronize dependencies:
    ```sh
    uv sync
    ```

## Usage

1. Run the Chainlit server:
    ```sh
    chainlit run medibot.py
    ```

2. Open your web browser and go to `http://localhost:8000` to access the MediAI Assistant.

## Features

- **Document Management**: Upload and process medical documents.
- **Chat Assistant**: Query the knowledge base using a chatbot interface.
- **Knowledge Base**: Analyze and inspect the vector store.
- **About**: Information about the MediAI Assistant.

## Configuration

Ensure you have a `.env` file with the necessary environment variables, such as `HF_TOKEN`.

## License

This project is licensed under the MIT License.

For any questions or suggestions, please open an issue on the [GitHub repository](https://github.com/Ayesha0300/medical-chatbot).

