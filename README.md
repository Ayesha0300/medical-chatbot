# Medical Chatbot

Welcome to the Medical Chatbot project! This chatbot utilizes advanced AI models to assist users with medical-related queries. The backend is powered by LangChain and Hugging Face models, while the frontend is built using Streamlit.

## Prerequisites

Ensure you have `UV` installed before proceeding. If you haven't installed `UV` yet, follow the official installation guide:

- [UV Documentation](https://install.uv/docs/)

## Setting Up Your Environment with UV

### Step 1: Install Required Packages

Run the following commands in your terminal to set up the environment and install dependencies:

```sh
uv pip install langchain langchain_community langchain_huggingface faiss-cpu pypdf
uv pip install huggingface_hub
uv pip install streamlit
```

## Running the Chatbot

Once dependencies are installed, you can start the chatbot with:

```sh
streamlit run app.py
```

This will launch the chatbot interface in your web browser.

## Contributing

Contributions are welcome! Feel free to submit issues or pull requests to improve the chatbot.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

---

For any questions or suggestions, please open an issue on the [GitHub repository](https://github.com/Ayesha0300/medical-chatbot).

