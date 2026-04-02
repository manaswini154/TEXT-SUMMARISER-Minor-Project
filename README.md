Text Summariser

This project is a full-stack text summarisation web application that generates concise summaries from long passages or dialogues using a transformer-based deep learning model. It is built with FastAPI for the backend and a simple HTML/CSS/JavaScript interface for user interaction.

The application uses a fine-tuned T5 (Text-to-Text Transfer Transformer) model from Hugging Face to perform abstractive summarisation. User input is first cleaned and tokenized, then passed through the model to generate a meaningful summary using beam search decoding. The result is returned via an API and dynamically displayed on the web interface.

The system also supports automatic device selection (CPU/GPU), making it efficient and scalable for different environments.

🔧 Tech Stack
FastAPI (Backend)
HTML, CSS, JavaScript (Frontend)
Hugging Face Transformers (T5)
PyTorch
