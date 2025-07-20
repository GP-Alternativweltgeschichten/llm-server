# LLM server for the alternate history project

## Requirements
- Python 3.8 or higher
- PIP

## Setup
Run the following command to install the required packages:
```bash
pip install -r requirements.txt
```
Copy the `.env.example` file to `.env` and fill in the required environment variables:
```bash
cp .env.example .env
```

Generate an access token on Huggingface and add the [Mistral 7B](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2) model.

## Usage
Run the server using the following command:
```bash
python app.py
```