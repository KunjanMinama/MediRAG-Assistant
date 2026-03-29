import requests
from config import API_URL


def upload_pdfs_api(files):
    files_payload = [
        ("files", (file.name, file.read(), "application/pdf"))
        for file in files
    ]
    return requests.post(f"{API_URL}/upload_pdfs/", files=files_payload)


def ask_question(question: str):
    return requests.post(f"{API_URL}/ask/", data={"question": question})