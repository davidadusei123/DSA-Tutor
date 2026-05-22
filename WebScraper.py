from pathlib import Path
from urllib.parse import urljoin, urlparse, unquote
import shutil
import requests
from bs4 import BeautifulSoup

index_file = Path(r"C:\Users\ik_ad\SLM-Tutor\Data\6.006-spring-2020\pages\lecture-notes\index.html")
output_dir = Path(r"C:\Users\ik_ad\Downloads\6006_lecture_notes")
output_dir.mkdir(parents=True, exist_ok=True)

base_url = index_file.as_uri()

with open(index_file, "r", encoding="utf-8") as f:
    soup = BeautifulSoup(f, "html.parser")

links = []
for a in soup.find_all("a", href=True):
    href = a["href"]
    if href.lower().endswith((".pdf", ".html", ".txt", ".docx", ".pptx")):
        links.append(urljoin(base_url, href))

print(f"Found {len(links)} lecture note files.")

for link in links:
    parsed = urlparse(link)
    filename = Path(unquote(parsed.path)).name
    dest = output_dir / filename

    try:
        if parsed.scheme == "file":
            src = Path(unquote(parsed.path.lstrip("/")))
            shutil.copy2(src, dest)
        else:
            r = requests.get(link, timeout=20)
            r.raise_for_status()
            dest.write_bytes(r.content)

        print(f"Saved: {dest}")

    except Exception as e:
        print(f"Failed: {link}")
        print(f"Reason: {e}")