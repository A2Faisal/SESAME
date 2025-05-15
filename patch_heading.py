# patch_heading.py
import os
import re

html_path = "docs/sesame/sesametoolbox.html"
if os.path.exists(html_path):
    with open(html_path, "r") as f:
        html = f.read()
    # Replace the <title>
    html = re.sub(r'<title>.*?API documentation</title>', '<title>SESAME API documentation</title>', html)
    # Replace the main heading (h1 with class "modulename")
    html = re.sub(r'<h1 class="modulename">.*?</h1>', '<h1 class="modulename">SESAME</h1>', html, flags=re.DOTALL)
    with open(html_path, "w") as f:
        f.write(html)
    print("Patched main heading and title in", html_path)
else:
    print("File not found:", html_path)