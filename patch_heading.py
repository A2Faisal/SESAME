import os
import re

html_path = "docs/sesame/sesametoolbox.html"
if os.path.exists(html_path):
    with open(html_path, "r") as f:
        html = f.read()
    # Replace the <title>
    html = re.sub(r'<title>.*?API documentation</title>', '<title>SESAME Toolbox</title>', html)
    # Replace the main heading (h1 with class "modulename")
    html = re.sub(r'<h1 class="modulename">.*?</h1>', '<h1 class="modulename">SESAME</h1>', html, flags=re.DOTALL)
    # Replace all occurrences of "API Documentation" (case-insensitive) with bold "Toolbox"
    html = re.sub(r'API Documentation', '<b>Toolbox</b>', html)
    html = re.sub(r'API documentation', '<b>Toolbox</b>', html)
    # Remove () from function names in the sidebar and anywhere else
    html = re.sub(r'([a-zA-Z_][a-zA-Z0-9_]*)\(\)', r'\1', html)
    with open(html_path, "w") as f:
        f.write(html)
    print("Patched main heading, title, sidebar label, and removed () from function names in", html_path)
else:
    print("File not found:", html_path)
    