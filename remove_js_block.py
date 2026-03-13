import re
import os

file_path = (
    r"D:\Python\src\otsumonogatari\templates\document.html"  # Use raw string for path
)

try:
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Regex to find script blocks containing 'adsbygoogle' or 'adSense'
    # Optimized regex to prevent excessive backtracking
    pattern = r"<script\b[^>]*>(?:(?!\<\/script\>).)*?(?:adsbygoogle|adSense)(?:(?!\<\/script\>).)*?<\/script>"

    # Use re.sub to remove all matching script blocks
    modified_content, num_replacements = re.subn(
        pattern, "", content, flags=re.DOTALL | re.IGNORECASE
    )

    if num_replacements > 0:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(modified_content)
        print(
            f"Successfully removed {num_replacements} JavaScript ad blocks from {file_path}"
        )
    else:
        print(f"No JavaScript ad blocks found or removed in {file_path}")

except FileNotFoundError:
    print(f"Error: File not found at {file_path}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
