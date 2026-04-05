import re

with open('eyeshield_training_preprocessor_deepdrid.py', 'r', encoding='utf-8') as f:
    content = f.read()

# We need to add '/' to candidate roots
old_str = "    candidate_roots.extend([\n        '/kaggle/input/deepdrid',\n        '/kaggle/input/deepdrid/regular_fundus_images',\n        '/kaggle/input/nancyhisham/deepdrid',\n        '/content/dataset',\n    ])"
new_str = "    candidate_roots.extend([\n        '/kaggle/input/deepdrid',\n        '/kaggle/input/deepdrid/regular_fundus_images',\n        '/kaggle/input/nancyhisham/deepdrid',\n        '/content/dataset',\n        '/',\n    ])"

content = content.replace(old_str, new_str)

with open('eyeshield_training_preprocessor_deepdrid.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("Added '/' to candidate roots.")
