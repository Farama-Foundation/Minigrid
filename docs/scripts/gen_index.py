__author__ = "Feng Gu"
__email__ = "contact@fenggu.me"

"""
   isort:skip_file
"""

import os
import re

readme_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "README.md",
)

output_path = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "index.md",
)

sections = []
all_text = f"""---
hide-toc: true
firstpage:
lastpage:
---\n"""

index_toctree = """
```{toctree}
:hidden:
:caption: Introduction

content/installation
content/basic_usage
api/wrappers
content/pubs
```


```{toctree}
:hidden:
:caption: Environments

environments/design
environments/index
```

```{toctree}
:hidden:
:caption: Development

Github <https://github.com/Farama-Foundation/MiniGrid>
Donate <https://farama.org/donations>
Contribute to the Docs <https://github.com/Farama-Foundation/MiniGrid/blob/master/.github/PULL_REQUEST_TEMPLATE.md>
```

"""
# gen index.md
with open(readme_path, "r") as f:
    readme = f.read()

    '''
    sections = [description, publications, installation, basic usage, wrappers, design, included envrionments&etc]
    '''
    sections = readme.split("<br>")
    all_text += sections[0]
    all_text += sections[2]
all_text += index_toctree

with open(output_path, "w") as f:
    f.write(all_text)


'''
1. gen index.md
2. gen /environments/index.md
3. gen /environments/design.md
'''