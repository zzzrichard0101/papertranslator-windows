"""Preserve dependency license files/metadata alongside the source release."""
from importlib.metadata import distributions
import json
from pathlib import Path
import shutil

project=Path(__file__).resolve().parents[1]
folder=project/'exports'/'dependency-notices'; folder.mkdir(parents=True,exist_ok=True)
entries=[]
for distribution in distributions():
    name=distribution.metadata.get('Name','unknown')
    entry=dict(name=name,version=distribution.version,license=distribution.metadata.get('License'),
               expression=distribution.metadata.get('License-Expression'),home=distribution.metadata.get('Home-page'))
    for file in distribution.files or []:
        if any(word in str(file).lower() for word in ['license','copying','notice']) and '.dist-info' in str(file):
            source=Path(distribution.locate_file(file))
            if source.is_file():
                # Keep nested license paths; identical basenames must not overwrite.
                index=next(i for i,part in enumerate(file.parts) if part.endswith('.dist-info'))
                dest=folder/name/Path(*file.parts[index+1:])
                dest.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(source,dest)
    entries.append(entry)
(folder/'manifest.json').write_text(json.dumps(entries,ensure_ascii=False,indent=2),encoding='utf-8')
print(f'Preserved metadata for {len(entries)} distributions')
