def flatten(data, parent_key='', sep='.'):
    items = []
    if isinstance(data, dict):
        for k, v in data.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            items.extend(flatten(v, new_key, sep=sep).items())
    elif isinstance(data, list):
        for i, v in enumerate(data):
            new_key = f"{parent_key}{sep}{i}" if parent_key else str(i)
            items.extend(flatten(v, new_key, sep=sep).items())
    else:
        items.append((parent_key, data))
    return dict(items)

def snake_to_camel(s: str) -> str:
    parts = s.split('_')
    return parts[0] + "".join([word.title() for word in parts])
