"""Persistent evaluation membership. No training version may reassign eval images."""
import hashlib
import json
from pathlib import Path
import uuid

POLICY = 'fixed_eval_v1'


def read(root):
    path = root / 'registry.json'
    return json.loads(path.read_text()) if path.exists() else None


def merge(root, pending, atomic_copy):
    registry = read(root) or {'policy': POLICY, 'id': uuid.uuid4().hex, 'eval_seed': 31, 'entries': {}}
    entries = registry['entries']
    unseen = {r['origin']: r for r in pending if r['origin'] not in entries}
    # Existing hash aliases inherit membership too, including removed entries.
    hash_groups = registry.setdefault('hash_groups', {r['sha256']: r['group'] for r in entries.values()})
    independent = [key for key, r in unseen.items() if r['sha256'] not in hash_groups]
    independent.sort(key=lambda key: hashlib.sha256(('31:' + key).encode()).hexdigest())
    needed = max(0, round((len(entries) + len(unseen)) * .2) - sum(r['group']=='eval' for r in entries.values()))
    new_eval = set(independent[:needed])
    for r in pending:
        previous = entries.get(r['origin'])
        group = previous['group'] if previous else hash_groups.get(r['sha256'], 'eval' if r['origin'] in new_eval else 'pool')
        # A relabel or a re-encoded frame cannot silently cross groups.
        if r['sha256'] in hash_groups and hash_groups[r['sha256']] != group:
            raise ValueError('Image hash conflicts with an existing eval assignment')
        image = Path(group) / 'images' / (r['id'] + Path(r['image']).suffix)
        mask = Path(group) / 'masks' / (r['id'] + '.png')
        atomic_copy(r['_image'], root / image)
        atomic_copy(r['_mask'], root / mask)
        entries[r['origin']] = {k: v for k, v in r.items() if not k.startswith('_')}
        entries[r['origin']].update(group=group, image=str(image), mask=str(mask))
        hash_groups[r['sha256']] = group
    root.mkdir(parents=True, exist_ok=True)
    temporary = root / ('.registry.' + uuid.uuid4().hex)
    temporary.write_text(json.dumps(registry, indent=2))
    temporary.replace(root / 'registry.json')
    return registry
