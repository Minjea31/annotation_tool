"""Share serialized library state between browsers and revalidate idle polls."""
from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import threading

from fastapi.responses import Response


def fingerprint(path):
    try:
        stat = path.stat()
        return (stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    except FileNotFoundError:
        return None


class StateCache:
    def __init__(self):
        self.lock = threading.Lock()
        self.entries = OrderedDict()

    def response(self, request, work, registry, build):
        database = work / 'review.sqlite'
        paths = (database, Path(str(database) + '-wal'), registry)
        key = (str(work), str(registry), str(request.url.query))
        with self.lock:
            entry = self.entries.get(key)
            base = tuple(fingerprint(path) for path in paths)
            if entry is not None:
                signature, metrics, content, etag = entry
                current = (base, tuple(fingerprint(path) for path in metrics))
            if entry is None or current != signature:
                data = build()
                metrics = tuple(work / 'runs' / job['id'] / 'metrics.jsonl'
                                for job in data['jobs'])
                # Never tag a snapshot while a writer was changing its sources.
                metric_before = tuple(fingerprint(path) for path in metrics)
                content = json.dumps(data, ensure_ascii=False, separators=(',', ':')).encode()
                etag = '"' + hashlib.sha256(content).hexdigest() + '"'
                after = tuple(fingerprint(path) for path in paths)
                # A running training process can append metrics independently of SQLite.
                # Rebuild on its next poll so no append during build can be cached.
                training = any(job['kind'] == 'train' and job['state'] == 'running'
                               for job in data['jobs'])
                if base != after or training or metric_before != tuple(fingerprint(path) for path in metrics):
                    self.entries.pop(key, None)
                    return Response(content, media_type='application/json',
                                    headers={'Cache-Control': 'no-store'})
                self.entries[key] = ((after, metric_before), metrics, content, etag)
            self.entries.move_to_end(key)
            while len(self.entries) > 16:
                self.entries.popitem(last=False)
            headers = {'ETag': etag, 'Cache-Control': 'no-cache'}
            if request.headers.get('if-none-match') == etag:
                return Response(status_code=304, headers=headers)
            return Response(content, media_type='application/json', headers=headers)
