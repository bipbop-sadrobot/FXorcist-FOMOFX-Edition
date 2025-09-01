import sqlite3, json, pathlib, time
from typing import Optional
from memory_system.core import MemoryEntry
DB_PATH = pathlib.Path(__file__).parent.parent.parent / "memory_store_final.db"
class SQLiteStore:
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or str(DB_PATH)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._init()
    def _init(self):
        c = self._conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS entries (
                id TEXT PRIMARY KEY,
                text TEXT,
                vector BLOB,
                metadata TEXT,
                tier TEXT,
                ts REAL
            )
        ''')
        self._conn.commit()
    def save_entry(self, entry: MemoryEntry):
        c = self._conn.cursor()
        vec = None
        if entry.vector is not None:
            vec = json.dumps(entry.vector)
        meta = json.dumps(entry.metadata)
        c.execute('REPLACE INTO entries (id, text, vector, metadata, tier, ts) VALUES (?, ?, ?, ?, ?, ?)',
                  (entry.id, entry.text, vec, meta, entry.tier, entry.ts))
        self._conn.commit()
    def delete_entry(self, id: str):
        c = self._conn.cursor()
        c.execute('DELETE FROM entries WHERE id=?', (id,))
        self._conn.commit()
    def get_entry(self, id: str):
        c = self._conn.cursor()
        c.execute('SELECT id, text, vector, metadata, tier, ts FROM entries WHERE id=?', (id,))
        r = c.fetchone()
        if not r:
            return None
        vec = json.loads(r[2]) if r[2] else None
        meta = json.loads(r[3]) if r[3] else {}
        return MemoryEntry(id=r[0], text=r[1], vector=vec, metadata=meta, tier=r[4], ts=r[5])

    def close(self):
        self._conn.close()
