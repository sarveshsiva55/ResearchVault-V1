import sqlite3
import os
from config.settings import DB_PATH

class DatabaseManager:
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._initialize_tables()

    def _initialize_tables(self):
        cursor = self.conn.cursor()
        
        # Papers table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS papers (
            paper_id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_hash TEXT UNIQUE,
            title TEXT,
            authors TEXT,
            year INTEGER,
            path TEXT,
            page_count INTEGER
        )
        ''')

        # Sections table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS sections (
            section_id INTEGER PRIMARY KEY AUTOINCREMENT,
            paper_id INTEGER,
            title TEXT,
            level INTEGER,
            page_start INTEGER,
            page_end INTEGER,
            FOREIGN KEY(paper_id) REFERENCES papers(paper_id)
        )
        ''')

        # Chunks table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS chunks (
            chunk_id INTEGER PRIMARY KEY AUTOINCREMENT,
            paper_id INTEGER,
            section_id INTEGER,
            page INTEGER,
            text TEXT,
            hash TEXT UNIQUE,
            vector_id INTEGER,
            FOREIGN KEY(paper_id) REFERENCES papers(paper_id),
            FOREIGN KEY(section_id) REFERENCES sections(section_id)
        )
        ''')

        # Questions, Answers, Feedback (from section 15.1)
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS questions (
            question_id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT,
            intent TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        ''')

        self.conn.commit()

    def close(self):
        self.conn.close()

if __name__ == "__main__":
    db = DatabaseManager()
    print(f"Database initialized at {db.db_path}")
