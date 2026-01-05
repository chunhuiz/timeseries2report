# db_utils.py
import sqlite3
import threading
import json
from contextlib import contextmanager
from typing import List, Dict, Optional, Any
from datetime import datetime
import os


# Thread-local storage for database connections
_thread_local = threading.local()


def get_db_path():
    """Get the database file path."""
    current_path = os.getcwd() + "/app.db"  # Set the database file to 'app.db' in the current working directory
    return current_path


def get_connection():
    """Get a thread-local database connection."""
    if not hasattr(_thread_local, "connection"):  # Check if a connection already exists for this thread
        _thread_local.connection = sqlite3.connect(
            get_db_path(),
            check_same_thread=False  # Allow connections to be shared across threads
        )
        _thread_local.connection.row_factory = sqlite3.Row  # Set row factory to return rows as dictionaries
    return _thread_local.connection  # Return the thread-local connection


@contextmanager
def get_db():
    """Database context manager for handling transaction."""
    conn = get_connection()  # Get the database connection
    try:
        yield conn  # Yield the connection for use
        conn.commit()  # Commit transaction if successful
    except Exception as e:
        conn.rollback()  # Roll back on exception
        raise e  # Re-raise the exception


def init_db():
    """Initialize database tables."""
    with get_db() as conn:
        cursor = conn.cursor()

        # Create sessions table if it does not exist
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_name TEXT UNIQUE NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Create messages table if it does not exist
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES sessions (id) ON DELETE CASCADE
            )
        ''')

        # Create analysis results table if it does not exist
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS analysis_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                result_data TEXT NOT NULL,
                markdown_table TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES sessions (id) ON DELETE CASCADE
            )
        ''')

        # Create index for improving query performance on messages
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_messages_session_id
            ON messages(session_id)
        ''')

        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_analysis_results_session_id
            ON analysis_results(session_id)
        ''')

        conn.commit()


# ============= Session Management =============

def create_session(session_name: str) -> int:
    """Create a new session."""
    # First, check if the session already exists
    existing_session = get_session_by_name(session_name)
    if existing_session:
        # If it exists, return the existing session's ID
        return existing_session['id']

    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO sessions (session_name) VALUES (?)",
            (session_name,)
        )
        return cursor.lastrowid


def get_all_sessions() -> List[Dict[str, Any]]:
    """Retrieve all sessions."""
    with get_db() as conn:  # Use the database context manager
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, session_name, created_at, updated_at FROM sessions ORDER BY updated_at DESC"
        )
        rows = cursor.fetchall()  # Fetch all rows from the query
        return [dict(row) for row in rows]  # Convert rows to dictionary format


def get_session_by_name(session_name: str) -> Optional[Dict[str, Any]]:
    """Retrieve a session by its name."""
    with get_db() as conn:  # Use the database context manager
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, session_name, created_at, updated_at FROM sessions WHERE session_name = ?",
            (session_name,)  # Use parameter substitution to prevent SQL injection
        )
        row = cursor.fetchone()  # Fetch one row from the query
        return dict(row) if row else None  # Return as dictionary if found, else None


def update_session_name(old_name: str, new_name: str) -> bool:
    """Update a session's name."""
    with get_db() as conn:  # Use the database context manager
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE sessions SET session_name = ?, updated_at = CURRENT_TIMESTAMP WHERE session_name = ?",
            (new_name, old_name)  # Update with new name and set updated timestamp
        )
        return cursor.rowcount > 0  # Return True if a row was updated


def delete_session(session_name: str) -> bool:
    """Delete a session and all associated messages."""
    with get_db() as conn:  # Use the database context manager
        cursor = conn.cursor()
        cursor.execute(
            "DELETE FROM sessions WHERE session_name = ?",
            (session_name, )  # Delete session by name
        )
        return cursor.rowcount > 0  # Return True if a row was deleted


def update_session_timestamp(session_name: str):
    """Update the last updated timestamp of a session."""
    with get_db() as conn:  # Use the database context manager
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE sessions SET updated_at = CURRENT_TIMESTAMP WHERE session_name = ?",
            (session_name, )  # Update timestamp for this session
        )


# ============= Message Management =============

def add_message(session_name: str, role: str, content: str) -> int:
    """Add a message to a session."""
    # Ensure the session exists
    session = get_session_by_name(session_name)
    if not session:
        session_id = create_session(session_name)  # Create a new session if it doesn't exist
    else:
        session_id = session['id']  # Get the existing session ID

    with get_db() as conn:  # Use the database context manager
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)",
            (session_id, role, content)  # Insert new message
        )
        # Update the session timestamp after adding a message
        update_session_timestamp(session_name)
        return cursor.lastrowid  # Return the last inserted message ID


def get_messages(session_name: str) -> List[Dict[str, str]]:
    """Retrieve all messages from a session."""
    session = get_session_by_name(session_name)  # Get session by name
    if not session:
        return []  # Return empty list if session does not exist

    with get_db() as conn:  # Use the database context manager
        cursor = conn.cursor()
        cursor.execute(
            "SELECT role, content, created_at FROM messages WHERE session_id = ? ORDER BY created_at ASC",
            (session['id'],)  # Get messages for this session in chronological order
        )
        rows = cursor.fetchall()  # Fetch all messages
        return [{"role": row['role'], "content": row['content']} for row in rows]  # Return messages as a list of dictionaries


def clear_messages(session_name: str) -> bool:
    """Clear all messages from a session."""
    session = get_session_by_name(session_name)  # Get session by name
    if not session:
        return False  # Return False if session does not exist

    with get_db() as conn:  # Use the database context manager
        cursor = conn.cursor()
        cursor.execute(
            "DELETE FROM messages WHERE session_id = ?",
            (session['id'],)  # Delete messages by session ID
        )
        return cursor.rowcount > 0  # Return True if messages were deleted

def get_message_count(session_name: str) -> int:
    """Get the count of messages in a session."""
    session = get_session_by_name(session_name)  # Get session by name
    if not session:
        return 0  # Return 0 if session does not exist

    with get_db() as conn:  # Use the database context manager
        cursor = conn.cursor()
        cursor.execute(
            "SELECT COUNT(*) as count FROM messages WHERE session_id = ?",
            (session['id'],)  # Count messages for this session
        )
        row = cursor.fetchone()  # Fetch the count
        return row['count'] if row else 0  # Return the count

# ============= Session Data Synchronization =============

def sync_sessions_from_memory(sessions_dict: Dict[str, List[Dict[str, str]]]):
    """Synchronize sessions from memory to the database."""
    with get_db() as conn:  # Use the database context manager
        for session_name, messages in sessions_dict.items():
            # Ensure the session exists
            session = get_session_by_name(session_name)
            if not session:
                create_session(session_name)  # Create session if necessary

            # Clear existing messages
            clear_messages(session_name)

            # Add all messages to the session
            for msg in messages:
                add_message(session_name, msg['role'], msg['content'])  # Add each message

def load_sessions_to_memory() -> Dict[str, List[Dict[str, str]]]:
    """Load all sessions from the database into memory."""
    sessions = get_all_sessions()  # Get all sessions
    sessions_dict = {}  # Dictionary to hold session data

    for session in sessions:
        session_name = session['session_name']  # Extract session name
        messages = get_messages(session_name)  # Get messages for this session
        sessions_dict[session_name] = messages  # Map session name to messages

    # Create a default session if no sessions exist
    if not sessions_dict:
        sessions_dict["Session 1"] = []  # Add default session

    return sessions_dict  # Return all sessions as a dictionary

# ============= Analysis Result Management =============

def save_analysis_result(session_name: str, result_data: dict, markdown_table: str) -> int:
    """Save analysis results to the database."""
    session = get_session_by_name(session_name)  # Get session by name
    if not session:
        session_id = create_session(session_name)  # Create session if it does not exist
    else:
        session_id = session['id']  # Get existing session ID

    # First, delete any old analysis results for this session
    with get_db() as conn:  # Use the database context manager
        cursor = conn.cursor()
        cursor.execute(
            "DELETE FROM analysis_results WHERE session_id = ?",
            (session_id,)  # Delete old results by session ID
        )

        # Save the new analysis results
        result_json = json.dumps(result_data, ensure_ascii=False)  # Convert result data to JSON format
        cursor.execute(
            "INSERT INTO analysis_results (session_id, result_data, markdown_table) VALUES (?, ?, ?)",
            (session_id, result_json, markdown_table)  # Insert new analysis results
        )
        # Update the session timestamp after saving analysis results
        update_session_timestamp(session_name)
        return cursor.lastrowid  # Return the last inserted analysis result ID


def get_analysis_result(session_name: str) -> Optional[Dict[str, Any]]:
    """Retrieve the latest analysis result for a session."""
    session = get_session_by_name(session_name)  # Get session by name
    if not session:
        return None  # Return None if session does not exist

    with get_db() as conn:  # Use the database context manager
        cursor = conn.cursor()
        cursor.execute(
            "SELECT result_data, markdown_table, created_at FROM analysis_results WHERE session_id = ? ORDER BY created_at DESC LIMIT 1",
            (session['id'],)  # Get the latest analysis result for this session
        )
        row = cursor.fetchone()  # Fetch the result
        if row:
            return {
                'result_data': json.loads(row['result_data']),  # Return result data as dictionary
                'markdown_table': row['markdown_table'],  # Return markdown table
                'created_at': row['created_at']  # Return creation timestamp
            }
        return None  # Return None if no results found


def delete_analysis_result(session_name: str) -> bool:
    """Delete the analysis result for a session."""
    session = get_session_by_name(session_name)  # Get session by name
    if not session:
        return False  # Return False if session does not exist

    with get_db() as conn:  # Use the database context manager
        cursor = conn.cursor()
        cursor.execute(
            "DELETE FROM analysis_results WHERE session_id = ?",
            (session['id'],)  # Delete analysis results by session ID
        )
        return cursor.rowcount > 0  # Return True if analysis results were deleted
