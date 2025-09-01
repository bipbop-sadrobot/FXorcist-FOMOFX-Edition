import json
import os
from pathlib import Path
from datetime import datetime
import shutil

# === Configuration ===
DATA_DIR = Path("chat_data")
HISTORY_FILE = DATA_DIR / "chat_history.json"
SESSION_DIR = DATA_DIR / "sessions"
LOG_FILE = DATA_DIR / "chat.log"

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
SESSION_DIR.mkdir(parents=True, exist_ok=True)


# === Utility Functions ===
def safe_write_json(path: Path, data: dict) -> None:
    """Safely write JSON to file (atomic write with temp replacement)."""
    temp_file = path.with_suffix(".tmp")
    try:
        with temp_file.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        temp_file.replace(path)
    except Exception as e:
        print(f"[ERROR] Failed to write {path}: {e}")


def safe_read_json(path: Path, default: dict) -> dict:
    """Safely read JSON file, return default if corrupted/missing."""
    if not path.exists():
        return default
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[WARNING] Failed to read {path}: {e}, resetting file.")
        return default


def log_message(msg: str) -> None:
    """Append a message to the log file with timestamp."""
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(f"{datetime.now():%Y-%m-%d %H:%M:%S} - {msg}\n")


# === Core Chat Handling ===
class ChatManager:
    def __init__(self):
        self.history = safe_read_json(HISTORY_FILE, {"messages": []})

    def add_message(self, role: str, content: str) -> None:
        """Add a message to chat history."""
        self.history["messages"].append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        safe_write_json(HISTORY_FILE, self.history)
        log_message(f"{role.upper()}: {content[:50]}{'...' if len(content) > 50 else ''}")

    def get_history(self, limit: int = 20):
        """Return recent messages (default last 20)."""
        return self.history["messages"][-limit:]

    def save_session(self) -> Path:
        """Save current chat history as a session file (rotated)."""
        session_file = SESSION_DIR / f"session_{datetime.now():%Y%m%d_%H%M%S}.json"
        safe_write_json(session_file, self.history)
        log_message(f"Session saved → {session_file.name}")
        return session_file

    def clear_history(self) -> None:
        """Clear current chat history."""
        self.history = {"messages": []}
        safe_write_json(HISTORY_FILE, self.history)
        log_message("History cleared.")


# === Example CLI Usage ===
if __name__ == "__main__":
    chat = ChatManager()

    print("🤖 Chat Manager Ready (type 'exit' to quit, 'save' to save session, 'clear' to reset)\n")

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() == "exit":
            print("👋 Goodbye!")
            break
        elif user_input.lower() == "save":
            session_path = chat.save_session()
            print(f"✅ Session saved as {session_path.name}")
        elif user_input.lower() == "clear":
            chat.clear_history()
            print("🧹 Chat history cleared.")
        else:
            chat.add_message("user", user_input)
            # Placeholder for AI/assistant response
            response = f"(Echo) {user_input}"
            chat.add_message("assistant", response)
            print(f"Bot: {response}")
