import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.core import ui_state_store


class UIStateStoreTests(unittest.TestCase):
    def test_save_and_load_chats_from_sqlite(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            db_path = tmp_path / "ui_state.db"
            data_dir = tmp_path / "data"

            with patch.object(ui_state_store, "UI_STATE_DB_PATH", db_path), patch.object(ui_state_store, "DATA_DIR", data_dir):
                chats = {
                    "chat_1": {
                        "id": "chat_1",
                        "title": "테스트",
                        "messages": [{"role": "user", "content": "안녕"}],
                    }
                }

                ui_state_store.save_chats(chats, user_id="rd")
                loaded = ui_state_store.get_chats(user_id="rd")

                self.assertEqual(loaded, chats)
                self.assertTrue(db_path.exists())

    def test_legacy_json_is_migrated_to_sqlite(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            db_path = tmp_path / "ui_state.db"
            data_dir = tmp_path / "data"
            legacy_dir = data_dir / "ui_state"
            legacy_dir.mkdir(parents=True, exist_ok=True)
            legacy_file = legacy_dir / "chats_rd.json"

            legacy_payload = {
                "chats": {
                    "chat_legacy": {
                        "id": "chat_legacy",
                        "title": "레거시",
                        "messages": [{"role": "assistant", "content": "기존 응답"}],
                    }
                }
            }
            legacy_file.write_text(json.dumps(legacy_payload, ensure_ascii=False), encoding="utf-8")

            with patch.object(ui_state_store, "UI_STATE_DB_PATH", db_path), patch.object(ui_state_store, "DATA_DIR", data_dir):
                loaded = ui_state_store.get_chats(user_id="rd")

                self.assertEqual(loaded, legacy_payload["chats"])
                self.assertFalse(legacy_file.exists())
                self.assertEqual(ui_state_store.get_chats(user_id="rd"), legacy_payload["chats"])


if __name__ == "__main__":
    unittest.main()
