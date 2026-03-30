import tempfile
import unittest
from pathlib import Path

from app.api import routes
from app.core import upload_requests


class UploadRequestsTest(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.base = Path(self.tempdir.name)
        self.original_requests_path = upload_requests.UPLOAD_REQUESTS_PATH
        self.original_pending_dir = upload_requests.PENDING_UPLOADS_DIR
        upload_requests.UPLOAD_REQUESTS_PATH = self.base / "upload_requests.json"
        upload_requests.PENDING_UPLOADS_DIR = self.base / "pending"

    def tearDown(self):
        upload_requests.UPLOAD_REQUESTS_PATH = self.original_requests_path
        upload_requests.PENDING_UPLOADS_DIR = self.original_pending_dir
        self.tempdir.cleanup()

    def test_create_and_update_upload_request(self):
        source_path = upload_requests.PENDING_UPLOADS_DIR / "sample.pdf"
        source_path.parent.mkdir(parents=True, exist_ok=True)
        source_path.write_bytes(b"pdf")

        created = upload_requests.create_upload_request(
            filename="sample.pdf",
            source_path=source_path,
            uploader_id="rd",
            department="RD",
            visibility="private",
            visible_departments=["RD", "MD", "RD"],
        )

        self.assertTrue(created["request_id"].startswith("req_"))
        self.assertEqual(created["status"], "pending")
        self.assertEqual(created["visible_departments"], ["RD", "MD"])

        items = upload_requests.list_upload_requests(uploader_id="rd")
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["source"], "sample.pdf")

        updated = upload_requests.update_upload_request(
            created["request_id"],
            {"status": "approved", "approved_by": "admin", "approved_at": "2026-03-30T00:00:00+00:00"},
        )
        self.assertIsNotNone(updated)
        self.assertEqual(updated["status"], "approved")
        self.assertEqual(updated["approved_by"], "admin")

    def test_build_request_payload_reflects_status(self):
        source_path = self.base / "approved.pdf"
        source_path.write_bytes(b"ok")

        payload = routes._build_request_payload(
            {
                "request_id": "req_123",
                "source": "approved.pdf",
                "source_path": str(source_path),
                "uploader_id": "rd",
                "department": "RD",
                "visibility": "private",
                "visible_departments": ["RD", "MD"],
                "access_level": "선택 부서 공개",
                "status": "approved",
                "approved_by": "admin",
                "approved_at": "2026-03-30T00:00:00+00:00",
                "created_at": "2026-03-30T00:00:00+00:00",
                "updated_at": "2026-03-30T00:00:00+00:00",
                "uploaded_at": "2026-03-30T00:00:00+00:00",
            }
        )

        self.assertEqual(payload["status_category"], "approved")
        self.assertEqual(payload["status"], "승인 완료")
        self.assertEqual(payload["visible_departments"], ["RD", "MD"])


if __name__ == "__main__":
    unittest.main()
