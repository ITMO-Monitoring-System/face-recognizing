import json
import unittest

from face_service.core.message_context import request_id_from_message


class MessageContextTest(unittest.TestCase):
    def test_binary_headers(self):
        self.assertEqual(request_id_from_message(b"jpeg", "image/jpeg", {"request_id": "crop-1"}), "crop-1")

    def test_legacy_json(self):
        body = json.dumps({"image_b64": "data", "request_id": "crop-2"}).encode()
        self.assertEqual(request_id_from_message(body, "application/json", None), "crop-2")

    def test_older_message_without_id(self):
        self.assertEqual(request_id_from_message(b"jpeg", "image/jpeg", {"lecture_id": 12}), "")
        self.assertEqual(request_id_from_message(b'{"image_b64":"data"}', "application/json", None), "")


if __name__ == "__main__":
    unittest.main()
