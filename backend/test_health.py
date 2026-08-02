import asyncio
import json
import unittest

from main import app


class HealthEndpointTest(unittest.TestCase):
    def test_health_returns_ok(self):
        messages = []
        request_sent = False

        async def receive():
            nonlocal request_sent
            if not request_sent:
                request_sent = True
                return {"type": "http.request", "body": b"", "more_body": False}
            return {"type": "http.disconnect"}

        async def send(message):
            messages.append(message)

        scope = {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "GET",
            "scheme": "http",
            "path": "/health",
            "raw_path": b"/health",
            "query_string": b"",
            "root_path": "",
            "headers": [],
            "client": ("testclient", 50000),
            "server": ("testserver", 80),
        }

        asyncio.run(app(scope, receive, send))

        response_start = next(
            message for message in messages if message["type"] == "http.response.start"
        )
        response_body = b"".join(
            message.get("body", b"")
            for message in messages
            if message["type"] == "http.response.body"
        )

        self.assertEqual(response_start["status"], 200)
        self.assertEqual(
            json.loads(response_body),
            {"status": "ok", "service": "SignBridge"},
        )


if __name__ == "__main__":
    unittest.main()
