"""Client for remote GPU execution server."""

import json
import logging
import time
from typing import Any, Dict, Optional, Union
import requests
from requests.exceptions import RequestException, Timeout

logger = logging.getLogger(__name__)


class RemoteGPUClient:
    """Client to interact with Video Brain GPU server."""

    def __init__(
        self,
        base_url: str,
        timeout: int = 30,
        retries: int = 3,
        retry_delay: float = 1.0,
        api_key: Optional[str] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.retries = retries
        self.retry_delay = retry_delay
        self.api_key = api_key
        self.session = requests.Session()
        if api_key:
            self.session.headers.update({"Authorization": f"Bearer {api_key}"})

    def _request(self, method: str, endpoint: str, data: Optional[Dict] = None) -> Dict[str, Any]:
        """Make a request with retries."""
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        last_exception = None
        for attempt in range(self.retries):
            try:
                if method.upper() == "GET":
                    resp = self.session.get(url, timeout=self.timeout)
                else:
                    resp = self.session.post(url, json=data, timeout=self.timeout)
                resp.raise_for_status()
                return resp.json()
            except (RequestException, Timeout) as e:
                last_exception = e
                logger.warning(f"Request failed (attempt {attempt+1}/{self.retries}): {e}")
                if attempt < self.retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
        raise last_exception

    def health(self) -> Dict[str, Any]:
        """Check server health."""
        return self._request("GET", "health")

    def analyze(self, module: str, **kwargs) -> Dict[str, Any]:
        """
        Send analysis request to a specific module endpoint.

        Args:
            module: One of 'scene', 'faces', 'identity', 'conversation', 'ocr', 'objects', 'audio'.
            **kwargs: Additional parameters to pass in the request.

        Returns:
            Response from server.
        """
        endpoint = module
        payload = {"module": module, **kwargs}
        return self._request("POST", endpoint, payload)

    def scene(self, video_path: str, output_path: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """Run scene detection."""
        return self.analyze("scene", video_path=video_path, output_path=output_path, **kwargs)

    def faces(self, video_path: str, scenes_json: str, output_path: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """Run face detection."""
        return self.analyze("faces", video_path=video_path, scenes_json=scenes_json, output_path=output_path, **kwargs)

    def identity(self, video_path: str, faces_json: str, scenes_json: str, output_path: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """Run identity tracking."""
        return self.analyze("identity", video_path=video_path, faces_json=faces_json, scenes_json=scenes_json, output_path=output_path, **kwargs)

    def conversation(self, video_path: str, scenes_json: str, faces_json: str, identities_json: str, output_path: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """Run conversation analysis."""
        return self.analyze("conversation", video_path=video_path, scenes_json=scenes_json, faces_json=faces_json, identities_json=identities_json, output_path=output_path, **kwargs)

    def ocr(self, video_path: str, output_path: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """Run OCR."""
        return self.analyze("ocr", video_path=video_path, output_path=output_path, **kwargs)

    def objects(self, video_path: str, output_path: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """Run object detection."""
        return self.analyze("objects", video_path=video_path, output_path=output_path, **kwargs)

    def audio(self, video_path: str, output_path: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """Run audio analysis."""
        return self.analyze("audio", video_path=video_path, output_path=output_path, **kwargs)

    def shutdown(self) -> Dict[str, Any]:
        """Shutdown the server."""
        return self._request("POST", "shutdown")