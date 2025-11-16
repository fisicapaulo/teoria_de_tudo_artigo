# -*- coding: utf-8 -*-
# I/O utilitários: leitura/escrita JSON canônico com SHA-256.

import json
import hashlib
from typing import Any, Dict


def dumps_canonical(obj: Dict[str, Any]) -> bytes:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_json_with_sha256(path: str, payload: Dict[str, Any], insert_checksum_field: bool = True) -> str:
    data = dict(payload)
    if insert_checksum_field:
        # Remover sha256 se vier no payload
        data.pop("sha256", None)
        raw = dumps_canonical(data)
        digest = sha256_bytes(raw)
        data["sha256"] = digest
        raw = dumps_canonical(data)
    else:
        raw = dumps_canonical(data)
        digest = sha256_bytes(raw)
    with open(path, "wb") as f:
        f.write(raw)
    return digest
