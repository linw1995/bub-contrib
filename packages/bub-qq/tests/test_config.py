from __future__ import annotations

from pydantic import ValidationError

from bub_qq.config import QQConfig


def test_inbound_dedupe_size_must_be_positive() -> None:
    try:
        QQConfig(receive_mode="webhook", inbound_dedupe_size=0)
    except ValidationError as exc:
        assert "inbound_dedupe_size" in str(exc)
    else:
        raise AssertionError("expected inbound_dedupe_size=0 to be rejected")


def test_receive_mode_is_required() -> None:
    try:
        QQConfig()
    except ValidationError as exc:
        assert "receive_mode" in str(exc)
    else:
        raise AssertionError("expected missing receive_mode to be rejected")


def test_webhook_port_defaults_to_official_allowed_port() -> None:
    config = QQConfig(receive_mode="webhook")

    assert config.webhook_port == 8080
