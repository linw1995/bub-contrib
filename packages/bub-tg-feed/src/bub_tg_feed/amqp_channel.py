"""AMQP channel adapter."""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
from datetime import datetime
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse, urlunparse

import aio_pika
import aiohttp
from bub.channels import Channel
from bub.channels.message import ChannelMessage
from bub.types import MessageHandler
from loguru import logger

if TYPE_CHECKING:
    from aio_pika.abc import AbstractIncomingMessage

MQ_URL = os.environ["AMQP_URL"]
TELEGRAM_TOKEN = os.environ["BUB_TELEGRAM_TOKEN"]
MQ_EXCHANGE = "telegram.messages"


def _redact_url(value: str) -> str:
    parsed = urlparse(value)
    if parsed.username or parsed.password:
        redacted_netloc = f"****:{'****' if parsed.password else ''}@{parsed.hostname}"
        if parsed.port:
            redacted_netloc += f":{parsed.port}"
        redacted = parsed._replace(netloc=redacted_netloc)
        return urlunparse(redacted)
    return value


def _exclude_none(d: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in d.items() if v is not None}


def _payload_date_to_timestamp(value: Any) -> float | None:
    if not isinstance(value, str) or not value:
        return None
    candidate = value.replace("Z", "+00:00")
    with contextlib.suppress(ValueError):
        return datetime.fromisoformat(candidate).timestamp()
    return None


def _duration_seconds(value: Any) -> int:
    if isinstance(value, int | float):
        return int(value)
    return 0


def _payload_message_type(payload: dict[str, Any]) -> str:
    media_type = (
        payload.get("media", {}).get("type")
        if isinstance(payload.get("media"), dict)
        else None
    )
    if media_type == "gif":
        return "video"
    if isinstance(media_type, str) and media_type in {
        "photo",
        "audio",
        "sticker",
        "video",
        "voice",
        "document",
    }:
        return media_type
    if isinstance(payload.get("text"), str) and payload["text"].strip():
        return "text"
    return "unknown"


def _parse_payload(
    message: AbstractIncomingMessage,
) -> tuple[str, dict[str, Any] | None]:
    body = message.body.decode("utf-8", errors="replace").strip()
    if not body:
        return "", None
    with contextlib.suppress(json.JSONDecodeError):
        loaded = json.loads(body)
        if isinstance(loaded, dict):
            return body, loaded
    return body, None


def _build_telegram_like_media(
    payload: dict[str, Any],
) -> tuple[str, dict[str, Any] | None, str | None]:
    media = payload.get("media")
    if not isinstance(media, dict):
        text = payload.get("text")
        return (text.strip() if isinstance(text, str) else ""), None, None

    media_type = media.get("type") or "unknown"
    caption = payload.get("text")
    caption_text = caption if isinstance(caption, str) and caption else None
    duration = _duration_seconds(media.get("duration"))

    if media_type == "photo":
        formatted = (
            f"[Photo message] Caption: {caption_text}"
            if caption_text
            else "[Photo message]"
        )
        metadata = _exclude_none(
            {
                "file_id": media.get("file_id"),
                "file_size": media.get("size"),
                "width": media.get("width"),
                "height": media.get("height"),
            }
        )
        return formatted, metadata or None, caption_text

    if media_type == "audio":
        title = media.get("name") or "Unknown"
        formatted = f"[Audio: {title} ({duration}s)]"
        metadata = _exclude_none(
            {
                "file_id": media.get("file_id"),
                "file_size": media.get("size"),
                "duration": media.get("duration"),
                "title": media.get("name"),
                "mime_type": media.get("mime_type"),
            }
        )
        return formatted, metadata or None, caption_text

    if media_type == "sticker":
        set_name = media.get("class_name") or ""
        formatted = f"[Sticker from {set_name}]" if set_name else "[Sticker]"
        metadata = _exclude_none(
            {
                "file_id": media.get("file_id"),
                "width": media.get("width"),
                "height": media.get("height"),
                "set_name": media.get("class_name"),
            }
        )
        return formatted, metadata or None, caption_text

    if media_type in {"video", "gif"}:
        formatted = f"[Video: {duration}s]"
        formatted = (
            f"{formatted} Caption: {caption_text}" if caption_text else formatted
        )
        metadata = _exclude_none(
            {
                "file_id": media.get("file_id"),
                "file_size": media.get("size"),
                "width": media.get("width"),
                "height": media.get("height"),
                "duration": media.get("duration"),
            }
        )
        return formatted, metadata or None, caption_text

    if media_type == "voice":
        formatted = f"[Voice message: {duration}s]"
        metadata = _exclude_none(
            {
                "file_id": media.get("file_id"),
                "duration": media.get("duration"),
            }
        )
        return formatted, metadata or None, caption_text

    if media_type == "document":
        file_name = media.get("name") or "unknown"
        mime_type = media.get("mime_type") or "unknown"
        formatted = f"[Document: {file_name} ({mime_type})]"
        formatted = (
            f"{formatted} Caption: {caption_text}" if caption_text else formatted
        )
        metadata = _exclude_none(
            {
                "file_id": media.get("file_id"),
                "file_name": media.get("name"),
                "file_size": media.get("size"),
                "mime_type": media.get("mime_type"),
            }
        )
        return formatted, metadata or None, caption_text

    return "[Unknown message type]", None, caption_text


def _extract_prompt_parts(
    payload: dict[str, Any],
) -> tuple[str | None, str, dict[str, Any]]:
    chat_id = str(payload["chat_id"]) if payload.get("chat_id") is not None else None
    content = ""
    media_meta: dict[str, Any] | None = None
    caption: str | None = None
    if payload.get("has_media") and isinstance(payload.get("media"), dict):
        content, media_meta, caption = _build_telegram_like_media(payload)
    elif isinstance(payload.get("text"), str):
        content = payload["text"].strip()

    metadata: dict[str, Any] = {
        "message_id": payload.get("message_id"),
        "type": _payload_message_type(payload),
        "username": payload.get("sender_username") or "",
        "full_name": payload.get("sender_fullname") or "",
        "sender_id": str(payload["sender_id"])
        if payload.get("sender_id") is not None
        else "",
        "sender_is_bot": payload.get("is_bot"),
        "date": _payload_date_to_timestamp(payload.get("date")),
    }
    if media_meta:
        metadata["media"] = media_meta
        if caption:
            metadata["caption"] = caption
    reply_to = payload.get("reply_to")
    if payload.get("is_reply") and isinstance(reply_to, dict):
        metadata["reply_to_message"] = _exclude_none(
            {
                "message_id": reply_to.get("message_id"),
                "from_user_id": str(reply_to["sender_id"])
                if reply_to.get("sender_id") is not None
                else "",
                "from_username": reply_to.get("sender_username") or "",
                "from_full_name": reply_to.get("sender_fullname") or "",
                "text": (reply_to.get("text") or "")[:100]
                if isinstance(reply_to.get("text"), str)
                else "",
            }
        )
    elif payload.get("is_reply") and payload.get("reply_to_msg_id") is not None:
        # Backward compatibility for legacy payload format.
        metadata["reply_to_message"] = {"message_id": payload["reply_to_msg_id"]}
    return chat_id, content.strip(), metadata


def _allow_message(payload: dict[str, Any]) -> bool:
    msg_type = _payload_message_type(payload)
    if msg_type == "unknown":
        logger.warning("amqp.inbound.unknown_message_type {}", payload)
        return False

    text = payload.get("text")
    content = text.strip() if isinstance(text, str) else ""
    has_media = bool(
        payload.get("has_media") and isinstance(payload.get("media"), dict)
    )
    if not content and not has_media:
        logger.warning("amqp.inbound.empty_message {}", payload)
        return False

    return payload.get("sender_username") != "wordle_tg_bot"


class AMQPChannel(Channel):
    """AMQP adapter based on aio-pika."""

    name = "amqp"
    _stop_event: asyncio.Event
    _missing = object()

    def __init__(self, on_receive: MessageHandler) -> None:
        self._on_receive = on_receive
        self._task: asyncio.Task | None = None
        self._me: dict[str, Any] | object = self._missing
        self._typing_tasks: dict[str, asyncio.Task] = {}

    @property
    def needs_debounce(self) -> bool:
        return True

    async def start(self, stop_event: asyncio.Event) -> None:
        logger.info("amqp.start url={} exchange={}", _redact_url(MQ_URL), MQ_EXCHANGE)
        self._stop_event = stop_event
        self._task = asyncio.create_task(self._main_loop())

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        for task in self._typing_tasks.values():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        self._typing_tasks.clear()
        logger.info("amqp.stop complete")

    async def _main_loop(self) -> None:
        connection = await aio_pika.connect_robust(MQ_URL)
        try:
            async with connection:
                channel = await connection.channel()
                await channel.set_qos(prefetch_count=16)
                exchange = await channel.get_exchange(MQ_EXCHANGE, ensure=True)
                queue = await channel.declare_queue("bub.messages", durable=True)
                await queue.bind(exchange, routing_key="#")
                async with queue.iterator() as queue_iter:
                    # Cancel consuming after __aexit__
                    async for message in queue_iter:
                        async with message.process(
                            requeue=False, ignore_processed=True
                        ):
                            await self._consume_message(message)
        finally:
            logger.info("amqp.stopped")

    async def _get_me(self) -> dict[str, Any] | None:
        if self._me is not self._missing:
            return self._me
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/getMe"
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get("ok") and isinstance(data.get("result"), dict):
                        self._me = data["result"]
                logger.error(
                    "Failed to get bot info from Telegram status={} response={}",
                    response.status,
                    await response.text(),
                )
        self._me = None
        return self._me

    def is_mentioned(self, payload: dict[str, Any]) -> bool:
        _, content, metadata = _extract_prompt_parts(payload)
        content = content.lower()
        if "bub" in content:
            return True

        if self._me is not self._missing and self._me is not None:
            username = self._me.get("username", "").lower()
            userid = str(self._me.get("id", ""))
            if username and f"@{username}" in content:
                return True

            reply_to = metadata.get("reply_to_message")
            if reply_to and (
                reply_to.get("from_username", "").lower() == username
                or reply_to.get("from_user_id") == userid
            ):
                return True
        return False

    async def _consume_message(self, message: AbstractIncomingMessage) -> None:
        raw_body, payload = _parse_payload(message)
        if not raw_body or not payload:
            logger.warning(
                "amqp.inbound.empty routing_key={} message_id={}",
                message.routing_key,
                message.message_id,
            )
            return None
        if not _allow_message(payload):
            return None

        _, content, _ = _extract_prompt_parts(payload)
        if not content:
            logger.warning(
                "amqp.inbound.empty_content routing_key={} message_id={}",
                message.routing_key,
                message.message_id,
            )
            return None
        await self._get_me()
        channel_message = self._build_message(payload)
        await self._on_receive(channel_message)

    def _build_message(self, payload: dict[str, Any]) -> ChannelMessage:
        chat_id, content, metadata = _extract_prompt_parts(payload)
        if not chat_id:
            chat_id = "unknown_chat"
        channel = "telegram"
        session_id = f"{channel}:{chat_id}"
        if content.strip().startswith("/bub "):
            content = content.strip()[5:]
        if content.strip().startswith(","):
            return ChannelMessage(
                session_id=session_id,
                content=content,
                channel=channel,
                chat_id=chat_id,
                kind="command",
            )
        content = json.dumps({"message": content, **metadata}, ensure_ascii=False)
        message = ChannelMessage(
            session_id=session_id,
            content=content,
            channel=channel,
            chat_id=chat_id,
            is_active=self.is_mentioned(payload),
            lifespan=self.typing_indicator(chat_id),
        )
        return message

    @contextlib.asynccontextmanager
    async def typing_indicator(self, chat_id: str):
        if chat_id in self._typing_tasks:
            # Already sending typing indicator for this chat, do nothing.
            yield
            return

        task = asyncio.create_task(self._typing_loop(chat_id))
        self._typing_tasks[chat_id] = task
        try:
            yield
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
            del self._typing_tasks[chat_id]

    async def _typing_loop(self, chat_id: str) -> None:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendChatAction"
        payload = {"chat_id": chat_id, "action": "typing"}
        async with aiohttp.ClientSession() as session:
            while not self._stop_event.is_set():
                async with session.post(url, json=payload) as response:
                    if response.status != 200:
                        logger.error(
                            "Failed to send typing action to Telegram status={} response={}",
                            response.status,
                            await response.text(),
                        )
                await asyncio.sleep(4)
