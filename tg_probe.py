"""Throwaway: probe which request shapes survive the current connection."""
import asyncio
from pathlib import Path

from telethon import TelegramClient, functions
from telethon.sessions import StringSession


def log(*args):
    print(*args, flush=True)


def read_env():
    values = {}
    for line in Path('/home/bogdanoff/apps/telegram-mcp/.env').read_text().splitlines():
        line = line.strip()
        if line and not line.startswith('#') and '=' in line:
            key, value = line.split('=', 1)
            values[key.strip()] = value.strip().strip('"').strip("'")
    return values


async def attempt(label, coro, timeout=25):
    try:
        result = await asyncio.wait_for(coro, timeout=timeout)
    except Exception as error:  # noqa: BLE001
        log('  %-22s FAILED %s' % (label, type(error).__name__))
        return None
    log('  %-22s OK' % label)
    return result


async def main():
    env = read_env()
    client = TelegramClient(
        StringSession(env['TELEGRAM_SESSION_STRING']),
        int(env['TELEGRAM_API_ID']), env['TELEGRAM_API_HASH'],
        timeout=20, connection_retries=2, retry_delay=1,
    )
    await client.connect()
    log('connected:', client.is_connected())

    me = await attempt('get_me', client.get_me())
    if me:
        log('    me ok')

    pinned = await attempt('GetPinnedDialogs', client(functions.messages.GetPinnedDialogsRequest(folder_id=0)))
    if pinned is not None:
        log('    pinned dialogs:', len(pinned.dialogs), 'chats:', len(pinned.chats))
        for chat in pinned.chats:
            log('    -', getattr(chat, 'id', '?'), getattr(chat, 'title', None))

    dialogs = await attempt('GetDialogs(limit=5)',
                            client(functions.messages.GetDialogsRequest(
                                offset_date=None, offset_id=0, offset_peer=None,
                                limit=5, hash=0)), timeout=30)
    if dialogs is not None:
        log('    dialogs:', len(dialogs.dialogs))

    for attempt_no in (1, 2):
        result = await attempt('get_dialogs retry %d' % attempt_no,
                               client.get_dialogs(limit=10), timeout=30)
        if result:
            for dialog in result:
                log('    *', getattr(dialog, 'name', None))
            break

    await client.disconnect()


asyncio.run(main())
