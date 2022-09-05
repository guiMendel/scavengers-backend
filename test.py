import asyncio
import websockets
from datetime import datetime


def log(message):
    print(f"[{datetime.now().time()}]> {message}")


async def connection_handler(websocket):
    log(f"New connection with {websocket.id}: {websocket.local_address}")

    try:
        async for message in websocket:
            log(f"Received message: {message}")

    finally:
        log(f"Closing connection with {websocket.id}")


async def main():
    async with websockets.serve(connection_handler, "localhost", 3001):
        log("Server listening")
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
