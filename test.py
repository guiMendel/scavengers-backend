import asyncio
import websockets
from datetime import datetime

host = "localhost"
port = 3001

# Maps each agent id to it's websocket connection
scavengers = {}

# Helper to print with a timestamp
def log(message):
    print(f"[{datetime.now().time()}]> {message}")


async def connection_handler(websocket):
    log(f"Connection with {websocket.id}")

    # FOR TEST PURPOSES ONLY
    await websocket.send("hullo")

    # Get agent id
    id = await websocket.recv()

    log(f"{id} connected")

    # Register it
    scavengers[id] = websocket

    try:
        async for message in websocket:
            log(f"Received message from {id}: {message}")
            websocket.send("random")

    finally:
        log(f"{id} disconnected")

        # Unregister it
        scavengers.pop(id)


async def main():
    try:
        async with websockets.serve(connection_handler, host, port):
            log(f"Server listening at {host}:{port}")
            await asyncio.Future()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    asyncio.run(main())
