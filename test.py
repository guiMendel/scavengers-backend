import asyncio
import websockets
import json
from datetime import datetime

host = "localhost"
port = 3001

# Maps each agent id to it's knowledge base
scavengers = {}

# Helper to print with a timestamp
def log(message):
    print(f"[{datetime.now().time()}]> {message}")


# Handles a message. The returned stirng should be the response, unless it's null
def handle_message(message):
    # Identify message type
    if "connect" in message:
        log(f"{message['id']} connected")

        # Register it
        scavengers[message['id']] = {}

    if "request" in message:
        return "random"


async def connection_handler(websocket):
    log("App connected")

    # Start listening for messages
    try:
        async for raw_message in websocket:
            message = json.loads(raw_message)
            
            # log(f"Received message from {message['id']}: {message}")
            response = handle_message(message)

            if response is not None:
                await websocket.send(response)

    # Ignore connection close errors
    except websockets.exceptions.ConnectionClosedError:
        pass

    finally:
        log("App disconnected")

        # Unregister agents
        scavengers.clear()


async def main():
    try:
        async with websockets.serve(connection_handler, host, port):
            log(f"Server listening at {host}:{port}")
            await asyncio.Future()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    asyncio.run(main())
