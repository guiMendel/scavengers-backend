import asyncio
from time import time
import websockets
import json
from datetime import datetime
from learning.index import new_agent, actions
from learning.agent import Agent

host = "localhost"
port = 3001

# Maps each agent id to it's Agent instance
scavengers: "dict[str, Agent]" = {}

# Keeps count of the average response time
serve_time: "dict[str, int]" = {}

# Helper to print with a timestamp
def log(message):
    print(f"[{datetime.now().time()}]> {message}")


# Handles a message. The returned string should be the response, unless it's null
def handle_message(message):
    # Get agent id
    id: str = message["id"]

    # Identify message type
    if "connect" in message:
        log(f"{id} connected")

        # Register it
        scavengers[id] = new_agent(id)

    if "request" in message:
        # Track serve time
        start = time()
        
        # Input this state observation (and reward) into RL model and get next action
        action_index: int = scavengers[id].iterate(message["request"])

        # Get action
        action = id + ';' + actions[action_index]

        # Print serve time
        serve_time["current"] = time() - start
        log(f"Served \"{action}\" to \"{id}\" in {serve_time['current'] * 1000} ms")
        serve_time["average"] = (serve_time["average"] + serve_time["current"]) / 2 \
            if "average" in serve_time else serve_time["current"]

        # Translate this index directly into the corresponding action
        return action

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

        if "average" in serve_time:
            # Log average time
            log(f"Average serve time was {serve_time['average'] * 1000} ms")

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
