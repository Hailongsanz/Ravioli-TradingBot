"""
Ravioli — Trading Dashboard
Run: python app.py → opens http://localhost:8080
"""

import asyncio
import json
import logging
import os
import signal
import subprocess
import sys
import threading
import webbrowser
from pathlib import Path

import uvicorn
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse

from bot_engine import BotEngine

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
LOG_FILE = "trades.log"
logger = logging.getLogger("TradingBot")
logger.setLevel(logging.DEBUG)
fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

fh = logging.FileHandler(LOG_FILE)
fh.setLevel(logging.DEBUG)
fh.setFormatter(fmt)
logger.addHandler(fh)

if sys.stdout is not None:
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
engine = BotEngine()
clients: set[WebSocket] = set()
event_queue: asyncio.Queue = asyncio.Queue()


@asynccontextmanager
async def lifespan(app: FastAPI):
    asyncio.create_task(broadcast_loop())
    asyncio.get_event_loop().call_later(1.5, lambda: webbrowser.open("http://localhost:8080"))
    yield


app = FastAPI(title="Ravioli Trading Bot", lifespan=lifespan)

# Resolve static dir (works both dev and PyInstaller)
if getattr(sys, "frozen", False):
    BASE_DIR = Path(sys._MEIPASS)
else:
    BASE_DIR = Path(__file__).parent
STATIC_DIR = BASE_DIR / "static"


def on_bot_event(event: dict):
    try:
        event_queue.put_nowait(event)
    except Exception:
        pass


engine.subscribe(on_bot_event)


# ---------------------------------------------------------------------------
# WebSocket broadcaster
# ---------------------------------------------------------------------------
async def broadcast_loop():
    global clients
    while True:
        event = await event_queue.get()
        msg = json.dumps(event, default=str)
        dead = set()
        for ws in clients:
            try:
                await ws.send_text(msg)
            except Exception:
                dead.add(ws)
        clients -= dead


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/status")
async def get_status():
    return JSONResponse(engine.get_state())


@app.get("/api/bars")
async def get_bars():
    return JSONResponse(engine.get_bars_snapshot())


@app.get("/api/trades")
async def get_trades():
    return JSONResponse(engine.trade_history)


@app.post("/api/bot/start")
async def start_bot():
    try:
        result = await engine.start()
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


@app.post("/api/bot/stop")
async def stop_bot():
    try:
        result = await engine.stop()
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


@app.post("/api/symbol")
async def set_symbol(request: Request):
    body = await request.json()
    symbol = body.get("symbol", "").strip().upper()
    if not symbol:
        return JSONResponse({"status": "error", "message": "No symbol provided"}, status_code=400)
    if symbol == engine.symbol:
        return JSONResponse({"status": "ok", "symbol": symbol})
    if engine.is_running:
        await engine.stop()
    engine.set_symbol(symbol)
    return JSONResponse({"status": "ok", "symbol": symbol, "restart_required": True})


@app.post("/api/quit")
async def quit_app():
    """Shut down the entire application."""
    if engine.is_running:
        await engine.stop()
    # Delay shutdown slightly so the response gets sent
    threading.Timer(0.5, lambda: os._exit(0)).start()
    return JSONResponse({"status": "shutting_down"})


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    clients.add(ws)
    try:
        await ws.send_text(json.dumps({"type": "status", "data": engine.get_state()}, default=str))
    except Exception:
        pass
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        clients.discard(ws)


# ---------------------------------------------------------------------------
# System tray
# ---------------------------------------------------------------------------
def run_tray():
    """System tray icon with menu."""
    import pystray
    from PIL import Image, ImageDraw

    # Create a simple green "R" icon
    def create_icon():
        img = Image.new("RGB", (64, 64), "#0d1117")
        draw = ImageDraw.Draw(img)
        # Green circle
        draw.ellipse([8, 8, 56, 56], fill="#3fb950")
        # "R" letter
        draw.text((22, 14), "R", fill="#0d1117")
        return img

    def on_open(icon, item):
        webbrowser.open("http://localhost:8080")

    def on_quit(icon, item):
        icon.stop()
        # Stop bot and exit
        if engine.is_running:
            engine._running = False
        os._exit(0)

    icon = pystray.Icon(
        "Ravioli",
        create_icon(),
        "Ravioli Trading Bot",
        menu=pystray.Menu(
            pystray.MenuItem("Open Dashboard", on_open, default=True),
            pystray.MenuItem("Quit Ravioli", on_quit),
        ),
    )
    icon.run()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def kill_existing_instance(port=8080):
    """Kill any process already using our port so a fresh instance can start."""
    try:
        result = subprocess.run(
            ["netstat", "-ano"],
            capture_output=True, text=True, creationflags=0x08000000  # CREATE_NO_WINDOW
        )
        for line in result.stdout.splitlines():
            if f":{port}" in line and "LISTENING" in line:
                pid = int(line.strip().split()[-1])
                if pid != os.getpid():
                    os.kill(pid, signal.SIGTERM)
    except Exception:
        pass


if __name__ == "__main__":
    # In windowed mode, sys.stdout/stderr are None — uvicorn needs them
    if sys.stdout is None:
        sys.stdout = open(os.devnull, "w")
    if sys.stderr is None:
        sys.stderr = open(os.devnull, "w")

    kill_existing_instance()

    try:
        # Start tray icon in background thread
        tray_thread = threading.Thread(target=run_tray, daemon=True)
        tray_thread.start()

        uvicorn.run(app, host="0.0.0.0", port=8080, log_level="warning")
    except Exception as e:
        # Write crash info to file since there's no console
        with open("ravioli_crash.log", "w") as f:
            import traceback
            f.write(traceback.format_exc())
        raise
