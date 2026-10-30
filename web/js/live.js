// WebSocket state feed with automatic reconnect. Calls onState(state) and onEvent(event).
export function connectLive(token, { onState, onEvent, onStatus, onPositions }) {
  let ws, closed = false, delay = 500, ping;
  const open = () => {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    ws = new WebSocket(`${proto}://${location.host}/ws?t=${encodeURIComponent(token)}`);
    ws.onopen = () => { delay = 500; onStatus?.("online"); ping = setInterval(() => ws.readyState === 1 && ws.send("ping"), 15000); };
    ws.onmessage = (m) => {
      const msg = JSON.parse(m.data);
      if (msg.type === "state") onState?.(msg.data);
      else if (msg.type === "event") onEvent?.(msg.data);
      else if (msg.type === "positions") onPositions?.(msg.data);
    };
    ws.onclose = () => {
      clearInterval(ping);
      if (closed) return;
      onStatus?.("offline");
      setTimeout(open, delay);
      delay = Math.min(delay * 2, 8000);
    };
    ws.onerror = () => ws.close();
  };
  open();
  return { close: () => { closed = true; ws?.close(); } };
}
