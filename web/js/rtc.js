// WHIP (publish) and WHEP (play) over the hub's signalling endpoints. LAN only: no STUN/TURN.
import { api } from "./core.js";

const RTC_CONFIG = { iceServers: [] };

async function signal(url, pc, token) {
  const offer = await pc.createOffer();
  await pc.setLocalDescription(offer);
  await waitIce(pc);
  const r = await api(url, { method: "POST", body: pc.localDescription.sdp, raw: true, token });
  const answer = await r.text();
  await pc.setRemoteDescription({ type: "answer", sdp: answer });
  return r.headers.get("Location");
}

function waitIce(pc) {
  if (pc.iceGatheringState === "complete") return Promise.resolve();
  return new Promise((resolve) => {
    const done = () => { pc.removeEventListener("icegatheringstatechange", onChange); resolve(); };
    const onChange = () => { if (pc.iceGatheringState === "complete") done(); };
    pc.addEventListener("icegatheringstatechange", onChange);
    setTimeout(done, 1500); // host candidates arrive fast on a LAN; don't wait forever
  });
}

/** Publish a MediaStream to /whip/<camId>. Returns {pc, stop}. */
export async function publish(camId, stream, token, { maxKbps = 4000 } = {}) {
  const pc = new RTCPeerConnection(RTC_CONFIG);
  for (const track of stream.getTracks()) {
    const sender = pc.addTrack(track, stream);
    if (track.kind === "video") {
      const p = sender.getParameters();
      p.encodings = [{ maxBitrate: maxKbps * 1000 }];
      p.degradationPreference = "maintain-framerate";
      try { await sender.setParameters(p); } catch { /* older browsers */ }
    }
  }
  pc.getTransceivers().forEach((tr) => { tr.direction = "sendonly"; });
  const location = await signal(`/whip/${camId}`, pc, token);
  return {
    pc,
    stop: async () => {
      pc.close();
      if (location) try { await api(location, { method: "DELETE", token }); } catch { /* ignore */ }
    },
  };
}

/** Play camera <camId> into a <video>. Returns {pc, stop}. */
export async function play(camId, videoEl, token) {
  const pc = new RTCPeerConnection(RTC_CONFIG);
  pc.addTransceiver("video", { direction: "recvonly" });
  pc.addTransceiver("audio", { direction: "recvonly" });
  const stream = new MediaStream();
  pc.ontrack = (e) => { stream.addTrack(e.track); videoEl.srcObject = stream; videoEl.play().catch(() => {}); };
  const location = await signal(`/whep/${camId}`, pc, token);
  return {
    pc,
    stop: async () => {
      pc.close();
      if (location) try { await api(location, { method: "DELETE", token }); } catch { /* ignore */ }
    },
  };
}
