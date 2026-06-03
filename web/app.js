const state = {
  stats: null,
  history: [],
};

const el = (id) => document.getElementById(id);

const statCount = el("statCount");
const statPred = el("statPred");
const statSpeed = el("statSpeed");
const statAnom = el("statAnom");
const statAvgCount = el("statAvgCount");
const statMaxCount = el("statMaxCount");
const statMaxSpeed = el("statMaxSpeed");
const statAnomRate = el("statAnomRate");
const statPeakTime = el("statPeakTime");
const statFrames = el("statFrames");
const statFps = el("statFps");
const statusPill = el("statusPill");
const lastUpdate = el("lastUpdate");
const countChart = el("countChart");
const chatLog = el("chatLog");
const chatForm = el("chatForm");
const chatInput = el("chatInput");
const chatButton = chatForm.querySelector("button");

function formatNumber(value, digits = 2) {
  if (value === undefined || value === null || Number.isNaN(value)) {
    return "--";
  }
  return Number(value).toFixed(digits);
}

function formatInt(value) {
  if (value === undefined || value === null || Number.isNaN(value)) {
    return "--";
  }
  return String(Math.round(Number(value)));
}

function updateStats(stats) {
  if (!stats) {
    return;
  }
  statCount.textContent = formatInt(stats.count);
  statPred.textContent = formatNumber(stats.predicted, 1);
  statSpeed.textContent = formatNumber(stats.avg_speed, 2);
  statAnom.textContent = formatInt(stats.anomaly_count);
  statAvgCount.textContent = formatNumber(stats.avg_count, 2);
  statMaxCount.textContent = formatInt(stats.max_count);
  statMaxSpeed.textContent = formatNumber(stats.max_speed, 2);
  statAnomRate.textContent = formatNumber(stats.anomaly_rate, 4);
  statPeakTime.textContent =
    stats.peak_time_s !== undefined ? `${formatNumber(stats.peak_time_s, 1)}s` : "--";
  statFrames.textContent = formatInt(stats.total_frames);
  statFps.textContent = formatNumber(stats.fps, 1);

  if (stats.status === "running" || stats.status === "finished") {
    statusPill.textContent = stats.status;
    statusPill.classList.add("live");
  } else {
    statusPill.textContent = "waiting";
    statusPill.classList.remove("live");
  }

  const timeText =
    stats.timestamp_s !== undefined
      ? `t=${formatNumber(stats.timestamp_s, 1)}s`
      : "--";
  lastUpdate.textContent = timeText;

  drawTrend(stats.series);
}

function drawTrend(series) {
  if (!series || !countChart) {
    return;
  }
  const ctx = countChart.getContext("2d");
  const ratio = window.devicePixelRatio || 1;
  const rect = countChart.getBoundingClientRect();
  countChart.width = Math.floor(rect.width * ratio);
  countChart.height = Math.floor(rect.height * ratio);
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  const width = rect.width;
  const height = rect.height;
  ctx.clearRect(0, 0, width, height);

  const counts = series.counts || [];
  const preds = series.preds || [];
  if (counts.length < 2) {
    ctx.fillStyle = "#65707d";
    ctx.font = "14px Arial";
    ctx.fillText("Ma'lumot kutilmoqda...", 20, 38);
    return;
  }

  const maxVal = Math.max(...counts, ...preds);
  const minVal = Math.min(...counts, ...preds);
  const pad = 20;
  const plotW = width - pad * 2;
  const plotH = height - pad * 2;
  const span = maxVal - minVal || 1;

  const toX = (i, total) => pad + (i / (total - 1)) * plotW;
  const toY = (val) => pad + plotH - ((val - minVal) / span) * plotH;

  ctx.strokeStyle = "#d9e0e8";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(pad, pad);
  ctx.lineTo(pad, height - pad);
  ctx.lineTo(width - pad, height - pad);
  ctx.stroke();

  drawLine(ctx, counts, "#146c94", toX, toY);
  drawLine(ctx, preds, "#b54708", toX, toY);
}

function drawLine(ctx, data, color, toX, toY) {
  if (!data || data.length < 2) {
    return;
  }
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.beginPath();
  data.forEach((value, index) => {
    const x = toX(index, data.length);
    const y = toY(value);
    if (index === 0) {
      ctx.moveTo(x, y);
    } else {
      ctx.lineTo(x, y);
    }
  });
  ctx.stroke();
}

async function fetchStats() {
  try {
    const res = await fetch("/api/stats", { cache: "no-store" });
    if (!res.ok) {
      return;
    }
    const data = await res.json();
    state.stats = data;
    updateStats(data);
  } catch (err) {
    console.warn("Stats fetch failed", err);
  }
}

function appendMessage(role, content) {
  const msg = document.createElement("div");
  msg.className = `msg ${role}`;
  msg.textContent = content;
  chatLog.appendChild(msg);
  chatLog.scrollTop = chatLog.scrollHeight;
  return msg;
}

async function sendChat(message) {
  const historySnapshot = state.history.slice(-8);
  const payload = {
    message,
    history: historySnapshot,
    stats: state.stats || {},
  };

  const res = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  if (!res.ok) {
    const errorBody = await res.json().catch(() => ({}));
    const errorText = errorBody.error || "Chat request failed";
    appendMessage("assistant error", errorText);
    return;
  }

  const data = await res.json();
  if (data.reply) {
    appendMessage("assistant", data.reply);
    state.history.push({ role: "assistant", content: data.reply });
  }
}

chatForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const message = chatInput.value.trim();
  if (!message) {
    return;
  }
  chatInput.value = "";
  appendMessage("user", message);
  state.history.push({ role: "user", content: message });
  chatInput.disabled = true;
  chatButton.disabled = true;
  const pending = appendMessage("assistant", "Javob tayyorlanmoqda...");
  try {
    await sendChat(message);
  } catch (err) {
    appendMessage("assistant error", "Chat bilan ulanishda xatolik yuz berdi.");
  } finally {
    pending.remove();
    chatInput.disabled = false;
    chatButton.disabled = false;
    chatInput.focus();
  }
});

fetchStats();
setInterval(fetchStats, 1000);
window.addEventListener("resize", () => drawTrend(state.stats?.series));
