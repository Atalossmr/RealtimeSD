"use strict";

// ---------- 全局状态 ----------
const state = {
  uri: null,
  segments: [],            // [{speaker_id, start, end, text|null}]
  duration: 0,             // 音频总时长（秒）
  viewStart: 0, viewEnd: 0, // 当前可视时间窗（秒）
  peaks: null,             // 波形 min/max 桶
  speakerVisible: new Map(),
  activeIdx: -1,
  overlapFlags: [],          // 每段是否与其他说话人存在时间重叠
  speakersInfo: null,        // refined 级 sidecar（{uri}.speakers.json）：时长/uncertain/合并事件
  // ---- 分层数据源（stage 越多信息越多）----
  source: null,              // 段结构数据源：segments（stage2）/ rttm（stage1）/ transcript（无 manifest 时的兜底）
  hasSegments: false,
  segFiles: new Map(),       // "spk{id}_{start}" -> 分段 wav 文件名（manifest 建立）
  textMap: new Map(),        // "spk{id}_{start}" -> 转写文本（transcript 叠加层，不改段结构）
  lastRaw: "",               // 上一轮的源数据 JSON 串（rttm/segments 源比对用）
  lastTranscriptRaw: "",     // 上一轮的 transcript JSON 串（叠加层比对用）
};

const PALETTE = ["#4f9cf9","#f97583","#56d4a0","#e3b341","#b083f0","#ff9f43",
                 "#39c5cf","#f778ba","#7ee787","#ffa657","#76a8ff","#d2a8ff"];

const player = document.getElementById("player");
const segPlayer = document.getElementById("segPlayer");
const canvas = document.getElementById("timeline");
const ctx = canvas.getContext("2d");
const tooltip = document.getElementById("tooltip");
const statusEl = document.getElementById("status");
const transcriptEl = document.getElementById("transcript");

const spkColor = id => PALETTE[id % PALETTE.length];
const fmtTime = t => {
  const h = Math.floor(t / 3600), m = Math.floor(t / 60) % 60, s = t % 60;
  const base = `${String(m).padStart(2,"0")}:${s.toFixed(3).padStart(6,"0")}`;
  return h ? `${h}:${base}` : base;
};

// ---------- refined 级 speaker 状态（uncertain 标记 / 合并事件） ----------
const uncertainToggle = document.getElementById("uncertainToggle");
const uncertainToggleWrap = document.getElementById("uncertainToggleWrap");

// global speaker id -> sidecar 条目
function spkInfoMap() {
  const map = new Map();
  for (const s of state.speakersInfo?.speakers || []) map.set(s.id, s);
  return map;
}

// 展示用有效 id：沿 merged_into 链解析，被并说话人归到幸存者
function effId(id) {
  const map = spkInfoMap();
  const seen = new Set();
  let cur = id;
  while (map.get(cur)?.merged_into != null && !seen.has(cur)) {
    seen.add(cur);
    cur = map.get(cur).merged_into;
  }
  return cur;
}

// sidecar 开启了小样本阈值（post_merge_min_count > 0）才提供 uncertain 语义
function uncertainEnabled() {
  return (state.speakersInfo?.post_merge_min_count || 0) > 0;
}

function showUncertain(id) {
  return uncertainToggle.checked && spkInfoMap().get(id)?.uncertain === true;
}

// 段的展示标签：合并显示 spkA→spkB，uncertain 加 ?
function spkLabel(id) {
  const eff = effId(id);
  const base = eff !== id ? `spk${id}→spk${eff}` : `spk${eff}`;
  return showUncertain(eff) ? base + "?" : base;
}

// sidecar 变化后重算派生状态（合并改变 effId，进而影响重叠判定/颜色/chip）
function applySpeakerInfo() {
  uncertainToggleWrap.hidden = !uncertainEnabled();
  state.overlapFlags = computeOverlapFlags(state.segments);
  buildSpeakerChips();
  renderTranscript();
}

uncertainToggle.onchange = applySpeakerInfo;

// 拉取 refined 级 sidecar；apply=false 时只更新状态（调用方随后统一渲染）
async function fetchSpeakersInfo(uri, apply = true) {
  try {
    const resp = await fetch(`/api/speakers/${encodeURIComponent(uri)}`);
    if (!resp.ok) return;
    const info = await resp.json();
    if (JSON.stringify(info) === JSON.stringify(state.speakersInfo)) return;
    state.speakersInfo = info;
    if (apply) applySpeakerInfo();
  } catch { /* sidecar 暂不可读（原子替换中），下轮轮询再试 */ }
}

// 不同说话人的段时间相交 = 语音重叠（segments 已按 start 排序，内层可提前 break；
// 说话人按合并后的有效 id 判定——已并入同一幸存者的段不算重叠）
function computeOverlapFlags(segments) {
  const flags = new Array(segments.length).fill(false);
  for (let i = 0; i < segments.length; i++) {
    for (let j = i + 1; j < segments.length; j++) {
      if (segments[j].start >= segments[i].end) break;
      if (effId(segments[j].speaker_id) !== effId(segments[i].speaker_id)) {
        flags[i] = flags[j] = true;
      }
    }
  }
  return flags;
}

// ---------- 分离段音频（stage>=2：manifest 建立的 start+speaker -> 文件名 映射） ----------
// manifest 段与 transcript 条目的 speaker_id/start/end 完全一致（ASR 原样复制
// manifest 记录），二者共用同一键型；transcript 的文本经 textMap 叠加到段上。
function segKey(s) {
  return `spk${s.speaker_id}_${s.start.toFixed(3)}`;
}

function buildSegFiles(entries) {
  state.segFiles = new Map(entries.map(e => [segKey(e), e.file]));
}

function segFileOf(s) {
  return state.segFiles.get(segKey(s)) || null;
}

function playSegFile(file, btn) {
  segPlayer.src =
    `/api/segment_audio/${encodeURIComponent(state.uri)}/${encodeURIComponent(file)}`;
  segPlayer.play().catch(() => {});
  document.querySelectorAll(".segplay.playing")
    .forEach(el => el.classList.remove("playing"));
  if (btn) btn.classList.add("playing");
}
segPlayer.onended = () =>
  document.querySelectorAll(".segplay.playing")
    .forEach(el => el.classList.remove("playing"));

// ---------- 数据加载（服务器模式：SSE 事件驱动为主，轮询降级） ----------
const POLL_INTERVAL = 2000;
const liveBadge = document.getElementById("liveBadge");

function setLiveBadge(live) { liveBadge.hidden = !live; }

async function fetchJSON(url) {
  const resp = await fetch(url);
  if (!resp.ok) return null;
  return resp.json();
}

// 用最新的会话列表刷新下拉框（新增会话、audio/live/能力位变化），保留当前选择
function updateSessionSelect(sessions) {
  const select = document.getElementById("sessionSelect");
  for (const s of sessions) {
    let opt = select.querySelector(`option[value="${CSS.escape(s.uri)}"]`);
    if (!opt) {
      opt = document.createElement("option");
      opt.value = s.uri;
      select.appendChild(opt);
    }
    // 能力标签：随产物出现逐档升级（rttm → +分离段 → +转写）
    const caps = [];
    if (s.has_rttm) caps.push("rttm");
    if (s.has_segments) caps.push("分离段");
    if (s.has_transcript) caps.push("转写");
    opt.textContent = s.uri + (caps.length ? `（${caps.join(" + ")}）` : "") +
      (s.audio_url ? "" : "（无匹配音频）");
    opt.dataset.audioUrl = s.audio_url || "";
    opt.dataset.live = s.live ? "1" : "";
    opt.dataset.transcript = s.has_transcript ? "1" : "";
    opt.dataset.segments = s.has_segments ? "1" : "";
    opt.dataset.rttm = s.has_rttm ? "1" : "";
  }
  select.hidden = select.options.length === 0;
  if (!select.value && select.options.length) {
    select.selectedIndex = 0;
    loadServerSession(select.value);
  }
}

async function loadServerSessions() {
  try {
    const resp = await fetch("/api/sessions");
    if (!resp.ok) return false;
    const sessions = await resp.json();
    const select = document.getElementById("sessionSelect");
    updateSessionSelect(sessions);
    select.onchange = () => loadServerSession(select.value);
    if (select.value) {
      await loadServerSession(select.value);
    } else {
      statusEl.textContent = "等待管线产出…";
    }
    return true; // 增量更新通道（SSE/轮询）由 init 决定
  } catch { return false; } // file:// 打开时 fetch 必失败 → 静态模式
}

function currentOption() {
  const select = document.getElementById("sessionSelect");
  return select.value
    ? select.querySelector(`option[value="${CSS.escape(select.value)}"]`)
    : null;
}

// 段结构数据源按能力位选择：segments manifest（跟 diarization 节奏，最快）
// > refined RTTM（仅 diarization 时间线）；transcript 只做文本叠加层，
// 仅在无 manifest 的会话（一次性 ASR 等）兜底为段结构源
function wantedSource(opt) {
  if (opt?.dataset.segments === "1") return "segments";
  if (opt?.dataset.transcript === "1") return "transcript";
  return "rttm";
}

// transcript 文本叠加层：拉取并按键贴到 textMap，有变化才重渲染。
// 段结构不随它变——ASR 转写慢于产段，显示节奏不应跟随 ASR。
async function pollTranscriptOverlay(uri) {
  const list = await fetchJSON(`/api/transcript/${encodeURIComponent(uri)}`);
  if (!list) return;
  const raw = JSON.stringify(list);
  if (raw === state.lastTranscriptRaw) return;
  state.lastTranscriptRaw = raw;
  state.textMap = new Map(list.map(e => [segKey(e), e.text]));
  renderTranscript();
}

async function fetchSourceSegments(uri, source) {
  if (source === "transcript") {
    const list = await fetchJSON(`/api/transcript/${encodeURIComponent(uri)}`);
    return list ? { raw: JSON.stringify(list), segments: list } : null;
  }
  if (source === "segments") {
    const list = await fetchJSON(`/api/segments/${encodeURIComponent(uri)}`);
    if (!list) return null;
    buildSegFiles(list);
    return {
      raw: JSON.stringify(list),
      segments: list.map(e => ({
        speaker_id: e.speaker_id, start: e.start, end: e.end, text: null,
      })),
    };
  }
  const list = await fetchJSON(`/api/rttm/${encodeURIComponent(uri)}`);
  return list ? { raw: JSON.stringify(list), segments: list } : null;
}

// 拉取会话列表并刷新当前会话（SSE sessions 事件与轮询共用）
async function refreshSessions() {
  try {
    const resp = await fetch("/api/sessions");
    if (!resp.ok) return;
    updateSessionSelect(await resp.json());
    const opt = currentOption();
    if (opt) await refreshCurrentSession(opt);
  } catch { /* 服务器暂时不可达，下轮重试 */ }
}

// 拉取当前段结构源，有变化才更新（SSE artifact 事件与轮询共用）
async function refreshSource(opt) {
  const data = await fetchSourceSegments(opt.value, state.source);
  if (data && data.raw !== state.lastRaw) {
    state.lastRaw = data.raw;
    // transcript 源（无 manifest 的兜底）沿用前缀增量追加；
    // rttm/segments 源会被 merge 事件或新段重写/追加，统一整体重渲染
    if (state.source === "transcript" &&
        data.segments.length !== state.segments.length) {
      mergeSegments(data.segments);
    } else if (state.source !== "transcript") {
      replaceSegments(data.segments);
    }
  }
}

// 当前会话的整体刷新：音频补挂、能力位升档、段结构源、文本叠加层、
// speakers sidecar、live 标记（事件与轮询共用的完整路径）
async function refreshCurrentSession(opt) {
  // 音频可能晚于产物被匹配到，出现后补挂
  if (opt.dataset.audioUrl && !player.getAttribute("src")) {
    player.src = opt.dataset.audioUrl;
  }
  // 能力位升档（rttm → segments）时在原状态上叠加新内容，
  // 不做整页重载（保留说话人可见性、时间线视口、滚动位置与播放状态）
  if (wantedSource(opt) !== state.source) {
    await upgradeSession(opt);
  } else {
    await refreshSource(opt);
    if (state.hasSegments) {
      // manifest append-only：段数变了才重建映射并重渲染（补 ▶ 按钮）
      const list = state.source === "segments" ? null :
        await fetchJSON(`/api/segments/${encodeURIComponent(opt.value)}`);
      if (list && list.length !== state.segFiles.size) {
        buildSegFiles(list);
        renderTranscript();
      }
    }
  }
  // transcript 文本叠加层：转写出一段就补一段文本，
  // 段结构始终跟随 manifest/RTTM（diarization 节奏）
  if (opt.dataset.transcript === "1" && state.source !== "transcript") {
    await pollTranscriptOverlay(opt.value);
  }
  await fetchSpeakersInfo(opt.value);
  setLiveBadge(opt.dataset.live === "1");
}

// SSE artifact 事件：按 kind 只拉变化的产物（其余逻辑经 raw 比对自然去重）
async function refreshArtifact(uri, kind) {
  const opt = currentOption();
  if (!opt || opt.value !== uri) return; // 只跟随当前会话的产物
  if (kind === "speakers") {
    await fetchSpeakersInfo(uri);
  } else if (kind === "transcript" && state.source !== "transcript") {
    await pollTranscriptOverlay(uri);
  } else {
    // rttm / segments（及 transcript 兜底源）：走完整会话级刷新，
    // 兼顾能力位升档与 merge 重写（内部经 raw 比对，无变化即无操作）
    await refreshCurrentSession(opt);
  }
}

// SSE 通道：产物变化由服务端推送，EventSource 自带断线重连；
// 从未连接成功（旧服务端无端点/不支持 EventSource）才降级为轮询
function startEventStream() {
  if (typeof EventSource === "undefined") return false;
  let opened = false;
  const es = new EventSource("/api/events");
  es.onopen = () => { opened = true; };
  es.onmessage = event => {
    let msg;
    try { msg = JSON.parse(event.data); } catch { return; }
    if (msg.type === "sessions") refreshSessions();
    else if (msg.type === "artifact") refreshArtifact(msg.uri, msg.kind);
  };
  es.onerror = () => {
    if (!opened) {
      es.close();
      setTimeout(pollUpdate, POLL_INTERVAL); // 降级：2s 轮询
    }
    // 连上过：依赖 EventSource 自动重连，不介入
  };
  return true;
}

// 降级通道：2s 轮询（SSE 不可用时的兜底，行为与旧版一致）
async function pollUpdate() {
  await refreshSessions();
  setTimeout(pollUpdate, POLL_INTERVAL);
}

// 增量合并新的转写结果（跟随落盘是全量排序重写，旧段通常是新段的前缀）
function mergeSegments(segments) {
  const old = state.segments;
  const isPrefix = old.every((s, i) => {
    const n = segments[i];
    return n && n.start === s.start && n.end === s.end
      && n.speaker_id === s.speaker_id;
  });
  const knownSpeakers = new Set(state.speakerVisible.keys());
  const hasNewSpeaker = segments.some(s => !knownSpeakers.has(effId(s.speaker_id)));
  state.segments = segments;
  state.overlapFlags = computeOverlapFlags(segments);
  if (hasNewSpeaker) buildSpeakerChips(); // 新说话人需要补 chip（保留已有可见性）
  if (isPrefix && !hasNewSpeaker) {
    const stickToBottom = isTranscriptAtBottom();
    for (let i = old.length; i < segments.length; i++) {
      appendTranscriptRow(segments[i], i);
    }
    if (stickToBottom) scrollTranscriptToBottom();
  } else {
    renderTranscript();
  }
  statusEl.textContent = `${state.uri}: ${segments.length} 段`;
}

// 非 transcript 源（rttm / segments）的整体替换：merge 事件会重写历史行，
// 前缀假设不成立，统一重渲染
function replaceSegments(segments) {
  const knownSpeakers = new Set(state.speakerVisible.keys());
  const hasNewSpeaker = segments.some(s => !knownSpeakers.has(effId(s.speaker_id)));
  state.segments = segments;
  state.overlapFlags = computeOverlapFlags(segments);
  if (hasNewSpeaker) buildSpeakerChips();
  renderTranscript();
  statusEl.textContent = `${state.uri}: ${segments.length} 段`;
}

// 能力位升档（rttm → segments）：只切换段结构数据源并叠加新内容
// （分离段 ▶ 按钮），保留说话人可见性、时间线视口、滚动位置、
// 主/段播放器状态——区别于 loadServerSession 的整页重置（初始加载与
// 切换会话仍走那里）。transcript 不参与升档：它是文本叠加层，
// 由 pollTranscriptOverlay 独立轮询。
async function upgradeSession(opt) {
  const uri = opt.value;
  const source = wantedSource(opt);
  const data = await fetchSourceSegments(uri, source);
  // 新源产物刚出现但还没有内容（如 transcript 已建空文件）时本轮不升档，
  // 避免用空列表顶掉已有的段；下轮轮询再试
  if (!data || (data.segments.length === 0 && state.segments.length > 0)) return;
  state.source = source;
  state.hasSegments = opt.dataset.segments === "1";
  state.lastRaw = data.raw;
  state.segments = data.segments;
  state.overlapFlags = computeOverlapFlags(state.segments);
  // 时长与视口不重置：音频时长以 player 元数据为准（不变）；
  // 仅无音频时按最后一段更新兜底估计
  if (!isFinite(player.duration) || player.duration <= 0) {
    const segs = state.segments;
    state.duration = segs.length ? segs[segs.length - 1].end : 0;
    if (state.viewEnd <= state.viewStart) {
      state.viewStart = 0;
      state.viewEnd = state.duration;
    }
  }
  uncertainToggleWrap.hidden = !uncertainEnabled();
  buildSpeakerChips(); // 内部保留旧可见性，新说话人默认可见
  renderTranscript();  // 内部仅在已贴底时跟随滚动
  statusEl.textContent = `${uri}: ${state.segments.length} 段`;
}

async function loadServerSession(uri) {
  const select = document.getElementById("sessionSelect");
  const opt = select.querySelector(`option[value="${CSS.escape(uri)}"]`);
  statusEl.textContent = `加载 ${uri} …`;
  state.speakersInfo = null; // 切换会话时清空旧 sidecar，防止串台
  await fetchSpeakersInfo(uri, /* apply= */ false);
  const source = wantedSource(opt);
  const data = await fetchSourceSegments(uri, source);
  const audioUrl = opt && opt.dataset.audioUrl;
  setLiveBadge(opt && opt.dataset.live === "1");
  state.source = source;
  state.hasSegments = opt?.dataset.segments === "1";
  state.lastRaw = data ? data.raw : "";
  setSession(uri, data ? data.segments : [], audioUrl || null);
  // 已完成的运行打开即带转写：首次加载就贴一次文本叠加层
  if (opt?.dataset.transcript === "1" && state.source !== "transcript") {
    await pollTranscriptOverlay(uri);
  }
}

function setSession(uri, segments, audioUrl) {
  state.uri = uri;
  state.segments = segments;
  state.overlapFlags = computeOverlapFlags(segments);
  state.activeIdx = -1;
  state.peaks = null; // 切换会话时清空旧波形（新音频的波形由 loadedmetadata 重建）
  state.speakerVisible = new Map(); // 切换会话时重置说话人可见性
  state.textMap = new Map();        // 切换会话时清空转写叠加层，防止串台
  state.lastTranscriptRaw = "";
  uncertainToggleWrap.hidden = !uncertainEnabled();
  buildSpeakerChips();
  renderTranscript();
  scrollTranscriptToBottom(); // 加载/切换会话后定位到最新段
  segPlayer.pause();
  segPlayer.removeAttribute("src");
  if (audioUrl) {
    player.src = audioUrl;
  } else {
    player.removeAttribute("src");
    player.load();
  }
  statusEl.textContent = `${uri}: ${segments.length} 段` +
    (audioUrl ? "" : "（无音频，仅时间线；音频时长按最后一段估计）");
  // 时长以音频为准；无音频时用最后一段的 end 兜底
  const lastEnd = segments.length ? segments[segments.length - 1].end : 0;
  state.duration = player.duration && isFinite(player.duration) ? player.duration : lastEnd;
  fitView();
}

// ---------- 静态模式：本地文件 ----------
document.getElementById("audioFile").onchange = e => {
  const file = e.target.files[0];
  if (file) player.src = URL.createObjectURL(file);
};
document.getElementById("transcriptFile").onchange = async e => {
  const file = e.target.files[0];
  if (!file) return;
  const text = await file.text();
  let segments, uri;
  if (/\.rttm$/i.test(file.name)) {
    // refined RTTM：SPEAKER uri 1 start dur <NA> <NA> spk <NA> <NA>
    segments = text.split("\n")
      .filter(l => l.startsWith("SPEAKER"))
      .map(l => {
        const p = l.trim().split(/\s+/);
        const start = parseFloat(p[3]);
        return { speaker_id: parseInt(p[7], 10), start,
                 end: start + parseFloat(p[4]), text: null };
      })
      .sort((a, b) => a.start - b.start || a.end - b.end);
    uri = file.name.replace(/\.refined\.rttm$/i, "").replace(/\.rttm$/i, "");
    state.source = "rttm";
  } else {
    segments = text.split("\n").filter(l => l.trim()).map(l => {
      const o = JSON.parse(l);
      return { speaker_id: o.speaker_id, start: o.start, end: o.end, text: o.text };
    }).sort((a, b) => a.start - b.start || a.end - b.end);
    uri = file.name.replace(/\.transcript\.jsonl$/, "");
    state.source = "transcript";
  }
  state.segFiles = new Map(); // 静态模式不支持分离段播放
  state.hasSegments = false;
  setSession(uri, segments, null);
};

player.addEventListener("loadedmetadata", () => {
  if (isFinite(player.duration) && player.duration > 0) {
    state.duration = player.duration;
    if (state.viewEnd === 0 || state.viewEnd > state.duration) fitView();
    buildWaveform();
  }
});

// ---------- 波形 ----------
const PEAK_BUCKETS = 8192;
// 全精度解码的时长上限：1h 44.1kHz 立体声解码后约 1.2GB Float32，会打爆
// 标签页；超长音频改用低采样率抽稀解码，只画概览波形
const WAVE_FULL_DECODE_MAX = 600;
const WAVE_MAX_SAMPLES = 1 << 23; // 抽稀解码的采样数上限（8M 采样 ≈ 32MB Float32）

// 单例 AudioContext：切换会话复用不重建（此前每次 loadedmetadata 新建且
// 从不 close，会累积泄漏）；suspended（自动播放策略）时 resume
let audioCtx = null;
function getAudioContext() {
  if (!audioCtx) audioCtx = new AudioContext();
  if (audioCtx.state === "suspended") audioCtx.resume().catch(() => {});
  return audioCtx;
}

async function buildWaveform() {
  state.peaks = null;
  const src = player.src;
  try {
    const resp = await fetch(src);
    let buf = await resp.arrayBuffer();
    // 下载/解码期间已切换会话：丢弃结果，上一个音频的大 buffer 随作用域释放（允许 GC）
    if (player.src !== src) return;
    const duration = player.duration;
    let audio;
    if (duration <= WAVE_FULL_DECODE_MAX) {
      audio = await getAudioContext().decodeAudioData(buf);
    } else {
      // OfflineAudioContext 按较低采样率解码（解码即抽稀），用完即抛由 GC 回收
      const rate = Math.min(44100, Math.max(3000, Math.floor(WAVE_MAX_SAMPLES / duration)));
      audio = await new OfflineAudioContext(1, Math.ceil(duration * rate), rate)
        .decodeAudioData(buf);
    }
    buf = null; // 解除大 arrayBuffer 引用，允许 GC
    const data = audio.getChannelData(0);
    const per = Math.max(1, Math.floor(data.length / PEAK_BUCKETS));
    const peaks = new Float32Array(PEAK_BUCKETS * 2);
    for (let b = 0; b < PEAK_BUCKETS; b++) {
      let lo = 0, hi = 0;
      const from = b * per, to = Math.min(data.length, from + per);
      for (let i = from; i < to; i++) {
        if (data[i] < lo) lo = data[i];
        if (data[i] > hi) hi = data[i];
      }
      peaks[b * 2] = lo; peaks[b * 2 + 1] = hi;
    }
    if (player.src !== src) return; // 解码期间已切换会话，丢弃峰值结果
    state.peaks = { peaks, buckets: PEAK_BUCKETS, duration: audio.duration };
  } catch (err) {
    console.warn("波形解码失败，仅画时间线", err);
  }
}

// ---------- 视图（缩放/平移） ----------
function fitView() {
  state.viewStart = 0;
  state.viewEnd = state.duration || 1;
}
document.getElementById("fitBtn").onclick = fitView;

const t2x = t => (t - state.viewStart) / (state.viewEnd - state.viewStart) * canvas.width;
const x2t = x => state.viewStart + x / canvas.width * (state.viewEnd - state.viewStart);

canvas.addEventListener("wheel", e => {
  e.preventDefault();
  if (!state.duration) return;
  const anchor = x2t(e.offsetX * devicePixelRatio);
  const factor = e.deltaY > 0 ? 1.25 : 0.8;
  let span = (state.viewEnd - state.viewStart) * factor;
  span = Math.min(Math.max(span, 0.05), state.duration);
  const ratio = (anchor - state.viewStart) / (state.viewEnd - state.viewStart);
  state.viewStart = Math.max(0, Math.min(anchor - span * ratio, state.duration - span));
  state.viewEnd = state.viewStart + span;
}, { passive: false });

let drag = null;
canvas.addEventListener("mousedown", e => {
  drag = { x: e.clientX, vs: state.viewStart, ve: state.viewEnd, moved: false };
});
window.addEventListener("mousemove", e => {
  if (!drag) return;
  const dx = e.clientX - drag.x;
  if (Math.abs(dx) > 3) drag.moved = true;
  if (drag.moved && state.duration) {
    const dt = dx / canvas.clientWidth * (drag.ve - drag.vs);
    let vs = drag.vs - dt;
    vs = Math.max(0, Math.min(vs, state.duration - (drag.ve - drag.vs)));
    state.viewStart = vs; state.viewEnd = vs + (drag.ve - drag.vs);
  }
});
window.addEventListener("mouseup", e => {
  if (drag && !drag.moved && e.target === canvas) {
    // 单击：seek（优先命中段块，其余位置直接按时间 seek）
    player.currentTime = x2t(e.offsetX * devicePixelRatio);
    player.play().catch(() => {});
  }
  drag = null;
});

canvas.addEventListener("mousemove", e => {
  const t = x2t(e.offsetX * devicePixelRatio);
  const idx = state.segments.findIndex(s =>
    state.speakerVisible.get(effId(s.speaker_id)) !== false && s.start <= t && t <= s.end);
  if (idx >= 0) {
    const seg = state.segments[idx];
    tooltip.style.display = "block";
    tooltip.style.left = (e.clientX + 12) + "px";
    tooltip.style.top = (e.clientY + 12) + "px";
    tooltip.textContent = `[${fmtTime(seg.start)} - ${fmtTime(seg.end)}] ${spkLabel(seg.speaker_id)}` +
      (state.overlapFlags[idx] ? "（与其他说话人重叠）" : "") +
      (seg.text ? `: ${seg.text}` : "");
  } else {
    tooltip.style.display = "none";
  }
});
canvas.addEventListener("mouseleave", () => tooltip.style.display = "none");

// ---------- 渲染 ----------
function resizeCanvas() {
  const w = canvas.clientWidth, h = canvas.clientHeight;
  if (canvas.width !== w * devicePixelRatio || canvas.height !== h * devicePixelRatio) {
    canvas.width = w * devicePixelRatio;
    canvas.height = h * devicePixelRatio;
  }
}

function pickTickStep(span) {
  const steps = [0.1,0.2,0.5,1,2,5,10,15,30,60,120,300,600,1800,3600];
  for (const s of steps) {
    if (s / span * canvas.width >= 90 * devicePixelRatio) return s;
  }
  return 7200;
}

function draw() {
  resizeCanvas();
  const W = canvas.width, H = canvas.height;
  const dpr = devicePixelRatio;
  ctx.clearRect(0, 0, W, H);
  if (!(state.viewEnd > state.viewStart)) return; // 尚未加载会话

  const rulerH = 22 * dpr;
  const waveH = Math.floor((H - rulerH) * 0.45);
  const segTop = rulerH + waveH + 4 * dpr;
  const segH = H - segTop - 4 * dpr;

  // 波形
  ctx.fillStyle = "#565a68";
  if (state.peaks) {
    const { peaks, buckets, duration } = state.peaks;
    const mid = rulerH + waveH / 2;
    for (let x = 0; x < W; x++) {
      const tA = x2t(x), tB = x2t(x + 1);
      const i0 = Math.max(0, Math.floor(tA / duration * buckets));
      const i1 = Math.min(buckets - 1, Math.ceil(tB / duration * buckets));
      let lo = 0, hi = 0;
      for (let i = i0; i <= i1; i++) {
        if (peaks[i*2] < lo) lo = peaks[i*2];
        if (peaks[i*2+1] > hi) hi = peaks[i*2+1];
      }
      ctx.fillRect(x, mid - hi * waveH / 2, 1, Math.max(1, (hi - lo) * waveH / 2));
    }
  }

  // 段块（时间线轨道；颜色/轨道/可见性按合并后的有效 id）
  const laneH = Math.max(8 * dpr, Math.min(16 * dpr, segH / 3));
  const laneGap = 2 * dpr;
  const lanes = new Map(); // 有效 speaker id -> lane idx
  for (let i = 0; i < state.segments.length; i++) {
    const s = state.segments[i];
    const eff = effId(s.speaker_id);
    if (state.speakerVisible.get(eff) === false) continue;
    if (s.end < state.viewStart || s.start > state.viewEnd) continue;
    if (!lanes.has(eff)) lanes.set(eff, lanes.size);
    const lane = lanes.get(eff) % 3;
    const x0 = Math.max(0, t2x(s.start)), x1 = Math.min(W, t2x(s.end));
    const y0 = segTop + lane * (laneH + laneGap);
    ctx.fillStyle = spkColor(eff);
    ctx.globalAlpha = showUncertain(eff) ? 0.4 : 0.85; // uncertain 段淡化
    ctx.fillRect(x0, y0, Math.max(2, x1 - x0), laneH);
    ctx.globalAlpha = 1;
    if (state.overlapFlags[i]) { // 重叠段：红色描边提示
      ctx.strokeStyle = "#f97583";
      ctx.lineWidth = 2 * dpr;
      ctx.strokeRect(x0 + 1, y0 + 1, Math.max(2, x1 - x0) - 2, laneH - 2);
      ctx.lineWidth = 1;
    }
  }

  // 刻度尺
  ctx.fillStyle = "#9a9ba5";
  ctx.font = `${10 * dpr}px ui-monospace, monospace`;
  const span = state.viewEnd - state.viewStart;
  const step = pickTickStep(span);
  ctx.strokeStyle = "#3a3b44";
  ctx.beginPath();
  ctx.moveTo(0, rulerH - 0.5); ctx.lineTo(W, rulerH - 0.5); ctx.stroke();
  for (let t = Math.ceil(state.viewStart / step) * step; t <= state.viewEnd; t += step) {
    const x = t2x(t);
    ctx.fillText(fmtTime(t), x + 2 * dpr, rulerH - 8 * dpr);
    ctx.beginPath(); ctx.moveTo(x, rulerH - 5 * dpr); ctx.lineTo(x, rulerH); ctx.stroke();
  }

  // 播放头
  if (state.duration) {
    const t = player.currentTime;
    if (t >= state.viewStart && t <= state.viewEnd) {
      const x = t2x(t);
      ctx.strokeStyle = "#fff";
      ctx.lineWidth = 1.5 * dpr;
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke();
      ctx.lineWidth = 1;
    }
  }
}

// ---------- 说话人筛选 ----------
function buildSpeakerChips() {
  // chip 按合并后的有效 id 组织：被并说话人的段归入幸存者
  const ids = [...new Set(state.segments.map(s => effId(s.speaker_id)))]
    .sort((a, b) => a - b);
  // 保留已有可见性设置（增量刷新时新说话人默认可见，旧设置不丢）
  state.speakerVisible = new Map(
    ids.map(id => [id, state.speakerVisible.get(id) !== false])
  );
  const info = spkInfoMap();
  const box = document.getElementById("speakers");
  box.innerHTML = "";
  for (const id of ids) {
    const chip = document.createElement("span");
    chip.className = "spk-chip";
    if (state.speakerVisible.get(id) === false) chip.classList.add("off");
    if (showUncertain(id)) chip.classList.add("uncertain");
    chip.innerHTML =
      `<span class="spk-dot" style="background:${spkColor(id)}"></span>${spkLabel(id)}`;
    const duration = info.get(id)?.duration;
    if (duration != null) chip.title = `发声时长 ${duration.toFixed(1)}s`;
    chip.onclick = () => {
      state.speakerVisible.set(id, !state.speakerVisible.get(id));
      chip.classList.toggle("off");
      renderTranscript();
    };
    box.appendChild(chip);
  }
  // 被并掉的说话人：合并事件提示chip（删除线），仅展示不可筛选
  const mergedIds = [...new Set(state.segments.map(s => s.speaker_id))]
    .filter(id => effId(id) !== id)
    .sort((a, b) => a - b);
  for (const id of mergedIds) {
    const chip = document.createElement("span");
    chip.className = "spk-chip merged";
    chip.innerHTML =
      `<span class="spk-dot" style="background:${spkColor(id)}"></span>${spkLabel(id)}`;
    chip.title = "合并事件：该说话人已并入幸存者，段按幸存者颜色展示";
    box.appendChild(chip);
  }
}

// ---------- 段列表 ----------
// 滚动条贴底时新段落自动下翻；用户上翻阅读历史时保持位置不动
function isTranscriptAtBottom() {
  return transcriptEl.scrollHeight - transcriptEl.scrollTop - transcriptEl.clientHeight < 40;
}
function scrollTranscriptToBottom() {
  transcriptEl.scrollTop = transcriptEl.scrollHeight;
}

function appendTranscriptRow(s, i) {
  const eff = effId(s.speaker_id);
  if (state.speakerVisible.get(eff) === false) return;
  const row = document.createElement("div");
  row.className = "seg"; row.dataset.idx = i;
  row.innerHTML =
    `<span class="time">${fmtTime(s.start)} - ${fmtTime(s.end)}</span>` +
    // uncertain 说话人标签用灰色（--dim），不再按调色板着色
    `<span class="spk" style="color:${showUncertain(eff) ? "var(--dim)" : spkColor(eff)}">${spkLabel(s.speaker_id)}</span>` +
    (state.overlapFlags[i] ? `<span class="overlap-badge">重叠</span>` : "") +
    `<span class="text"></span>`;
  row.querySelector(".text").textContent =
    s.text || state.textMap.get(segKey(s)) || "";
  // stage>=2：有对应分离段音频时提供独立播放（不打断主播放器）
  const file = segFileOf(s);
  if (file) {
    const btn = document.createElement("button");
    btn.className = "segplay";
    btn.textContent = "▶ 分离";
    btn.title = "播放该段的分离音频（单说话人音轨）";
    btn.onclick = ev => { ev.stopPropagation(); playSegFile(file, btn); };
    row.insertBefore(btn, row.querySelector(".text"));
  }
  row.onclick = () => { player.currentTime = s.start + 1e-3; player.play().catch(() => {}); };
  transcriptEl.appendChild(row);
}

function renderTranscript() {
  const stickToBottom = isTranscriptAtBottom();
  transcriptEl.innerHTML = "";
  state.segments.forEach((s, i) => appendTranscriptRow(s, i));
  if (stickToBottom) scrollTranscriptToBottom();
}

function syncActiveRow() {
  const t = player.currentTime;
  let idx = -1;
  for (let i = 0; i < state.segments.length; i++) {
    const s = state.segments[i];
    if (s.start <= t && t <= s.end) { idx = i; break; }
    if (s.start > t) break;
  }
  if (idx === state.activeIdx) return;
  state.activeIdx = idx;
  transcriptEl.querySelectorAll(".seg.active").forEach(el => el.classList.remove("active"));
  if (idx >= 0) {
    const row = transcriptEl.querySelector(`.seg[data-idx="${idx}"]`);
    if (row) {
      row.classList.add("active");
      row.scrollIntoView({ block: "nearest", behavior: "smooth" });
    }
  }
}

// 播放时播放头接近视口右缘 → 视图跟随
function followPlayhead() {
  if (player.paused || !state.duration) return;
  const span = state.viewEnd - state.viewStart;
  if (player.currentTime > state.viewEnd - span * 0.1) {
    state.viewStart = Math.min(player.currentTime - span * 0.1, state.duration - span);
    state.viewStart = Math.max(0, state.viewStart);
    state.viewEnd = state.viewStart + span;
  }
}

// ---------- 主循环 ----------
function tick() {
  followPlayhead();
  syncActiveRow();
  draw();
  requestAnimationFrame(tick);
}

// ---------- 启动 ----------
(async function init() {
  const ok = await loadServerSessions();
  if (!ok) {
    statusEl.textContent = "静态模式：请手动选择音频与 transcript.jsonl / refined.rttm 文件";
  } else if (!startEventStream()) {
    setTimeout(pollUpdate, POLL_INTERVAL); // 浏览器不支持 EventSource：轮询兜底
  }
  tick();
})();
