(function () {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const CONFIG = window.PETS || { models: {}, env: {}, maxVideoMb: 200, maxUploadMb: 40, samples: {}, voices: {} };
  const MAX = 12;
  const VIDEO = /\.(mp4|mov|m4v|webm|mkv|avi)$/i;
  // {id, time, title, thought, punch, best, search, media: [], chosen, own, ownUrl, state, problem}
  let list = [];
  let nextId = 0;
  let pollTimer = null;
  const queue = [];
  let searching = false;

  function showError(msg) {
    $("form-error").textContent = msg;
    $("form-error").hidden = !msg;
  }

  // ------------------------------------------------------------------ keys (this browser only, shared with Travel)
  const store = {
    get() { try { return JSON.parse(localStorage.getItem("travel-settings") || "{}"); } catch (e) { return {}; } },
    set(value) { try { localStorage.setItem("travel-settings", JSON.stringify(value)); } catch (e) { /* private mode */ } },
  };
  const saved = store.get();
  saved.keys = saved.keys || {};
  saved.models = saved.models || {};

  function provider() { return $("provider").value; }

  function showProvider() {
    const p = provider();
    $("ai-key").value = saved.keys[p] || "";
    $("model").value = saved.models[p] || "";
    $("model").placeholder = CONFIG.models[p] || "";
    $("key-note").textContent = CONFIG.env[p] && !saved.keys[p]
      ? "using the key set on the server" : "saved in this browser only";
  }

  if (saved.provider) $("provider").value = saved.provider;
  $("pexels-key").value = saved.keys.pexels || "";
  $("pixabay-key").value = saved.keys.pixabay || "";
  showProvider();
  $("provider").addEventListener("change", () => { saved.provider = provider(); store.set(saved); showProvider(); });
  $("ai-key").addEventListener("change", () => { saved.keys[provider()] = $("ai-key").value.trim(); store.set(saved); showProvider(); });
  $("model").addEventListener("change", () => { saved.models[provider()] = $("model").value.trim(); store.set(saved); });
  for (const name of ["pexels", "pixabay"]) {
    $(`${name}-key`).addEventListener("change", () => { saved.keys[name] = $(`${name}-key`).value.trim(); store.set(saved); });
  }

  function libraryKeys() {
    return { pexels: $("pexels-key").value.trim(), pixabay: $("pixabay-key").value.trim() };
  }

  // ------------------------------------------------------------------ a whole diary in (sample or AI)
  function fill(diary) {
    if (diary.pet) $("pet").value = diary.pet;
    if (diary.kind) $("kind").value = diary.kind;
    if (diary.title) $("title").value = diary.title;
    if (diary.hook) $("hook").value = diary.hook;
    if (diary.question) $("question").value = diary.question;
    $("guess").value = (diary.guess || []).join("\n");
    $("answer").value = diary.answer || "";
    list.forEach((m) => { if (m.ownUrl) URL.revokeObjectURL(m.ownUrl); });
    list = (diary.moments || []).slice(0, MAX).map((m) => newMoment({
      time: m.time || "", title: m.title || "", thought: m.thought || "", punch: m.punch || "",
      best: !!m.best, search: m.search || m.title || "" }));
    show();
    list.forEach((m) => find(m));
  }

  document.querySelectorAll("[data-sample]").forEach((b) => b.addEventListener("click", () => {
    const diary = CONFIG.samples[b.dataset.sample];
    if (!diary) return;
    fill(JSON.parse(JSON.stringify(diary)));
    $("sample-info").textContent = `${diary.pet}'s day is loaded. Pictures are being looked up; swap in your own pet's clips if you have them.`;
  }));

  $("write").addEventListener("click", async () => {
    const pet = $("pet").value.trim();
    if (!pet) { $("write-info").textContent = "Give the pet a name first."; return; }
    $("write").disabled = true;
    $("write-info").textContent = "Writing… this can take up to a minute.";
    try {
      const res = await fetch("/api/pets/write", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ provider: provider(), key: $("ai-key").value.trim(), model: $("model").value.trim(),
                               pet, kind: $("kind").value, idea: $("idea").value.trim(), count: Number($("count").value) }),
      });
      const body = await res.json().catch(() => ({ error: `Server error (${res.status}).` }));
      if (!res.ok) throw new Error(body.error || `Server error (${res.status}).`);
      fill(Object.assign({ pet, kind: $("kind").value }, body));
      $("write-info").textContent = `${list.length} moments written. Read them through, then pick the pictures.`;
    } catch (err) {
      $("write-info").textContent = err.message;
    } finally {
      $("write").disabled = false;
    }
  });

  function newMoment(fields) {
    return Object.assign({ id: nextId++, time: "", title: "", thought: "", punch: "", best: false, search: "",
                           media: [], chosen: null, own: null, ownUrl: null, state: "", problem: "" }, fields);
  }

  $("add").addEventListener("click", () => {
    if (list.length >= MAX) return;
    list.push(newMoment({}));
    show();
    const rows = document.querySelectorAll("#moment-list .time");
    if (rows.length) rows[rows.length - 1].focus();
  });

  $("find-all").addEventListener("click", () => {
    list.forEach((m) => { if (!m.own && (m.search || m.title)) find(m); });
  });

  // ------------------------------------------------------------------ pictures
  // One search at a time: Openverse allows few requests from one computer.
  function find(moment) {
    if (queue.includes(moment) || !(moment.search || moment.title).trim()) return;
    moment.state = "waiting";
    queue.push(moment);
    update(moment);
    next();
  }

  async function next() {
    if (searching || !queue.length) return;
    searching = true;
    const moment = queue.shift();
    if (list.includes(moment)) {
      moment.state = "searching";
      moment.problem = "";
      update(moment);
      try {
        const res = await fetch("/api/travel/media", {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ q: (moment.search || moment.title).trim(), keys: libraryKeys() }),
        });
        const body = await res.json().catch(() => ({ error: `Server error (${res.status}).` }));
        if (!res.ok) throw new Error(body.error || `Server error (${res.status}).`);
        moment.media = body.media;
        moment.chosen = body.media.length ? 0 : null;
        moment.problem = body.media.length ? (body.problems || []).join(" · ")
          : "Nothing found. Try other words, or upload your own photo or clip.";
      } catch (err) {
        moment.problem = err.message;
      }
      moment.state = "";
      update(moment);
    }
    searching = false;
    next();
  }

  function preview(m) {
    if (m.own) return { url: m.ownUrl, video: VIDEO.test(m.own.name) };
    if (m.chosen != null && m.media[m.chosen]) {
      const media = m.media[m.chosen];
      return { url: media.thumb, video: media.kind === "video", credit: `${media.provider} · ${media.author || "unknown"}` };
    }
    return null;
  }

  // ------------------------------------------------------------------ the list
  function show() {
    const box = $("moment-list");
    box.innerHTML = "";
    list.forEach((m, i) => box.appendChild(row(m, i)));
    $("export-card").hidden = !list.length;
    $("find-all").hidden = !list.length;
    $("list-info").textContent = list.length ? "Mark one moment as the big moment: it opens the video and is guessed." : "";
    $("generate").textContent = list.length ? `Draw ${list.length} moment${list.length > 1 ? "s" : ""}` : "Draw the diary";
  }

  function update(moment) {
    const old = document.querySelector(`#moment-list [data-id="${moment.id}"]`);
    if (old) old.replaceWith(row(moment, list.indexOf(moment)));
  }

  function row(m, i) {
    const el = document.createElement("div");
    el.className = "animal place";
    el.dataset.id = m.id;
    el.innerHTML = `
      <div class="pic-col">
        <span class="pic"></span>
        <span class="credit"></span>
      </div>
      <div class="fields">
        <div class="pair">
          <label>Time <span class="unit">7:00 AM, noon…</span> <input type="text" class="time" maxlength="12"></label>
          <label>Title <input type="text" class="title" maxlength="40"></label>
        </div>
        <label>The pet thinks <span class="unit">while the moment is drawn</span>
          <textarea class="thought" rows="2" maxlength="160" placeholder="The leash! The leash means outside!"></textarea></label>
        <label>Punchline <span class="unit">when the real clip plays · optional</span>
          <input type="text" class="punch" maxlength="90" placeholder="I smelled forty-two trees today."></label>
        <label class="check"><input type="radio" name="best" class="best"> <span>The big moment (cold open and "What happens next?")</span></label>
        <div class="picker-bar">
          <input type="text" class="search" maxlength="100" placeholder="words to search a photo or clip with" aria-label="Search words">
          <button type="button" class="small find">Find pictures</button>
          <span class="hint status" aria-live="polite"></span>
        </div>
        <div class="picker"></div>
        <label>Your own photo or clip <span class="unit">best: your pet</span>
          <input type="file" class="own" accept="image/*,video/*,.jpg,.jpeg,.png,.webp,.mp4,.mov,.m4v,.webm"></label>
      </div>
      <span class="tools">
        <span class="num">#${i + 1}</span>
        <button type="button" class="small" data-move="-1" aria-label="Move up">▲</button>
        <button type="button" class="small" data-move="1" aria-label="Move down">▼</button>
        <button type="button" class="small" data-remove aria-label="Remove">✕</button>
      </span>`;
    const look = preview(m);
    const pic = el.querySelector(".pic");
    if (look) {
      const media = document.createElement(look.video && m.own ? "video" : "img");
      media.src = look.url;
      if (media.tagName === "VIDEO") { media.muted = true; media.preload = "metadata"; } else media.alt = m.title;
      pic.appendChild(media);
      if (look.video) pic.insertAdjacentHTML("beforeend", '<span class="badge">▶ clip</span>');
      if (look.credit) el.querySelector(".credit").textContent = look.credit;
    } else {
      pic.textContent = m.state ? "searching…" : "no picture yet";
    }
    for (const field of ["time", "title", "thought", "punch", "search"]) {
      const input = el.querySelector(`.${field}`);
      input.value = m[field] || "";
      input.addEventListener("input", () => { m[field] = input.value; });
    }
    const best = el.querySelector(".best");
    best.checked = m.best;
    best.addEventListener("change", () => { list.forEach((x) => { x.best = x === m; }); });
    el.querySelector(".search").placeholder = m.title || "words to search a photo or clip with";
    el.querySelector(".find").addEventListener("click", () => {
      if (!(m.search || m.title).trim()) { m.problem = "Type a title or search words first."; update(m); return; }
      find(m);
    });
    el.querySelector(".find").disabled = !!m.state;
    el.querySelector(".status").textContent = m.state === "waiting" ? "waiting…"
      : m.state === "searching" ? "searching… (up to a minute)" : m.problem;
    const picker = el.querySelector(".picker");
    m.media.forEach((media, j) => {
      const b = document.createElement("button");
      b.type = "button";
      b.className = "thumb" + (j === m.chosen && !m.own ? " selected" : "");
      b.title = `${media.title} · ${media.author || "unknown"} · ${media.licence} (${media.provider})`;
      b.setAttribute("aria-label", `${media.kind === "video" ? "Video" : "Photo"}: ${b.title}`);
      b.innerHTML = `<img alt="" loading="lazy"><span class="badge">${media.kind === "video" ? `▶ ${Math.round(media.duration)} s · ` : ""}${media.provider}</span>`;
      b.querySelector("img").src = media.thumb;
      b.addEventListener("click", () => { m.chosen = j; if (m.own) { URL.revokeObjectURL(m.ownUrl); m.own = null; m.ownUrl = null; } update(m); });
      picker.appendChild(b);
    });
    const own = el.querySelector(".own");
    own.addEventListener("change", () => {
      const file = own.files[0];
      if (!file) return;
      const limit = (VIDEO.test(file.name) ? CONFIG.maxVideoMb : CONFIG.maxUploadMb) * 1048576;
      if (file.size > limit) { m.problem = `That file is larger than ${Math.round(limit / 1048576)} MB.`; update(m); return; }
      if (m.ownUrl) URL.revokeObjectURL(m.ownUrl);
      m.own = file;
      m.ownUrl = URL.createObjectURL(file);
      update(m);
    });
    if (m.own) own.insertAdjacentText("afterend", ` (${m.own.name})`);
    el.querySelectorAll("[data-move]").forEach((b) => b.addEventListener("click", () => {
      const j = i + Number(b.dataset.move);
      if (j < 0 || j >= list.length) return;
      [list[i], list[j]] = [list[j], list[i]];
      show();
    }));
    el.querySelector("[data-remove]").addEventListener("click", () => {
      if (m.ownUrl) URL.revokeObjectURL(m.ownUrl);
      list.splice(i, 1);
      show();
    });
    return el;
  }

  // ------------------------------------------------------------------ render
  $("generate").addEventListener("click", async () => {
    showError("");
    const pet = $("pet").value.trim();
    if (!pet) return showError("Give the pet a name.");
    if (!list.length) return showError("Add at least one moment.");
    const empty = list.findIndex((m) => !m.title.trim() && !m.thought.trim());
    if (empty >= 0) return showError(`Moment ${empty + 1} needs a title or a thought.`);
    const bare = list.findIndex((m) => !m.own && (m.chosen == null || !m.media[m.chosen]));
    if (bare >= 0) return showError(`Choose a picture for moment ${bare + 1}, or upload your own.`);
    const seconds = parseFloat($("seconds").value);
    if (!(seconds >= 5 && seconds <= 30)) return showError("Drawing time must be between 5 and 30 seconds.");
    const reveal = parseFloat($("reveal").value);
    if (!(reveal >= 1.5 && reveal <= 10)) return showError("The real clip must last between 1.5 and 10 seconds.");
    const guess = $("guess").value.split("\n").map((g) => g.trim()).filter(Boolean);
    if ($("guess-on").checked && guess.length && !guess.includes($("answer").value.trim()) && guess.length >= 4) {
      return showError("The right answer must be one of the answers.");
    }

    const form = new FormData();
    const moments = list.map((m) => {
      if (m.own) form.append(`picture_${m.id}`, m.own, m.own.name);
      return { time: m.time.trim(), title: m.title.trim(), thought: m.thought.trim(), punch: m.punch.trim(),
               best: m.best, media: m.own ? null : m.media[m.chosen], upload: m.own ? `picture_${m.id}` : null };
    });
    const music = $("music").files[0];
    if (music) form.append("music", music, music.name);
    form.append("data", JSON.stringify({
      pet, kind: $("kind").value, voice: $("voice").value, pitch: parseFloat($("pitch").value) || 0,
      moments, title: $("title").value.trim(), hook: $("hook").value.trim(), guess, answer: $("answer").value.trim(),
      question: $("question").value.trim(), resolution: $("resolution").value, seconds, reveal, tune: $("tune").value,
      cold_open: $("cold-open").checked, guess_on: $("guess-on").checked, outro: $("outro").checked,
      sound_effects: $("sfx").checked,
    }));

    $("generate").disabled = true;
    $("result").hidden = true;
    setStatus({ state: "queued", progress: 0, message: "Uploading…" });
    try {
      const res = await fetch("/api/pets/render", { method: "POST", body: form });
      const body = await res.json().catch(() => ({ error: `Server error (${res.status}).` }));
      if (!res.ok) throw new Error(body.error || `Server error (${res.status}).`);
      setStatus(body);
      poll(body.id);
    } catch (err) {
      $("status").hidden = true;
      $("generate").disabled = false;
      showError(err.message);
    }
  });

  function poll(id) {
    clearTimeout(pollTimer);
    pollTimer = setTimeout(async () => {
      try {
        const res = await fetch(`/api/jobs/${id}`, { cache: "no-store" });
        const job = await res.json();
        if (!res.ok) throw new Error(job.error || "Lost track of the job.");
        setStatus(job);
        if (job.state === "done") return showResult(job);
        if (job.state === "error") { $("generate").disabled = false; return; }
        poll(id);
      } catch (err) {
        setStatus({ state: "error", progress: 0, error: err.message });
        $("generate").disabled = false;
      }
    }, 800);
  }

  function setStatus(job) {
    $("status").hidden = false;
    const pct = Math.round((job.progress || 0) * 100);
    const isError = job.state === "error";
    $("status").classList.toggle("error", isError);
    $("status-title").textContent = isError ? "Drawing failed" : job.state === "done" ? "Finished"
      : job.state === "queued" ? "Queued" : "Drawing";
    $("status-detail").textContent = isError ? job.error : job.message || "";
    $("bar").style.width = `${isError ? 100 : pct}%`;
    $("progressbar").setAttribute("aria-valuenow", String(pct));
    $("pct").textContent = isError ? "" : `${pct}%`;
    $("eta").textContent = job.eta != null ? `about ${Math.ceil(job.eta)} s left`
      : job.elapsed ? `${Math.round(job.elapsed)} s elapsed` : "";
  }

  function showResult(job) {
    $("generate").disabled = false;
    const info = job.info || {};
    $("video").src = `/api/jobs/${job.id}/video?v=${Date.now()}`;
    $("download").href = `/api/jobs/${job.id}/download`;
    $("image").href = `/api/jobs/${job.id}/image`;
    $("credits").href = `/api/jobs/${job.id}/credits`;
    $("credits").hidden = !job.has_credits;
    const size = info.file_size ? ` · ${(info.file_size / 1048576).toFixed(1)} MB` : "";
    const notes = (job.warnings || []).length ? ` · ${job.warnings.join(" ")}` : "";
    $("result-summary").textContent = `${(info.moments || []).length} moments · ${Math.round(info.duration || 0)} s` +
      `${size} · made in ${Math.round(info.render_seconds || job.elapsed)} s${notes}`;
    $("result").hidden = false;
    $("result").scrollIntoView({ behavior: "smooth", block: "start" });
  }

  show();
})();
