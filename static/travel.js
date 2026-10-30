(function () {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const CONFIG = window.TRAVEL || { models: {}, env: {}, maxVideoMb: 200, maxUploadMb: 40 };
  const MAX = 20;
  const VIDEO = /\.(mp4|mov|m4v|webm|mkv|avi)$/i;
  // {id, name, location, country, lat, lon, facts, search, media: [], chosen, own, ownUrl, voice, state, problem}
  let list = [];
  let nextId = 0;
  let pollTimer = null;
  const queue = [];
  let searching = false;

  function showError(msg) {
    $("form-error").textContent = msg;
    $("form-error").hidden = !msg;
  }

  // ------------------------------------------------------------------ keys (this browser only)
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

  // ------------------------------------------------------------------ AI script
  $("write").addEventListener("click", async () => {
    const names = $("names").value.split("\n").map((s) => s.trim()).filter(Boolean);
    const theme = $("theme").value.trim();
    if (!names.length && !theme) {
      $("write-info").textContent = "Type a theme or a list of places first.";
      return;
    }
    $("write").disabled = true;
    $("write-info").textContent = "Writing… this can take up to a minute.";
    try {
      const res = await fetch("/api/travel/write", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ provider: provider(), key: $("ai-key").value.trim(), model: $("model").value.trim(),
                               theme, places: names, count: Number($("count").value),
                               facts: Number($("fact-count").value) }),
      });
      const body = await res.json().catch(() => ({ error: `Server error (${res.status}).` }));
      if (!res.ok) throw new Error(body.error || `Server error (${res.status}).`);
      if (!$("title").value.trim() && body.title) $("title").value = body.title;
      if (!$("hook").value.trim() && body.hook) $("hook").value = body.hook;
      if (body.question) $("question").value = body.question;
      const room = MAX - list.length;
      for (const p of body.places.slice(0, room)) {
        list.push(newPlace({ name: p.name, location: p.location, country: p.country, lat: p.lat, lon: p.lon,
                             facts: (p.facts || []).join("\n"), search: p.search || p.name }));
      }
      $("write-info").textContent = `${Math.min(body.places.length, room)} places written. Check the facts, then pick the pictures.`;
      show();
      list.forEach((p) => { if (!p.media.length && !p.own) find(p); });
    } catch (err) {
      $("write-info").textContent = err.message;
    } finally {
      $("write").disabled = false;
    }
  });

  function newPlace(fields) {
    return Object.assign({ id: nextId++, name: "", location: "", country: "", lat: null, lon: null, facts: "",
                           search: "", media: [], chosen: null, own: null, ownUrl: null, voice: null,
                           state: "", problem: "" }, fields);
  }

  $("add").addEventListener("click", () => {
    if (list.length >= MAX) return;
    list.push(newPlace({}));
    show();
    const rows = document.querySelectorAll("#place-list .name");
    if (rows.length) rows[rows.length - 1].focus();
  });

  $("find-all").addEventListener("click", () => {
    list.forEach((p) => { if (!p.own && (p.search || p.name)) find(p); });
  });

  // ------------------------------------------------------------------ pictures
  // One search at a time: Openverse allows few requests from one computer.
  function find(place) {
    if (queue.includes(place)) return;
    place.state = "waiting";
    queue.push(place);
    update(place);
    next();
  }

  async function next() {
    if (searching || !queue.length) return;
    searching = true;
    const place = queue.shift();
    if (list.includes(place)) {
      place.state = "searching";
      place.problem = "";
      update(place);
      try {
        const res = await fetch("/api/travel/media", {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ q: (place.search || place.name).trim(), keys: libraryKeys() }),
        });
        const body = await res.json().catch(() => ({ error: `Server error (${res.status}).` }));
        if (!res.ok) throw new Error(body.error || `Server error (${res.status}).`);
        place.media = body.media;
        place.chosen = body.media.length ? 0 : null;
        place.problem = body.media.length ? (body.problems || []).join(" · ")
          : "Nothing found. Try other words, or upload your own photo.";
      } catch (err) {
        place.problem = err.message;
      }
      place.state = "";
      update(place);
    }
    searching = false;
    next();
  }

  function preview(place) {
    if (place.own) return { url: place.ownUrl, video: VIDEO.test(place.own.name) };
    if (place.chosen != null && place.media[place.chosen]) {
      const m = place.media[place.chosen];
      return { url: m.thumb, video: m.kind === "video", credit: `${m.provider} · ${m.author || "unknown"}` };
    }
    return null;
  }

  // ------------------------------------------------------------------ the list
  function show() {
    const box = $("place-list");
    box.innerHTML = "";
    list.forEach((p, i) => box.appendChild(row(p, i)));
    $("export-card").hidden = !list.length;
    $("find-all").hidden = !list.length;
    $("generate").textContent = list.length ? `Draw ${list.length} place${list.length > 1 ? "s" : ""}` : "Draw the places";
  }

  function update(place) {
    const old = document.querySelector(`#place-list [data-id="${place.id}"]`);
    if (old) old.replaceWith(row(place, list.indexOf(place)));
  }

  function row(p, i) {
    const el = document.createElement("div");
    el.className = "animal place";
    el.dataset.id = p.id;
    el.innerHTML = `
      <div class="pic-col">
        <span class="pic"></span>
        <span class="credit"></span>
      </div>
      <div class="fields">
        <div class="pair">
          <label>Name <input type="text" class="name" maxlength="60"></label>
          <label>Where <span class="unit">city, country</span> <input type="text" class="location" maxlength="60"></label>
        </div>
        <div class="coords">
          <label>Country <input type="text" class="country" maxlength="40"></label>
          <label>Latitude <input type="number" class="lat" step="any" min="-90" max="90"></label>
          <label>Longitude <input type="number" class="lon" step="any" min="-180" max="180"></label>
        </div>
        <label>Facts <span class="unit">one per line</span>
          <textarea class="facts" rows="4" placeholder="It is about 330 metres tall.&#10;It was built for the 1889 World's Fair."></textarea></label>
        <div class="picker-bar">
          <input type="text" class="search" maxlength="100" placeholder="words to search a photo with" aria-label="Photo search words">
          <button type="button" class="small find">Find pictures</button>
          <span class="hint status" aria-live="polite"></span>
        </div>
        <div class="picker"></div>
        <div class="pair">
          <label>Your own photo or video <span class="unit">optional</span>
            <input type="file" class="own" accept="image/*,video/*,.jpg,.jpeg,.png,.webp,.mp4,.mov,.m4v,.webm"></label>
          <label>Recording <span class="unit">optional, plays from the first stroke</span>
            <input type="file" class="voice" accept="audio/*,.m4a,.mp3,.wav,.aac,.ogg,.flac"></label>
        </div>
      </div>
      <span class="tools">
        <span class="num">#${i + 1}</span>
        <button type="button" class="small" data-move="-1" aria-label="Move up">▲</button>
        <button type="button" class="small" data-move="1" aria-label="Move down">▼</button>
        <button type="button" class="small" data-remove aria-label="Remove">✕</button>
      </span>`;
    const look = preview(p);
    const pic = el.querySelector(".pic");
    if (look) {
      const media = document.createElement(look.video && p.own ? "video" : "img");
      media.src = look.url;
      if (media.tagName === "VIDEO") { media.muted = true; media.preload = "metadata"; } else media.alt = p.name;
      pic.appendChild(media);
      if (look.video) pic.insertAdjacentHTML("beforeend", '<span class="badge">▶ clip</span>');
      if (look.credit) el.querySelector(".credit").textContent = look.credit;
    } else {
      pic.textContent = p.state ? "searching…" : "no picture yet";
    }
    for (const field of ["name", "location", "country", "facts", "search"]) {
      const input = el.querySelector(`.${field}`);
      input.value = p[field] || "";
      input.addEventListener("input", () => { p[field] = input.value; });
    }
    for (const field of ["lat", "lon"]) {
      const input = el.querySelector(`.${field}`);
      input.value = p[field] == null ? "" : p[field];
      input.addEventListener("input", () => { p[field] = input.value === "" ? null : Number(input.value); });
    }
    el.querySelector(".search").placeholder = p.name || "words to search a photo with";
    el.querySelector(".find").addEventListener("click", () => {
      if (!(p.search || p.name).trim()) { p.problem = "Type a name or search words first."; update(p); return; }
      find(p);
    });
    el.querySelector(".find").disabled = !!p.state;
    el.querySelector(".status").textContent = p.state === "waiting" ? "waiting…"
      : p.state === "searching" ? "searching… (up to a minute)" : p.problem;
    const picker = el.querySelector(".picker");
    p.media.forEach((m, j) => {
      const b = document.createElement("button");
      b.type = "button";
      b.className = "thumb" + (j === p.chosen && !p.own ? " selected" : "");
      b.title = `${m.title} · ${m.author || "unknown"} · ${m.licence} (${m.provider})`;
      b.setAttribute("aria-label", `${m.kind === "video" ? "Video" : "Photo"}: ${b.title}`);
      b.innerHTML = `<img alt="" loading="lazy"><span class="badge">${m.kind === "video" ? `▶ ${Math.round(m.duration)} s · ` : ""}${m.provider}</span>`;
      b.querySelector("img").src = m.thumb;
      b.addEventListener("click", () => { p.chosen = j; if (p.own) { URL.revokeObjectURL(p.ownUrl); p.own = null; p.ownUrl = null; } update(p); });
      picker.appendChild(b);
    });
    const own = el.querySelector(".own");
    own.addEventListener("change", () => {
      const file = own.files[0];
      if (!file) return;
      const limit = (VIDEO.test(file.name) ? CONFIG.maxVideoMb : CONFIG.maxUploadMb) * 1048576;
      if (file.size > limit) { p.problem = `That file is larger than ${Math.round(limit / 1048576)} MB.`; update(p); return; }
      if (p.ownUrl) URL.revokeObjectURL(p.ownUrl);
      p.own = file;
      p.ownUrl = URL.createObjectURL(file);
      if (!p.name) p.name = nameFromFile(file.name);
      update(p);
    });
    if (p.own) own.insertAdjacentText("afterend", ` (${p.own.name})`);
    const voice = el.querySelector(".voice");
    voice.addEventListener("change", () => { p.voice = voice.files[0] || null; });
    if (p.voice) voice.insertAdjacentText("afterend", ` (${p.voice.name})`);
    el.querySelectorAll("[data-move]").forEach((b) => b.addEventListener("click", () => {
      const j = i + Number(b.dataset.move);
      if (j < 0 || j >= list.length) return;
      [list[i], list[j]] = [list[j], list[i]];
      show();
    }));
    el.querySelector("[data-remove]").addEventListener("click", () => {
      if (p.ownUrl) URL.revokeObjectURL(p.ownUrl);
      list.splice(i, 1);
      show();
    });
    return el;
  }

  // "eiffel_tower-2.jpg" -> "Eiffel Tower"
  function nameFromFile(filename) {
    let words = filename.replace(/\.[^.]+$/, "").replace(/\(.*?\)/g, " ").split(/[\s_\-.]+/).filter(Boolean);
    while (words.length && /^\d+$/.test(words[words.length - 1])) words.pop();
    const name = words.join(" ");
    return name === name.toLowerCase() ? name.replace(/(^|\s)\S/g, (c) => c.toUpperCase()) : name;
  }

  // ------------------------------------------------------------------ render
  $("generate").addEventListener("click", async () => {
    showError("");
    if (!list.length) return showError("Add at least one place.");
    const unnamed = list.findIndex((p) => !p.name.trim());
    if (unnamed >= 0) return showError(`Place ${unnamed + 1} needs a name.`);
    const bare = list.findIndex((p) => !p.own && (p.chosen == null || !p.media[p.chosen]));
    if (bare >= 0) return showError(`Choose a picture of ${list[bare].name}, or upload your own.`);
    const seconds = parseFloat($("seconds").value);
    if (!(seconds >= 4 && seconds <= 60)) return showError("Drawing time must be between 4 and 60 seconds.");
    const reveal = parseFloat($("reveal").value);
    if (!(reveal >= 1.5 && reveal <= 15)) return showError("The time in colour must be between 1.5 and 15 seconds.");

    const form = new FormData();
    const places = list.map((p) => {
      if (p.own) form.append(`picture_${p.id}`, p.own, p.own.name);
      if (p.voice) form.append(`voice_${p.id}`, p.voice, p.voice.name);
      return { name: p.name.trim(), location: p.location.trim(), country: p.country.trim(),
               lat: p.lat, lon: p.lon, facts: p.facts.split("\n").map((f) => f.trim()).filter(Boolean),
               media: p.own ? null : p.media[p.chosen], upload: p.own ? `picture_${p.id}` : null,
               voice: p.voice ? `voice_${p.id}` : null };
    });
    const music = $("music").files[0];
    if (music) form.append("music", music, music.name);
    form.append("data", JSON.stringify({
      places, title: $("title").value.trim(), resolution: $("resolution").value, seconds, reveal,
      guess: $("guess").checked, intro: $("intro").checked, hook: $("hook").value.trim(),
      outro: $("outro").checked, question: $("question").value.trim(), sound_effects: $("sfx").checked,
    }));

    $("generate").disabled = true;
    $("result").hidden = true;
    setStatus({ state: "queued", progress: 0, message: "Uploading…" });
    try {
      const res = await fetch("/api/travel/render", { method: "POST", body: form });
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
    $("result-summary").textContent = `${(info.places || []).join(", ")} · ${Math.round(info.duration || 0)} s` +
      `${size} · made in ${Math.round(info.render_seconds || job.elapsed)} s`;
    $("result").hidden = false;
    $("result").scrollIntoView({ behavior: "smooth", block: "start" });
  }

  show();
})();
