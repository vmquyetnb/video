(function () {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const MAX = 12;
  // Every player shown: {name, on, file?, preview?, ...Commons fields}
  let list = [];
  let pollTimer = null;
  let nextId = 0;

  function showError(msg) {
    $("form-error").textContent = msg;
    $("form-error").hidden = !msg;
  }

  // "cristiano_ronaldo-2.jpg" -> "Cristiano Ronaldo" (same rule as the command line)
  function nameFromFile(filename) {
    let words = filename.replace(/\.[^.]+$/, "").replace(/\(.*?\)/g, " ").split(/[\s_\-.]+/).filter(Boolean);
    while (words.length && /^\d+$/.test(words[words.length - 1])) words.pop();
    const name = words.join(" ");
    return name === name.toLowerCase() ? name.replace(/(^|\s)\S/g, (c) => c.toUpperCase()) : name;
  }

  function selectedCount() {
    return list.filter((p) => p.on).length;
  }

  // ------------------------------------------------------------------ sources
  $("photos").addEventListener("change", () => {
    const files = [...$("photos").files];
    for (const file of files) {
      const on = selectedCount() < MAX;
      list.push({ id: nextId++, name: nameFromFile(file.name), on, file, preview: URL.createObjectURL(file) });
    }
    $("photos").value = "";
    $("photos-info").textContent = files.length ? `${files.length} photo${files.length > 1 ? "s" : ""} added.` : "";
    show();
  });

  async function findPlayers() {
    const country = $("country").value.trim();
    const info = $("find-info");
    showError("");
    if (!country) { info.textContent = "Type a country first."; return; }
    info.textContent = "Searching Wikidata…";
    $("find").disabled = true;
    try {
      const res = await fetch(`/api/players?country=${encodeURIComponent(country)}`);
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Search failed.");
      if (!data.players.length) throw new Error("No players with a photo were found.");
      list = list.filter((p) => p.file);  // a new search replaces the previous one
      data.players.forEach((p, i) => {
        list.push(Object.assign({ id: nextId++, on: i < 6 && selectedCount() < MAX }, p));
      });
      if (!$("title").value.trim()) $("title").value = country.toUpperCase();
      info.textContent = `${data.players.length} players found.`;
      show();
    } catch (err) {
      info.textContent = err.message;
    } finally {
      $("find").disabled = false;
    }
  }
  $("find").addEventListener("click", findPlayers);
  $("country").addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); findPlayers(); } });

  // ------------------------------------------------------------------ list
  function show() {
    const grid = $("player-grid");
    grid.innerHTML = "";
    list.forEach((p, i) => {
      const card = document.createElement("div");
      card.className = "player-pick";
      card.title = p.file ? `Your photo: ${p.file.name}` : `${p.name} — photo: ${p.author || "unknown author"} (${p.licence})`;
      card.innerHTML = `
        <label class="pick"><input type="checkbox"><span class="pic"><img alt="" loading="lazy"></span></label>
        <input type="text" class="name-input" maxlength="60" aria-label="Name">
        <span class="tools">
          <button type="button" class="small" data-move="-1" aria-label="Move earlier">◀</button>
          <button type="button" class="small" data-move="1" aria-label="Move later">▶</button>
          <button type="button" class="small" data-remove aria-label="Remove">✕</button>
        </span>`;
      const img = card.querySelector("img");
      img.src = p.preview || p.thumb || p.photo_url;
      img.alt = p.name;
      const box = card.querySelector("input[type=checkbox]");
      box.checked = p.on;
      box.addEventListener("change", () => { p.on = box.checked; limit(); });
      const name = card.querySelector(".name-input");
      name.value = p.name;
      name.addEventListener("input", () => { p.name = name.value; });
      card.querySelectorAll("[data-move]").forEach((b) => b.addEventListener("click", () => {
        const j = i + Number(b.dataset.move);
        if (j < 0 || j >= list.length) return;
        [list[i], list[j]] = [list[j], list[i]];
        show();
      }));
      card.querySelector("[data-remove]").addEventListener("click", () => {
        if (p.preview) URL.revokeObjectURL(p.preview);
        list.splice(i, 1);
        show();
      });
      grid.appendChild(card);
    });
    $("players-card").hidden = !list.length;
    $("export-card").hidden = !list.length;
    limit();
  }

  function limit() {
    const count = selectedCount();
    document.querySelectorAll(".player-pick").forEach((card, i) => {
      const box = card.querySelector("input[type=checkbox]");
      box.disabled = !box.checked && count >= MAX;
      card.classList.toggle("on", box.checked);
    });
    $("generate").textContent = count ? `Draw ${count} player${count > 1 ? "s" : ""}` : "Draw the players";
  }

  $("audio").addEventListener("change", () => { $("fit-row").hidden = !$("audio").files.length; });

  // ------------------------------------------------------------------ render
  $("generate").addEventListener("click", async () => {
    showError("");
    const chosen = list.filter((p) => p.on);
    if (!chosen.length) return showError("Tick at least one player.");
    const unnamed = chosen.find((p) => !p.name.trim());
    if (unnamed) return showError("Every player needs a name.");
    const seconds = parseFloat($("seconds").value);
    if (!(seconds >= 4 && seconds <= 60)) return showError("Drawing time must be between 4 and 60 seconds.");

    const form = new FormData();
    const players = chosen.map((p) => {
      if (!p.file) {
        const { id, on, preview, ...commons } = p;
        return commons;
      }
      const field = `photo_${p.id}`;
      form.append(field, p.file, p.file.name);
      return { name: p.name.trim(), upload: field };
    });
    const audio = $("audio").files[0];
    if (audio) form.append("audio", audio, audio.name);
    form.append("data", JSON.stringify({
      players, title: $("title").value.trim(), resolution: $("resolution").value, seconds,
      background: $("background").value, fit_to_audio: $("fit").checked,
      intro: $("intro").checked, hook: $("hook").value.trim(), highlights: $("highlights").checked,
      outro: $("outro").checked, question: $("question").value.trim(), sound_effects: $("sfx").checked,
    }));

    $("generate").disabled = true;
    $("result").hidden = true;
    setStatus({ state: "queued", progress: 0, message: audio || chosen.some((p) => p.file) ? "Uploading…" : "Starting…" });
    try {
      const res = await fetch("/api/players/render", { method: "POST", body: form });
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
    $("result-summary").textContent = `${(info.players || []).join(", ")} · ${Math.round(info.duration || 0)} s` +
      `${size} · made in ${Math.round(info.render_seconds || job.elapsed)} s`;
    $("result").hidden = false;
    $("result").scrollIntoView({ behavior: "smooth", block: "start" });
  }
})();
