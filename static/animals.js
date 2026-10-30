(function () {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const MAX = 20;
  let list = [];  // {id, name, facts, file, preview, voice}
  let nextId = 0;
  let pollTimer = null;

  function showError(msg) {
    $("form-error").textContent = msg;
    $("form-error").hidden = !msg;
  }

  // "snow_leopard-2.jpg" -> "Snow Leopard"
  function nameFromFile(filename) {
    let words = filename.replace(/\.[^.]+$/, "").replace(/\(.*?\)/g, " ").split(/[\s_\-.]+/).filter(Boolean);
    while (words.length && /^\d+$/.test(words[words.length - 1])) words.pop();
    const name = words.join(" ");
    return name === name.toLowerCase() ? name.replace(/(^|\s)\S/g, (c) => c.toUpperCase()) : name;
  }

  $("photos").addEventListener("change", () => {
    const files = [...$("photos").files].slice(0, MAX - list.length);
    for (const file of files) {
      list.push({ id: nextId++, name: nameFromFile(file.name), facts: "", file,
                  preview: URL.createObjectURL(file), voice: null });
    }
    $("photos").value = "";
    $("photos-info").textContent = files.length ? `${files.length} photo${files.length > 1 ? "s" : ""} added.` : "";
    show();
  });

  function show() {
    const box = $("animal-list");
    box.innerHTML = "";
    list.forEach((a, i) => {
      const row = document.createElement("div");
      row.className = "animal";
      row.innerHTML = `
        <span class="pic"><img alt=""></span>
        <div class="fields">
          <label>Name <input type="text" class="name" maxlength="60"></label>
          <label>Facts <span class="unit">one per line</span>
            <textarea class="facts" rows="5" placeholder="Lions live in groups called prides.&#10;A lion's roar can be heard 8 km away."></textarea></label>
          <label>Recording <span class="unit">optional, plays from the moment this animal is drawn</span>
            <input type="file" class="voice" accept="audio/*,.m4a,.mp3,.wav,.aac,.ogg,.flac"></label>
        </div>
        <span class="tools">
          <span class="num">#${i + 1}</span>
          <button type="button" class="small" data-move="-1" aria-label="Move up">▲</button>
          <button type="button" class="small" data-move="1" aria-label="Move down">▼</button>
          <button type="button" class="small" data-remove aria-label="Remove">✕</button>
        </span>`;
      row.querySelector("img").src = a.preview;
      row.querySelector("img").alt = a.name;
      const name = row.querySelector(".name");
      name.value = a.name;
      name.addEventListener("input", () => { a.name = name.value; });
      const facts = row.querySelector(".facts");
      facts.value = a.facts;
      facts.addEventListener("input", () => { a.facts = facts.value; });
      const voice = row.querySelector(".voice");
      voice.addEventListener("change", () => { a.voice = voice.files[0] || null; });
      row.querySelectorAll("[data-move]").forEach((b) => b.addEventListener("click", () => {
        const j = i + Number(b.dataset.move);
        if (j < 0 || j >= list.length) return;
        [list[i], list[j]] = [list[j], list[i]];
        show();
      }));
      row.querySelector("[data-remove]").addEventListener("click", () => {
        URL.revokeObjectURL(a.preview);
        list.splice(i, 1);
        show();
      });
      if (a.voice) row.querySelector(".voice").insertAdjacentText("afterend", ` (${a.voice.name})`);
      box.appendChild(row);
    });
    $("export-card").hidden = !list.length;
    $("generate").textContent = list.length ? `Draw ${list.length} animal${list.length > 1 ? "s" : ""}` : "Draw the animals";
  }

  // ------------------------------------------------------------------ render
  $("generate").addEventListener("click", async () => {
    showError("");
    if (!list.length) return showError("Add at least one animal.");
    const unnamed = list.findIndex((a) => !a.name.trim());
    if (unnamed >= 0) return showError(`Animal ${unnamed + 1} needs a name.`);
    const seconds = parseFloat($("seconds").value);
    if (!(seconds >= 4 && seconds <= 60)) return showError("Drawing time must be between 4 and 60 seconds.");

    const form = new FormData();
    const animals = list.map((a) => {
      form.append(`photo_${a.id}`, a.file, a.file.name);
      if (a.voice) form.append(`voice_${a.id}`, a.voice, a.voice.name);
      return { name: a.name.trim(), facts: a.facts.split("\n").map((f) => f.trim()).filter(Boolean),
               photo: `photo_${a.id}`, voice: a.voice ? `voice_${a.id}` : null };
    });
    const music = $("music").files[0];
    if (music) form.append("music", music, music.name);
    form.append("data", JSON.stringify({
      animals, title: $("title").value.trim(), resolution: $("resolution").value, seconds,
      guess: $("guess").checked, intro: $("intro").checked, hook: $("hook").value.trim(),
      outro: $("outro").checked, question: $("question").value.trim(), sound_effects: $("sfx").checked,
    }));

    $("generate").disabled = true;
    $("result").hidden = true;
    setStatus({ state: "queued", progress: 0, message: "Uploading…" });
    try {
      const res = await fetch("/api/animals/render", { method: "POST", body: form });
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
    const size = info.file_size ? ` · ${(info.file_size / 1048576).toFixed(1)} MB` : "";
    $("result-summary").textContent = `${(info.animals || []).join(", ")} · ${Math.round(info.duration || 0)} s` +
      `${size} · made in ${Math.round(info.render_seconds || job.elapsed)} s`;
    $("result").hidden = false;
    $("result").scrollIntoView({ behavior: "smooth", block: "start" });
  }
})();
