(function () {
  "use strict";

  const cfg = window.PENCIL;
  const form = document.getElementById("job-form");
  const generate = document.getElementById("generate");
  const formError = document.getElementById("form-error");
  const statusCard = document.getElementById("status");
  const resultCard = document.getElementById("result");
  const files = { start: null, end: null };
  const ACCEPTED = /\.(png|jpe?g|webp|bmp|tiff?)$/i;
  let pollTimer = null;

  // ------------------------------------------------------------------ uploads
  const startCard = document.querySelector('.upload[data-key="start"]');
  const startMode = () => (form.querySelector("input[name=start_mode]:checked") || {}).value || "blank";
  function applyStartMode() {
    const blank = startMode() === "blank";
    startCard.classList.toggle("blank", blank);
    startCard.querySelector("input[type=file]").disabled = blank;
    formError.hidden = true;
  }
  form.querySelectorAll("input[name=start_mode]").forEach((radio) => radio.addEventListener("change", applyStartMode));
  applyStartMode();

  document.querySelectorAll(".upload").forEach((card) => {
    const key = card.dataset.key;
    const input = card.querySelector("input[type=file]");
    const drop = card.querySelector(".drop");
    const img = card.querySelector("[data-role=preview]");
    const placeholder = card.querySelector("[data-role=placeholder]");
    const meta = card.querySelector("[data-role=meta]");

    function setMeta(text, isError) {
      meta.textContent = text;
      meta.classList.toggle("error", !!isError);
    }

    function accept(file) {
      files[key] = null;
      if (!file) return;
      if (!ACCEPTED.test(file.name) && !/^image\/(png|jpeg|webp|bmp|tiff)$/.test(file.type)) {
        setMeta(`Unsupported file type: ${file.name}. Use PNG, JPEG, WEBP, BMP or TIFF.`, true);
        return;
      }
      if (file.size > cfg.maxUploadMb * 1024 * 1024) {
        setMeta(`File is ${(file.size / 1048576).toFixed(1)} MB; the limit is ${cfg.maxUploadMb} MB.`, true);
        return;
      }
      const url = URL.createObjectURL(file);
      const probe = new Image();
      probe.onload = () => {
        const w = probe.naturalWidth, h = probe.naturalHeight;
        if (Math.min(w, h) < cfg.minSide) {
          setMeta(`Image is ${w}×${h}; the shorter side must be at least ${cfg.minSide} px.`, true);
          URL.revokeObjectURL(url);
          return;
        }
        if (img.src) URL.revokeObjectURL(img.src);
        img.src = url;
        img.hidden = false;
        placeholder.hidden = true;
        files[key] = file;
        setMeta(`${file.name} · ${w}×${h} · ${(file.size / 1048576).toFixed(2)} MB`);
        formError.hidden = true;
      };
      probe.onerror = () => {
        // Some formats (e.g. TIFF) can't be previewed in every browser; the server still validates them.
        URL.revokeObjectURL(url);
        img.hidden = true;
        placeholder.hidden = false;
        files[key] = file;
        setMeta(`${file.name} · preview not available in this browser · ${(file.size / 1048576).toFixed(2)} MB`);
      };
      probe.src = url;
    }

    input.addEventListener("change", () => accept(input.files[0]));
    ["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => {
      e.preventDefault();
      drop.classList.add("dragover");
    }));
    ["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => {
      e.preventDefault();
      drop.classList.remove("dragover");
    }));
    drop.addEventListener("drop", (e) => {
      const file = e.dataTransfer.files[0];
      if (file && !(key === "start" && startMode() === "blank")) accept(file);
    });
  });

  // ------------------------------------------------------------------ settings & timeline
  const resolution = document.getElementById("resolution");
  const customSize = document.getElementById("custom-size");
  const durationInput = document.getElementById("duration");
  const holdInput = document.getElementById("hold");
  const timeline = document.getElementById("timeline");

  resolution.addEventListener("change", () => {
    customSize.hidden = resolution.value !== "custom";
    updateAspect();
  });
  customSize.addEventListener("input", updateAspect);

  function updateAspect() {
    let w = 1080, h = 1920;
    if (resolution.value === "custom") {
      w = +form.width.value || w;
      h = +form.height.value || h;
    } else {
      [w, h] = resolution.value.split("x").map(Number);
    }
    document.querySelectorAll(".frame").forEach((f) => { f.style.aspectRatio = `${w} / ${h}`; });
  }

  function stageSpans() {
    const duration = Math.max(0, +durationInput.value || 0);
    const hold = Math.max(0, +holdInput.value || 0);
    const drawing = Math.max(0, duration - hold);
    const units = cfg.stages.reduce((a, s) => a + s[1], 0);
    let t = 0;
    const spans = cfg.stages.map(([name, u], i) => {
      const span = { name, start: t, end: t + (drawing * u) / units, cls: `s${i + 1}` };
      t = span.end;
      return span;
    });
    if (hold > 0) spans.push({ name: "Hold", start: drawing, end: duration, cls: "hold" });
    return { spans, duration };
  }

  function renderTimeline() {
    const { spans, duration } = stageSpans();
    timeline.innerHTML = "";
    if (!duration) return;
    spans.forEach((s) => {
      const seg = document.createElement("div");
      seg.className = `seg ${s.cls}`;
      seg.style.flex = String(Math.max(s.end - s.start, 0.0001));
      seg.title = `${s.name}: ${s.start.toFixed(1)}–${s.end.toFixed(1)} s`;
      seg.innerHTML = `<b></b><span>${s.start.toFixed(1)}–${s.end.toFixed(1)} s</span>`;
      seg.querySelector("b").textContent = s.name;
      timeline.appendChild(seg);
    });
  }

  function highlightStage(message) {
    // Frame-rendering messages start with the stage title ("Feature refinement - frame 12/450").
    const titles = cfg.stages.map(([name]) => name).concat(["Final artwork (hold)"]);
    const msg = message || "";
    const hit = msg.includes(" - frame ") ? titles.findIndex((title) => msg.startsWith(title)) : -1;
    timeline.querySelectorAll(".seg").forEach((el, i) => el.classList.toggle("active", i === hit));
  }

  durationInput.addEventListener("input", renderTimeline);
  holdInput.addEventListener("input", renderTimeline);
  renderTimeline();
  updateAspect();

  // ------------------------------------------------------------------ submit & poll
  function showFormError(msg) {
    formError.textContent = msg;
    formError.hidden = false;
  }

  function validate() {
    if (startMode() === "image" && !files.start) return "Please choose a valid Start image, or start from blank paper.";
    if (!files.end) return "Please choose a valid End image.";
    const d = +durationInput.value, h = +holdInput.value;
    if (!(d >= 4 && d <= 120)) return "Duration must be between 4 and 120 seconds.";
    if (!(h >= 0 && h <= 10)) return "Final hold must be between 0 and 10 seconds.";
    if (d - h < 3) return "Duration minus the final hold must leave at least 3 seconds of drawing.";
    if (resolution.value === "custom") {
      const w = +form.width.value, hh = +form.height.value;
      if (!(w >= 160 && hh >= 160 && w <= 3840 && hh <= 3840)) return "Width and height must be between 160 and 3840 px.";
      if (w % 2 || hh % 2) return "Width and height must be even numbers.";
    }
    return null;
  }

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    formError.hidden = true;
    const problem = validate();
    if (problem) return showFormError(problem);

    const data = new FormData(form);
    if (startMode() === "image") data.set("start", files.start);
    else data.delete("start");
    data.set("end", files.end);
    form.querySelectorAll("input[type=checkbox]").forEach((cb) => data.set(cb.name, cb.checked ? "true" : "false"));

    generate.disabled = true;
    resultCard.hidden = true;
    setStatus({ state: "uploading", progress: 0, message: "Uploading images…" });
    try {
      const res = await fetch("/api/jobs", { method: "POST", body: data });
      const body = await res.json().catch(() => ({ error: `Server error (${res.status}).` }));
      if (!res.ok) throw new Error(body.error || `Server error (${res.status}).`);
      setStatus(body);
      poll(body.id);
    } catch (err) {
      statusCard.hidden = true;
      generate.disabled = false;
      showFormError(err.message || "Upload failed.");
    }
  });

  function poll(id) {
    clearTimeout(pollTimer);
    pollTimer = setTimeout(async () => {
      try {
        const res = await fetch(`/api/jobs/${id}`, { cache: "no-store" });
        const job = await res.json();
        if (!res.ok) throw new Error(job.error || "Lost track of the render job.");
        setStatus(job);
        if (job.state === "done") return showResult(job);
        if (job.state === "error") { generate.disabled = false; return; }
        poll(id);
      } catch (err) {
        setStatus({ state: "error", progress: 0, error: err.message });
        generate.disabled = false;
      }
    }, 600);
  }

  function setStatus(job) {
    statusCard.hidden = false;
    const pct = Math.round((job.progress || 0) * 100);
    const isError = job.state === "error";
    statusCard.classList.toggle("error", isError);
    document.getElementById("status-title").textContent =
      isError ? "Render failed" : job.state === "done" ? "Finished" : job.state === "queued" ? "Queued" : "Rendering";
    document.getElementById("status-detail").textContent = isError
      ? job.error
      : job.state === "queued" && job.queue_position > 1
        ? `Waiting for ${job.queue_position - 1} other render(s) to finish.`
        : job.message || "";
    document.getElementById("bar").style.width = `${isError ? 100 : pct}%`;
    document.getElementById("progressbar").setAttribute("aria-valuenow", String(pct));
    document.getElementById("pct").textContent = isError ? "" : `${pct}%`;
    const eta = document.getElementById("eta");
    eta.textContent = job.eta != null ? `about ${Math.ceil(job.eta)} s left` : job.elapsed ? `${job.elapsed} s elapsed` : "";
    highlightStage(job.state === "running" ? job.message : "");
  }

  function showResult(job) {
    generate.disabled = false;
    highlightStage("");
    const video = document.getElementById("video");
    video.src = `/api/jobs/${job.id}/video?v=${Date.now()}`;
    document.getElementById("download").href = `/api/jobs/${job.id}/download`;
    const c = job.config, info = job.info || {};
    const size = info.file_size ? ` · ${(info.file_size / 1048576).toFixed(1)} MB` : "";
    document.getElementById("result-summary").textContent =
      `${c.width}×${c.height}, ${c.fps} fps, ${c.duration} s${size} · rendered in ${info.render_seconds ?? job.elapsed} s`;
    const notes = document.getElementById("notes");
    notes.innerHTML = "";
    const add = (text, cls) => {
      const li = document.createElement("li");
      li.textContent = text;
      if (cls) li.className = cls;
      notes.appendChild(li);
    };
    if (info.focus) {
      const f = info.focus;
      add(f.face ? `Face detected (${f.method}); features are drawn from the eyes outwards.`
        : "No face detected; drawing order follows the most detailed area.");
    }
    if (info.alignment) add(`Alignment: ${info.alignment}.`);
    (job.warnings || []).forEach((w) => add(w, "warn"));
    resultCard.hidden = false;
    resultCard.scrollIntoView({ behavior: "smooth", block: "start" });
  }
})();
