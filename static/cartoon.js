(function () {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const CONFIG = window.CARTOON || { vocab: {}, models: {}, env: {} };
  const V = CONFIG.vocab;
  const HERO = CONFIG.hero || "Peeko";
  let scenes = [];  // {setting, time, weather, narration, actions (text), say, effect, holds, friend, board, item, count}
  let picture = null;
  let pollTimer = null;

  function showError(msg) {
    $("form-error").textContent = msg;
    $("form-error").hidden = !msg;
  }

  // ------------------------------------------------------------------ AI keys (shared with the Travel tab)
  const store = {
    get() { try { return JSON.parse(localStorage.getItem("travel-settings") || "{}"); } catch (e) { return {}; } },
    set(value) { try { localStorage.setItem("travel-settings", JSON.stringify(value)); } catch (e) { /* private mode */ } },
  };
  const saved = store.get();
  saved.keys = saved.keys || {};
  saved.models = saved.models || {};
  const provider = () => $("provider").value;

  function showProvider() {
    const p = provider();
    $("ai-key").value = saved.keys[p] || "";
    $("model").value = saved.models[p] || "";
    $("model").placeholder = CONFIG.models[p] || "";
    $("key-note").textContent = CONFIG.env[p] && !saved.keys[p]
      ? "using the key set on the server" : "saved in this browser only";
  }
  if (saved.provider) $("provider").value = saved.provider;
  showProvider();
  $("provider").addEventListener("change", () => { saved.provider = provider(); store.set(saved); showProvider(); });
  $("ai-key").addEventListener("change", () => { saved.keys[provider()] = $("ai-key").value.trim(); store.set(saved); showProvider(); });
  $("model").addEventListener("change", () => { saved.models[provider()] = $("model").value.trim(); store.set(saved); });

  // ------------------------------------------------------------------ character
  const character = () => (document.querySelector('input[name="character"]:checked') || {}).value || "peeko";

  function showCharacter() {
    const own = character() === "picture";
    $("peeko-box").hidden = own;
    $("picture-box").hidden = !own;
    const name = $("name").value.trim();
    if (own && name === HERO) $("name").value = picture ? nameFromFile(picture.name) : "";
    if (!own && !name) $("name").value = HERO;
    $("name").placeholder = own ? "Pip" : HERO;
  }
  document.querySelectorAll('input[name="character"]').forEach((r) => r.addEventListener("change", showCharacter));
  showCharacter();

  $("picture").addEventListener("change", async () => {
    picture = $("picture").files[0] || null;
    $("cutout").hidden = true;
    if (!picture) { $("cutout-note").textContent = "no character yet"; return; }
    if (!$("name").value.trim()) $("name").value = nameFromFile(picture.name);
    $("cutout-note").textContent = "cutting out…";
    $("cutout-note").hidden = false;
    const form = new FormData();
    form.append("picture", picture, picture.name);
    try {
      const res = await fetch("/api/cartoon/cutout", { method: "POST", body: form });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.error || `Server error (${res.status}).`);
      }
      const old = $("cutout").src;
      $("cutout").src = URL.createObjectURL(await res.blob());
      if (old.startsWith("blob:")) URL.revokeObjectURL(old);
      $("cutout").alt = `${$("name").value || "The character"}, cut out`;
      $("cutout").hidden = false;
      $("cutout-note").hidden = true;
    } catch (err) {
      $("cutout-note").textContent = err.message;
    }
  });

  function nameFromFile(filename) {
    const words = filename.replace(/\.[^.]+$/, "").split(/[\s_\-.]+/).filter((w) => w && !/^\d+$/.test(w));
    const name = words.join(" ");
    return name === name.toLowerCase() ? name.replace(/(^|\s)\S/g, (c) => c.toUpperCase()) : name;
  }

  // ------------------------------------------------------------------ story
  let meta = { episode: 0, season: 0 };  // kept from a loaded episode

  const linesText = (lines) => (lines || []).map((l) =>
    `${l.who ? `${l.who}: ` : ""}${l.say}${l.do ? ` (${l.do})` : ""}`).join("\n");

  function useStory(story) {
    $("title").value = story.title || "";
    $("question").value = story.question || "";
    $("lesson").value = story.lesson || "";
    $("next").value = story.next || "";
    $("tune").value = story.music || "happy";
    $("hook").checked = !!story.hook;
    meta = { episode: story.episode || 0, season: story.season || 0 };
    scenes = (story.scenes || []).map((s) => ({
      ...s, actions: (s.actions || []).join(", "), say: "", cast: (s.cast || []).join(", "),
      lines: linesText((s.say ? [{ who: "", say: s.say }] : []).concat(s.lines || [])),
      quiz: (s.quiz || []).join(", "), answer: s.answer || "" }));
    show();
  }
  $("hook").addEventListener("change", () => show());

  async function askStory(body) {
    const res = await fetch("/api/cartoon/story", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ...body, name: $("name").value.trim(), peeko: character() === "peeko" }),
    });
    const story = await res.json().catch(() => ({ error: `Server error (${res.status}).` }));
    if (!res.ok) throw new Error(story.error || `Server error (${res.status}).`);
    useStory(story);
    return story;
  }

  const episodes = CONFIG.episodes || [];
  function showLesson() {
    const e = episodes.find((x) => x.id === $("episode").value);
    $("episode-lesson").textContent = e ? `Teaches: ${e.lesson}` : "";
  }
  $("episode").addEventListener("change", showLesson);
  showLesson();

  $("load-episode").addEventListener("click", async () => {
    try {
      const story = await askStory({ episode: $("episode").value });
      $("episode-info").textContent = `${story.scenes.length} scenes loaded: change anything you like.`;
    } catch (err) {
      $("episode-info").textContent = err.message;
    }
  });

  $("sample").addEventListener("click", async () => {
    try {
      await askStory({ sample: true });
      $("write-info").textContent = "Sample story loaded: change anything you like.";
    } catch (err) {
      $("write-info").textContent = err.message;
    }
  });

  $("write").addEventListener("click", async () => {
    $("write").disabled = true;
    $("write-info").textContent = "Writing… this can take up to a minute.";
    try {
      const body = await askStory({ provider: provider(), key: $("ai-key").value.trim(),
                                    model: $("model").value.trim(), idea: $("idea").value.trim(),
                                    scenes: Number($("scene-count").value) });
      $("write-info").textContent = `${body.scenes.length} scenes written. Read them and change what you like.`;
    } catch (err) {
      $("write-info").textContent = err.message;
    } finally {
      $("write").disabled = false;
    }
  });

  $("add-scene").addEventListener("click", () => {
    const last = scenes[scenes.length - 1];
    scenes.push({ setting: last ? last.setting : "meadow", time: last ? last.time : "day", weather: "clear",
                  narration: "", actions: "walk, happy", say: "", effect: "none", holds: "", friend: "",
                  board: "", item: "", count: 0, cast: last ? last.cast : "", lines: "", quiz: "", answer: "" });
    show();
  });

  function options(values, chosen, none) {
    return values.map((v) => `<option value="${v}"${v === (chosen || "") ? " selected" : ""}>${v || none || v}</option>`)
      .join("");
  }
  const props = [""].concat(V.props || []);

  function show() {
    const box = $("scene-list");
    box.innerHTML = "";
    scenes.forEach((s, i) => {
      const el = document.createElement("div");
      el.className = "animal scene-row";
      el.innerHTML = `
        <div class="fields">
          <div class="quad">
            <label>Where <select class="setting">${options(V.settings, s.setting)}</select></label>
            <label>When <select class="time">${options(V.times, s.time)}</select></label>
            <label>Weather <select class="weather">${options(V.weather, s.weather)}</select></label>
            <label>Effect <select class="effect">${options(V.effects, s.effect)}</select></label>
          </div>
          <label>Narration <textarea class="narration" rows="2" maxlength="400"></textarea></label>
          <div class="pair">
            <label>Actions <span class="unit">in order, separated by commas</span>
              <input type="text" class="actions" maxlength="120"></label>
            <label>Holds <select class="holds">${options(props, s.holds, "nothing")}</select></label>
          </div>
          <div class="talk">
            <label>Friends here <span class="unit">e.g. Lulu, sad Bobo</span>
              <input type="text" class="cast" maxlength="60"></label>
            <label>Conversation <span class="unit">one line each: "Kiko: Hello! (wave)"</span>
              <textarea class="lines" rows="3" maxlength="1500"></textarea></label>
          </div>
          <div class="chalk">
            <label>Chalkboard words <input type="text" class="board" maxlength="40"></label>
            <label>Picture <select class="item">${options(props, s.item, "none")}</select></label>
            <label>Count <input type="number" class="count" min="0" max="10" step="1"></label>
          </div>
          <div class="pair">
            <label>Quiz pictures <span class="unit">2 or 3, e.g. carrot, candy</span>
              <input type="text" class="quiz" maxlength="80"></label>
            <label>Right answer <input type="text" class="answer" maxlength="30"></label>
          </div>
        </div>
        <span class="tools">
          <span class="num">#${i + 1}${i === 0 && $("hook").checked ? '<span class="badge-hook">teaser</span>' : ""}</span>
          <button type="button" class="small" data-move="-1" aria-label="Move up">▲</button>
          <button type="button" class="small" data-move="1" aria-label="Move down">▼</button>
          <button type="button" class="small" data-remove aria-label="Remove">✕</button>
        </span>`;
      for (const field of ["setting", "time", "weather", "effect", "narration", "actions", "holds", "cast", "lines",
                           "board", "item", "count", "quiz", "answer"]) {
        const input = el.querySelector(`.${field}`);
        if (input.tagName !== "SELECT") input.value = field === "count" ? (s.count || 0) : (s[field] || "");
        input.addEventListener(input.tagName === "SELECT" ? "change" : "input", () => {
          s[field] = field === "count" ? Math.max(0, Math.min(10, Number(input.value) || 0)) : input.value;
        });
      }
      el.querySelectorAll("[data-move]").forEach((b) => b.addEventListener("click", () => {
        const j = i + Number(b.dataset.move);
        if (j < 0 || j >= scenes.length) return;
        [scenes[i], scenes[j]] = [scenes[j], scenes[i]];
        show();
      }));
      el.querySelector("[data-remove]").addEventListener("click", () => { scenes.splice(i, 1); show(); });
      box.appendChild(el);
    });
    $("generate").textContent = scenes.length ? `Animate ${scenes.length} scene${scenes.length > 1 ? "s" : ""}` : "Animate the story";
  }

  // ------------------------------------------------------------------ render
  $("generate").addEventListener("click", async () => {
    showError("");
    const own = character() === "picture";
    if (own && !picture) return showError("Add the character's picture first, or choose Peeko.");
    if (!scenes.length) return showError("Write the story first, or use the sample story.");
    const empty = scenes.findIndex((s) => !s.narration.trim() && !(s.say || "").trim() && !(s.lines || "").trim());
    if (empty >= 0) return showError(`Scene ${empty + 1} needs a narration or a line of conversation.`);
    const known = new Set(V.actions.concat(["talk"]));
    for (const [i, s] of scenes.entries()) {
      const bad = s.actions.split(",").map((a) => a.trim().toLowerCase()).filter((a) => a && !known.has(a));
      if (bad.length) return showError(`Scene ${i + 1}: unknown action "${bad[0]}". Use: ${V.actions.join(", ")}.`);
    }
    const form = new FormData();
    if (own) form.append("picture", picture, picture.name);
    const music = $("music").files[0];
    if (music) form.append("music", music, music.name);
    const story = { title: $("title").value.trim(), question: $("question").value.trim(),
                    lesson: $("lesson").value.trim(), next: $("next").value.trim(), music: $("tune").value,
                    hook: $("hook").checked, episode: meta.episode, season: meta.season,
                    scenes: scenes.map((s) => ({ ...s, actions: s.actions.split(",").map((a) => a.trim()).filter(Boolean) })) };
    form.append("data", JSON.stringify({
      character: own ? "picture" : "peeko",
      name: $("name").value.trim(), faces: $("faces").value, story, resolution: $("resolution").value,
      narrator: $("narrator") ? $("narrator").value : "", voice: $("voice") ? $("voice").value : "",
      pitch: $("pitch") ? Number($("pitch").value) : 1.25,
      intro: $("intro").checked, outro: $("outro").checked, sound_effects: $("sfx").checked,
      series: $("series").checked,
    }));
    $("generate").disabled = true;
    $("result").hidden = true;
    setStatus({ state: "queued", progress: 0, message: "Uploading…" });
    try {
      const res = await fetch("/api/cartoon/render", { method: "POST", body: form });
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
    $("status-title").textContent = isError ? "Animation failed" : job.state === "done" ? "Finished"
      : job.state === "queued" ? "Queued" : "Animating";
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
    const notes = (job.warnings || []).length ? ` · ${job.warnings.join(" ")}` : "";
    $("result-summary").textContent = `${info.scenes || 0} scenes · ${Math.round(info.duration || 0)} s${size}` +
      ` · made in ${Math.round(info.render_seconds || job.elapsed)} s${notes}`;
    $("result").hidden = false;
    $("result").scrollIntoView({ behavior: "smooth", block: "start" });
  }

  show();
})();
