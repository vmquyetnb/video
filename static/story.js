(function () {
  "use strict";

  const cfg = window.STORY;
  const form = document.getElementById("story-form");
  const sceneList = document.getElementById("scene-list");
  const formError = document.getElementById("form-error");
  const generate = document.getElementById("generate");
  const statusCard = document.getElementById("status");
  const resultCard = document.getElementById("result");
  let pollTimer = null;

  const $ = (id) => document.getElementById(id);
  const voiceMode = () => (form.querySelector("input[name=voice_mode]:checked") || {}).value || "single";

  function showError(msg) {
    formError.textContent = msg;
    formError.hidden = !msg;
  }

  // ------------------------------------------------------------------ scenes
  // Scenes are split and illustrated automatically. Once the user edits a scene,
  // nothing is re-split without asking, so their changes are never lost.
  let edited = false;
  let lastSplit = "";
  let splitTimer = null;
  let searchRun = 0;

  async function split(auto) {
    showError("");
    const text = $("story-text").value;
    const info = $("split-info");
    if (!text.trim()) { info.textContent = "Hãy dán câu chuyện vào trước."; return; }
    if (!auto && edited && !confirm("Chia lại sẽ bỏ các chỉnh sửa ở từng cảnh. Tiếp tục?")) return;
    info.textContent = "Đang chia cảnh…";
    try {
      const res = await fetch("/api/story/scenes", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, length: $("scene-length").value }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Không chia được cảnh.");
      if (data.title && (!$("title").value.trim() || $("title").dataset.auto === "1")) {
        $("title").value = data.title;
        $("title").dataset.auto = "1";
      }
      lastSplit = text;
      edited = false;
      renderScenes(data.scenes);
      info.textContent = `Đã chia thành ${data.scenes.length} cảnh.`;
      $("split").textContent = "Chia lại";
      findAllImages();
    } catch (err) {
      info.textContent = err.message;
    }
  }

  $("split").addEventListener("click", () => split(false));
  $("title").addEventListener("input", () => { $("title").dataset.auto = "0"; });
  $("scene-length").addEventListener("change", () => { if (sceneList.children.length) split(false); });
  $("story-text").addEventListener("paste", () => setTimeout(() => { if (!edited) split(true); }, 50));
  $("story-text").addEventListener("input", () => {
    clearTimeout(splitTimer);
    splitTimer = setTimeout(() => {
      const text = $("story-text").value;
      if (text === lastSplit || text.trim().length < 20) return;
      if (edited) {
        $("split-info").textContent = "Nội dung đã đổi. Bấm “Chia lại” để chia theo nội dung mới.";
      } else {
        split(true);
      }
    }, 1200);
  });

  function renderScenes(scenes) {
    searchRun += 1; // ignore results of searches started for the previous scenes
    sceneList.innerHTML = "";
    scenes.forEach((scene, i) => {
      const card = document.createElement("div");
      card.className = "scene-card";
      card.innerHTML = `
        <div class="scene-num">${i + 1}</div>
        <div class="scene-body">
          <textarea name="text" rows="3" aria-label="Lời kể cảnh ${i + 1}"></textarea>
          <div class="scene-image">
            <button type="button" class="preview" data-role="preview" title="Đổi ảnh">Chưa có ảnh</button>
            <div class="caption">
              <span class="title" data-role="picked">Đang chờ tìm ảnh…</span>
              <span class="buttons">
                <button type="button" class="small" data-role="browse">Đổi ảnh</button>
                <button type="button" class="small" data-role="research">Tìm lại</button>
              </span>
            </div>
          </div>
          <div class="pair">
            <label>Từ khoá tìm ảnh (tiếng Anh)
              <input type="text" name="query" placeholder="ví dụ: doctor examining child">
            </label>
            <label>Hoặc ảnh riêng
              <input type="file" name="image" accept="image/*,.png,.jpg,.jpeg,.webp">
            </label>
          </div>
          <div class="picker" data-role="picker" hidden></div>
          <label class="scene-audio" hidden>File thu âm của cảnh này
            <input type="file" name="audio" accept="audio/*,.m4a,.mp3,.wav,.aac,.ogg,.flac">
          </label>
        </div>`;
      card.querySelector("textarea").value = scene.text;
      card.querySelector("input[name=query]").value = scene.image_query || "";
      card.querySelector("textarea").addEventListener("input", () => { edited = true; });
      card.querySelector("input[name=query]").addEventListener("change", () => {
        edited = true;
        findImage(card, true);
      });
      card.querySelector("input[name=image]").addEventListener("change", () => {
        edited = true;
        useOwnImage(card);
      });
      const toggle = () => {
        const picker = card.querySelector("[data-role=picker]");
        if (!card._candidates) return findImage(card, true);
        picker.hidden = !picker.hidden;
      };
      card.querySelector("[data-role=browse]").addEventListener("click", toggle);
      card.querySelector("[data-role=preview]").addEventListener("click", toggle);
      card.querySelector("[data-role=research]").addEventListener("click", () => findImage(card, true));
      sceneList.appendChild(card);
    });
    ["scenes-card", "audio-card", "export-card"].forEach((id) => { $(id).hidden = false; });
    applyVoiceMode();
  }

  const cards = () => [...sceneList.querySelectorAll(".scene-card")];
  const choiceOf = (card) => (card.dataset.choice ? JSON.parse(card.dataset.choice) : null);

  function refreshNeeds(card) {
    const own = card.querySelector("input[name=image]").files.length;
    card.classList.toggle("needs-picture", !own && !card.dataset.choice);
  }

  function showChoice(card, cand, how) {
    const preview = card.querySelector("[data-role=preview]");
    const picked = card.querySelector("[data-role=picked]");
    preview.innerHTML = "";
    if (cand) {
      card.dataset.choice = JSON.stringify(cand);
      const img = document.createElement("img");
      img.src = cand.thumb || cand.url;
      img.alt = cand.title;
      preview.appendChild(img);
      picked.textContent = `${how}: ${cand.title} — ${cand.author || "không rõ tác giả"} (${cand.licence})`;
    } else {
      delete card.dataset.choice;
      preview.textContent = "Chưa có ảnh";
    }
    card.querySelectorAll(".thumb").forEach((t) => t.classList.toggle("selected", !!cand && t.dataset.url === cand.url));
    refreshNeeds(card);
  }

  function useOwnImage(card) {
    const file = card.querySelector("input[name=image]").files[0];
    const preview = card.querySelector("[data-role=preview]");
    if (!file) { refreshNeeds(card); return; }
    delete card.dataset.choice;
    preview.innerHTML = "";
    const img = document.createElement("img");
    img.src = URL.createObjectURL(file);
    img.alt = file.name;
    preview.appendChild(img);
    card.querySelector("[data-role=picked]").textContent = `Ảnh riêng của bạn: ${file.name}`;
    card.querySelectorAll(".thumb").forEach((t) => t.classList.remove("selected"));
    refreshNeeds(card);
  }

  function renderPicker(card) {
    const picker = card.querySelector("[data-role=picker]");
    picker.innerHTML = "";
    const current = choiceOf(card);
    card._candidates.forEach((cand) => {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "thumb" + (current && current.url === cand.url ? " selected" : "");
      btn.dataset.url = cand.url;
      btn.title = `${cand.title} — ${cand.author || "không rõ tác giả"} (${cand.licence})`;
      const img = document.createElement("img");
      img.src = cand.thumb || cand.url;
      img.alt = cand.title;
      img.loading = "lazy";
      btn.appendChild(img);
      btn.addEventListener("click", () => {
        edited = true;
        card.querySelector("input[name=image]").value = "";
        showChoice(card, cand, "Bạn chọn");
      });
      picker.appendChild(btn);
    });
  }

  // Search one scene; when ``open`` the candidates are shown so the user can pick.
  async function findImage(card, open) {
    const run = searchRun;
    const query = card.querySelector("input[name=query]").value.trim();
    const picked = card.querySelector("[data-role=picked]");
    const picker = card.querySelector("[data-role=picker]");
    if (card.querySelector("input[name=image]").files.length && !open) return;
    if (!query) {
      picked.textContent = "Không nhận ra được gì để tìm ảnh: hãy nhập từ khoá tiếng Anh hoặc chọn ảnh riêng.";
      showChoice(card, null);
      return;
    }
    picked.textContent = "Đang tìm ảnh…";
    try {
      const res = await fetch(`/api/story/images?q=${encodeURIComponent(query)}`);
      const data = await res.json();
      if (run !== searchRun) return;
      if (!res.ok) throw new Error(data.error || "Không tìm được ảnh.");
      card._candidates = data.candidates;
      renderPicker(card);
      if (open) picker.hidden = false;
      // Prefer a picture no other scene uses.
      const taken = new Set(cards().filter((c) => c !== card).map(choiceOf).filter(Boolean).map((c) => c.url));
      const pick = data.candidates.find((c) => !taken.has(c.url)) || data.candidates[0];
      if (!pick) {
        picked.textContent = "Không có ảnh phù hợp. Thử từ khoá khác, hoặc chọn ảnh riêng.";
        showChoice(card, null);
        return;
      }
      if (!card.querySelector("input[name=image]").files.length) showChoice(card, pick, "Ảnh tự chọn");
    } catch (err) {
      if (run === searchRun) picked.textContent = err.message;
    }
  }

  // Illustrate every scene, two searches at a time.
  async function findAllImages() {
    const info = $("image-info");
    const queue = cards();
    const total = queue.length;
    const run = searchRun;
    let done = 0;
    info.textContent = `Đang tự tìm ảnh cho ${total} cảnh…`;
    const worker = async () => {
      while (queue.length) {
        const card = queue.shift();
        await findImage(card, false);
        done += 1;
        if (run === searchRun) info.textContent = `Đang tự tìm ảnh: ${done}/${total} cảnh…`;
      }
    };
    await Promise.all([worker(), worker()]);  // the server sends archive requests one by one
    if (run !== searchRun) return;
    const missing = cards().filter((c) => c.classList.contains("needs-picture")).length;
    info.textContent = missing
      ? `Đã chọn ảnh cho ${total - missing}/${total} cảnh. ${missing} cảnh cần từ khoá khác hoặc ảnh riêng (viền đỏ).`
      : `Đã tự chọn ảnh cho cả ${total} cảnh. Hãy xem qua và đổi ảnh nào chưa hợp.`;
  }

  function applyVoiceMode() {
    const mode = voiceMode();
    $("narration-field").hidden = mode !== "single";
    sceneList.querySelectorAll(".scene-audio").forEach((el) => { el.hidden = mode !== "per_scene"; });
  }
  form.querySelectorAll("input[name=voice_mode]").forEach((r) => r.addEventListener("change", applyVoiceMode));

  // ------------------------------------------------------------------ submit
  function tooBig(file, mb) { return file && file.size > mb * 1024 * 1024; }

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    showError("");
    const cards = [...sceneList.querySelectorAll(".scene-card")];
    if (!cards.length) return showError("Hãy chia câu chuyện thành cảnh trước.");
    const mode = voiceMode();
    const data = new FormData();
    const scenes = [];
    for (const [i, card] of cards.entries()) {
      const text = card.querySelector("textarea").value.trim();
      const query = card.querySelector("input[name=query]").value.trim();
      const image = card.querySelector("input[name=image]").files[0];
      const audio = card.querySelector("input[name=audio]").files[0];
      if (!text) return showError(`Cảnh ${i + 1} chưa có lời kể.`);
      const choice = card.dataset.choice ? JSON.parse(card.dataset.choice) : null;
      if (!query && !image && !choice) return showError(`Cảnh ${i + 1} cần từ khoá tìm ảnh hoặc một ảnh riêng.`);
      if (tooBig(image, cfg.maxUploadMb)) return showError(`Ảnh của cảnh ${i + 1} lớn hơn ${cfg.maxUploadMb} MB.`);
      if (image) data.set(`scene_${i}_image`, image);
      if (mode === "per_scene") {
        if (!audio) return showError(`Cảnh ${i + 1} chưa có file thu âm.`);
        if (tooBig(audio, cfg.maxAudioMb)) return showError(`File thu âm cảnh ${i + 1} lớn hơn ${cfg.maxAudioMb} MB.`);
        data.set(`scene_${i}_audio`, audio);
      }
      scenes.push({ text, image_query: query, image_choice: image ? null : choice });
    }
    if (mode === "single") {
      const narration = $("narration").files[0];
      if (!narration) return showError("Hãy chọn file thu âm cho cả câu chuyện (hoặc chọn kiểu lời kể khác).");
      if (tooBig(narration, cfg.maxAudioMb)) return showError(`File thu âm lớn hơn ${cfg.maxAudioMb} MB.`);
      data.set("narration", narration);
    }
    const music = $("music").files[0];
    if (music) data.set("music", music);
    data.set("script", JSON.stringify({
      title: $("title").value.trim(), scenes, music_volume: parseFloat($("music-volume").value),
    }));
    data.set("resolution", $("resolution").value);
    data.set("preview_voice", mode === "preview" ? "true" : "false");

    generate.disabled = true;
    resultCard.hidden = true;
    setStatus({ state: "uploading", progress: 0, message: "Đang tải lên…" });
    try {
      const res = await fetch("/api/story", { method: "POST", body: data });
      const body = await res.json().catch(() => ({ error: `Lỗi máy chủ (${res.status}).` }));
      if (!res.ok) throw new Error(body.error || `Lỗi máy chủ (${res.status}).`);
      setStatus(body);
      poll(body.id);
    } catch (err) {
      statusCard.hidden = true;
      generate.disabled = false;
      showError(err.message);
    }
  });

  // ------------------------------------------------------------------ progress
  const PHRASES = [
    [/^Preparing picture (\d+)\/(\d+)/, "Đang tìm và chuyển ảnh $1/$2 thành tranh chì"],
    [/^Preparing narration/, "Đang chuẩn bị lời kể"],
    [/^Scene (\d+)\/(\d+): planning the drawing/, "Cảnh $1/$2: đang lên kế hoạch nét vẽ"],
    [/^Scene (\d+)\/(\d+): drawing/, "Cảnh $1/$2: đang vẽ"],
    [/^Scene (\d+)\/(\d+): narration/, "Cảnh $1/$2: giữ tranh theo lời kể"],
    [/^Encoding video/, "Đang mã hoá video"],
    [/^Mixing sound/, "Đang ghép âm thanh"],
    [/^Waiting in queue/, "Đang chờ đến lượt"],
    [/^Starting/, "Bắt đầu"],
    [/^Done/, "Xong"],
  ];
  function vi(message) {
    for (const [re, text] of PHRASES) if (re.test(message || "")) return message.replace(re, text);
    return message || "";
  }

  function poll(id) {
    clearTimeout(pollTimer);
    pollTimer = setTimeout(async () => {
      try {
        const res = await fetch(`/api/jobs/${id}`, { cache: "no-store" });
        const job = await res.json();
        if (!res.ok) throw new Error(job.error || "Mất kết nối với công việc đang chạy.");
        setStatus(job);
        if (job.state === "done") return showResult(job);
        if (job.state === "error") { generate.disabled = false; return; }
        poll(id);
      } catch (err) {
        setStatus({ state: "error", progress: 0, error: err.message });
        generate.disabled = false;
      }
    }, 800);
  }

  function setStatus(job) {
    statusCard.hidden = false;
    const pct = Math.round((job.progress || 0) * 100);
    const isError = job.state === "error";
    statusCard.classList.toggle("error", isError);
    $("status-title").textContent = isError ? "Không tạo được video" : job.state === "done" ? "Hoàn tất"
      : job.state === "queued" ? "Đang chờ" : "Đang tạo video";
    $("status-detail").textContent = isError ? job.error : vi(job.message);
    $("bar").style.width = `${isError ? 100 : pct}%`;
    $("progressbar").setAttribute("aria-valuenow", String(pct));
    $("pct").textContent = isError ? "" : `${pct}%`;
    $("eta").textContent = job.eta != null ? `còn khoảng ${Math.ceil(job.eta / 60)} phút`
      : job.elapsed ? `đã chạy ${Math.round(job.elapsed)} giây` : "";
  }

  function showResult(job) {
    generate.disabled = false;
    const info = job.info || {};
    $("video").src = `/api/jobs/${job.id}/video?v=${Date.now()}`;
    $("download").href = `/api/jobs/${job.id}/download`;
    const credits = $("credits");
    credits.hidden = !job.has_credits;
    credits.href = `/api/jobs/${job.id}/credits`;
    const mins = Math.floor((info.duration || 0) / 60), secs = Math.round((info.duration || 0) % 60);
    const size = info.file_size ? ` · ${(info.file_size / 1048576).toFixed(1)} MB` : "";
    $("result-summary").textContent =
      `${mins}:${String(secs).padStart(2, "0")} · ${info.scenes} cảnh · ${job.config.width}×${job.config.height}${size}` +
      ` · tạo trong ${Math.round(info.render_seconds || job.elapsed)} giây`;
    const notes = $("notes");
    notes.innerHTML = "";
    const add = (text, cls) => {
      const li = document.createElement("li");
      li.textContent = text;
      if (cls) li.className = cls;
      notes.appendChild(li);
    };
    (job.warnings || []).forEach((w) => add(w, "warn"));
    (info.credits || []).forEach((c, i) => add(`Ảnh ${i + 1}: ${c}`));
    resultCard.hidden = false;
    resultCard.scrollIntoView({ behavior: "smooth", block: "start" });
  }
})();
