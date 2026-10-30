# Pencil Timelapse Studio

Create a vertical (9:16) "pencil drawing timelapse" video from a finished pencil portrait:

- **End image** (required): the completed pencil portrait.
- **Start** (choose one):
  - **Blank white paper** (default): the portrait is drawn on an empty sheet that matches the
    paper of the end image.
  - **Your own start image**: a drawing that is already partly finished. The video continues from it
    instead of starting over.

A simulated pencil moves along the strokes as they appear (it can be switched off). The video
draws the portrait in five stages of real strokes, following an artist's working order (construction,
contours, features, shading, finishing), and ends by holding the exact end image. It works
for any portrait subject (real people, historical figures, fictional characters) because it only
analyses the images you provide.

> **Important limitation: this is a simulation.**
> The drawing order, strokes, hatching, construction lines and the moving pencil are produced by
> deterministic image processing (OpenCV + NumPy). The software plans a *plausible* order from the finished picture; it
> cannot know, and does not reconstruct, how the artist actually drew it. The result shows
> simulated pencil effects, not real hand animation. No AI video model, generative model, paid API
> or cloud service is used, and everything runs locally.

---

## Features

- Web interface: upload and preview the images, choose blank paper or your own start image, choose
  duration / FPS / resolution, watch progress, preview the result and download the MP4.
- A simulated pencil whose tip follows the line or area being drawn. It is rendered with lit
  hexagonal facets, glossy lacquer, a sharpened cedar cone and a graphite point, in black (default),
  yellow or natural wood. Its shadow is crisp at the tip and softer further up, and it has
  camera-like motion blur. It lifts slightly between strokes and leaves the frame before the final
  hold.
- Command-line interface and a small Python API.
- 1080 × 1920 at 30 FPS by default; any even resolution from 160 px up to 2160 × 3840.
- Deterministic: the same inputs and settings always give the same video.
- Continues from existing artwork in the start image instead of restarting the drawing.
- Validates inputs with clear error messages, reports progress, and cleans up temporary files.

## Requirements

- Python 3.9 or newer (tested with 3.9 and 3.11)
- FFmpeg with an H.264 encoder. You normally don't need to install it yourself: the
  `imageio-ffmpeg` dependency ships a static FFmpeg binary. If an `ffmpeg` is already on your
  `PATH`, that one is used instead. To force a specific binary, set `PENCIL_FFMPEG=/path/to/ffmpeg`.
- About 1.2 GB of free RAM for 1080 × 1920 output (about 3 GB for 2160 × 3840).

## Installation

```bash
cd video                       # the project folder
python3 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` pins OpenCV to 4.x on purpose: OpenCV 5 no longer bundles the Haar cascades
used for face and eye detection. If you install OpenCV 5 anyway, the app still works and falls back
to a detail-density focus estimate.

## Usage

### Web interface

```bash
python app.py                  # then open http://127.0.0.1:8000
python app.py --port 9000      # other port; --host 0.0.0.0 to allow other machines on your network
```

1. Under **Start**, keep **Blank white paper** (default) or switch to **My start image** and upload a
   partly finished drawing.
2. Drop or choose the **End image**. The previews use the output aspect ratio, so you can see how
   the picture will sit on the vertical canvas.
3. Set the duration, frame rate, resolution and final hold, and choose whether to **show a pencil
   drawing the strokes**. The timeline bar shows when each stage happens. Under
   **Advanced options** you can change alignment, paper-tone matching, construction lines, the
   strength of the simulated stroke and paper textures, the pencil look (black / yellow / natural
   wood) and size, encoder quality (CRF), encoder speed and the texture seed.
4. Click **Generate video**. Progress shows the current stage and an estimated time remaining.
5. Preview the result in the page and click **Download MP4**.

Renders run one at a time. Others wait in a queue and show their position. Uploaded images are
deleted as soon as a render finishes, and videos are deleted after 6 hours. Working files live in
`./runtime` (change it with `--work-dir` or `PENCIL_WORK_DIR`).

### Command line

```bash
# One image: drawn on blank paper with a pencil (1080x1920, 30 fps, 15 s)
python -m pencil_timelapse.cli end.png -o drawing.mp4

# Two images: continue from a partly finished drawing
python -m pencil_timelapse.cli start.png end.png -o drawing.mp4

# Yellow pencil instead of the default black one
python -m pencil_timelapse.cli end.png -o drawing.mp4 --pencil-style yellow

# 20 s at 24 fps, 720x1280, no pencil, no construction lines
python -m pencil_timelapse.cli end.png -o drawing.mp4 \
    --duration 20 --fps 24 --size 720x1280 --no-pencil --no-construction

# Generate synthetic sample images to try it out
python -m pencil_timelapse.cli --make-samples samples/
python -m pencil_timelapse.cli samples/sample_end.png -o demo.mp4
```

Run `python -m pencil_timelapse.cli --help` to see every option.

### Python API

```python
from pencil_timelapse import RenderConfig, render_video

result = render_video(
    None, "end.png", "drawing.mp4",  # None = blank paper (start_mode="blank", the default)
    RenderConfig(duration=15, fps=30, width=1080, height=1920, show_pencil=True),
    progress=lambda fraction, message: print(f"{fraction:.0%} {message}"),
)
print(result.warnings, result.info["focus"])
```

## Story videos (3–5 minutes)

Turn a narrated story (education, health, …) into a horizontal 16:9 video in which every scene is
a picture drawn in pencil, with subtitles, a title card and image credits.

**In the web interface**, open the **Video câu chuyện** tab (the portrait tool stays on the first tab):

1. Paste the story. It is **split into scenes automatically**: no button is needed, and no blank
   lines are required. Paragraphs stay together when they fit the chosen scene length (short,
   medium or long, about 8, 12 or 18 seconds of narration). Longer paragraphs, or a story pasted as
   one block, are cut at sentence ends, and very short pieces join the previous scene. A short first
   line becomes the title. After you edit a scene, nothing is re-split until you click **Chia lại**.
2. **Every scene is illustrated automatically.** English keywords are suggested offline from the
   Vietnamese text (education, health, family and daily-life vocabulary). Scenes that name nothing
   photographable use the story's main subject. The best free-licensed photo is picked and
   previewed, and no two scenes get the same photo. Click **Đổi ảnh** to choose another candidate,
   edit the keywords to search again, or upload your own picture. Automatic picks prefer photos
   whose title shares at least two keywords, and skip buildings, monuments, famous people and
   animals unless asked for, but they are a starting point: look through them before rendering.
3. Upload the narration: one recording for the whole story, or one per scene. You can also
   preview with the Mac voice or render without narration. Optionally add background music.
4. Click **Tạo video**, follow the progress, then watch and download the MP4 and the image
   credits.

**From the command line:**

```bash
# 1. Plain-text story -> script skeleton (one scene per paragraph; a short first line becomes the title)
python -m pencil_timelapse.story my_story.txt --template -o my_story.json
# 2. Fill in "image_query" for each scene (English keywords), add your recording, then render:
python -m pencil_timelapse.story my_story.json -o my_story.mp4
# Draft without a recording: read the scenes with the macOS Vietnamese voice
python -m pencil_timelapse.story my_story.json -o draft.mp4 --preview-voice Linh
```

A script looks like this (see `examples/story_example.json`; paths are relative to the script):

```json
{
  "title": "Thói quen rửa tay của An",
  "narration": "recording.m4a",
  "music": "background.mp3",
  "music_volume": 0.12,
  "scenes": [
    {"text": "An là một cậu bé học lớp ba...", "image_query": "children playing schoolyard"},
    {"text": "Bác sĩ khám cho An...", "image_query": "doctor examining child | pediatrician"},
    {"text": "...", "image": "my_photo.jpg", "audio": "scene3.m4a"}
  ]
}
```

- **Pictures.** `image` uses your own photo, `image_choice` a search result picked in the web
  interface. Otherwise `image_query` searches Wikimedia Commons
  (and Openverse when reachable) for a free-licensed photo; alternatives can be separated with
  `|`. Photos whose title names the subject and that show people are preferred. Book scans, signs,
  diagrams, animals (unless asked for) and graphic content are skipped. Keyword search on photo
  archives is noisy, so check the pictures, or pick them in the web interface. Downloads are cached in
  `story_images/` next to the script together with author and licence, so you can review them. To
  replace a picture, set `image` for that scene. Photos are converted into graphite drawings
  (colour-dodge contours plus compressed tones) and drawn by the pencil animation.
- **Narration.** `narration` is one recording for the whole story: scenes are matched to it by
  text length, and each scene change moves to the nearest natural pause. Leave about one second
  of silence between paragraphs when recording. Alternatively, give a scene its own `audio`, and
  that scene lasts as long as its clip. Any format FFmpeg reads works (m4a, mp3, wav, …).
- **Timing.** A 3.5 s title card, then each scene: the picture is drawn during roughly the first
  60 % of the scene (4.5–14 s), held while the narration continues, then fades to a fresh page.
  The credits card comes at the end. Subtitles follow the narration.
- **Credits.** Image titles, authors and licences are shown at the end and written to
  `<video>.credits.txt` with full links. CC BY and CC BY-SA images require this attribution.
  Please also check pictures of identifiable people before publishing.
- **Speed.** About one second of rendering per second of video on an Apple Silicon Mac, so a
  4-minute story takes about 4 minutes. Image search needs an internet connection; everything else
  runs offline.

## Football players

The **Football players** tab (all text in English) draws footballers in pencil one after another
into a single picture, with a hand-lettered title and each player's name under their portrait.

1. **Players.** Upload your own photos (one per player, showing the whole head with some space
   above the hair; names come from the file names and can be edited), and/or find a country's
   famous players on Wikidata (ranked by how many Wikipedia editions have an article about them;
   the best free-licensed portrait on Wikimedia Commons is picked automatically). Tick up to 12 and
   put them in order with the arrows.
2. **Video.** Title, frame (16:9, 9:16 or square), background, drawing time per player, and an
   optional sound file (music or voice). With a sound file, the video can be made exactly as long as
   the sound: the drawing time is then set automatically.
3. You get the MP4, the finished picture as a PNG, and the photo credits when Commons photos were
   used.

**Picture.** By default the **whole photo** is redrawn in pencil: nothing is cut or covered, and if
the photo's shape differs from the 4:5 frame, the strips beside it continue the photo's edges
softly. (A player in a full-body photo therefore stays as small as in the photo.) Two other
framings are available: **head and shoulders with the background shaded in pencil** (the player
is separated from the background by colour with OpenCV GrabCut, and hair by sharpness, since
hair is in focus while the crowd behind is usually blurred), and **head and shoulders with the
photo's background**. Both keep the whole head, hair and ears, at the same scale for every player.

**Graphite look.** The drawing shows the pencil itself: hatching strokes that follow the hair and
folds and run diagonally in flat areas, crossed strokes in the darks, white paper tooth showing
through light strokes, outlines of varying pressure, and a cool grey graphite instead of black.

**Keeping viewers watching** (each can be switched off):

- *Intro* (5 s): the finished group picture first, a big headline ("PORTUGAL LEGENDS", editable)
  with "6 PLAYERS · 1 PENCIL", an eraser sweep back to blank paper, then "Let's draw!".
- *Who's next?* (1.4 s before each player): a mystery silhouette with a big question mark and
  "3 / 6" in the empty frame.
- *Highlight* (2.6 s after each player): the camera pushes in on the new portrait, a glint runs over
  the graphite and a numbered badge pops on.
- *Ending* (7 s): "Which team should I draw next?" (editable), "Tell me in the comments!", and
  Like / Subscribe / bell buttons that a cursor clicks one after another.
- *Sound effects*: whoosh, pop, ding, clicks, synthesised locally and mixed with your sound.

Timing: intro 5 s + title 4 s + per player (1.4 + drawing time + 2.6 s) + 2 s hold + 7 s ending.
Six players at 8 s each make a 90 s video, rendered in about a minute on an Apple Silicon Mac.

```bash
# Your own photos ("Name=file", or the file name is used), with music fitted to its length
python -m pencil_timelapse.players --photos "Cristiano Ronaldo=cr7.jpg" "Pepe=pepe.png" \
    --title PORTUGAL --hook "PORTUGAL LEGENDS" --audio music.mp3 -o portugal.mp4

# A country's famous players, found online
python -m pencil_timelapse.players Portugal --list                    # who would be drawn
python -m pencil_timelapse.players Brazil -o brazil.mp4 --count 9 --size 1080x1920 --seconds 8

# Options: --background shade|photo (head and shoulders), --keep-timing (do not fit to the audio), --question "...",
#          --no-intro, --no-highlights, --no-outro, --no-sound-effects
```

Commons photos are free-licensed (CC BY, CC BY-SA, public domain…); their authors and licences are
shown at the bottom of the ending and saved to `<video>.credits.txt`. Make sure you may use your own
photos and music, and note that using a real person's likeness commercially (advertising,
merchandise) may need their permission.

## Animals

The **Animals** tab makes a video about animals from your photos and facts. Each animal is drawn in
pencil (the whole photo, nothing cut), then its name and facts are written beside it, one by one.
The video moves on to the next animal only when all of its facts have been shown: each fact
stays long enough to read. A recording added for an animal starts when the pencil starts drawing
that animal; the video stays with it until the recording has ended, and its facts are spread over
what is left of the recording once the drawing is done.

1. **Animals.** Add one photo per animal (names come from the file names and can be edited), write
   one fact per line, and optionally add a recording per animal. Reorder with the arrows.
2. **Video.** Title, frame (16:9 with the text beside the drawing, or 9:16 / square with the text
   below), drawing time per animal and optional music (quieter under recordings).

Each scene is staged like a real drawing session:

- With 8 seconds or more of drawing time, the **animal is drawn first** on blank paper (it is
  separated from the background with GrabCut) while the **camera moves in and follows the
  pencil**, then pulls back to show it; then the scenery is drawn around it.
- A **pencil writes** the name, underlines it, and writes each fact line by line next to a
  numbered badge; **numbers** ("8 kilometres", "40,000") get a **highlighter** stroke; the pencil
  leaves the page while you read.
- The pencil is **heard scratching** the paper, following its real movements (silent when lifted).

Keeping viewers watching (each can be switched off):

- *Guess the animal?* while the pencil draws; the name is written when the drawing is finished.
- *Intro* (5 s): a blurred glimpse of all the drawings, a big headline and "Let's draw!".
- An eraser sweep from one animal to the next, sound effects, and at the end the **gallery** of all
  the drawings followed by "Which animal should I draw next?" with Like / Subscribe / bell buttons.

Timing per animal: drawing time + 1.7 s for the name + each fact (the time to write it and read
it: about 3 s for a short fact, more for longer ones) or the recording + 0.9 s, then a 0.8 s sweep. Text in Vietnamese or other languages
the handwriting font lacks is written in a plain font.

```bash
python -m pencil_timelapse.animals animals.json -o animals.mp4 --music music.mp3
```

where `animals.json` is `{"title": "Amazing Animals", "animals": [{"name": "Lion", "photo":
"lion.jpg", "facts": ["Lions live in prides.", "..."], "voice": "lion.m4a"}]}` (paths relative to the
file; `voice` optional). Options: `--size 1080x1920`, `--seconds 8`, `--no-guess`, `--hook`,
`--question`, `--no-intro`, `--no-outro`, `--no-sound-effects`.

## Travel

The **Travel** tab makes a video about famous places around the world. Each place is drawn in
pencil from a photo (or from the first picture of a video clip) while its facts are written beside
it, exactly as in the Animals tab. When the drawing and the facts are done, the drawing **comes to
life**: colour washes over the graphite, the real photo or clip grows to fill the screen with the
place's name, its location, its typed coordinates, a map card with a pin and a red **passport
stamp**; then a **paper plane** flies across and the next page follows it in.

1. **Places.** Either let **AI write the script**, or add places by hand.
   - *Write with AI*: choose Gemini or GPT, paste your API key (kept in this browser only; or set
     `GEMINI_API_KEY` / `OPENAI_API_KEY` before starting the server), and type a theme ("The 7 new
     wonders of the world") with the number of places, or list the places. The AI writes each
     place's name, "city, country", coordinates, facts and photo search words, plus the video title,
     the intro headline and the closing question. The defaults are `gemini-flash-latest` and
     `gpt-5-mini`; another model can be typed in. **Check the facts before publishing.**
   - *Pictures*: every place is searched automatically. Without keys the photos come from
     **Openverse** (Creative Commons photos, mostly Flickr; only licences that allow commercial use
     and changes; Wikimedia is excluded; a search can take up to a minute). With a free **Pexels**
     or **Pixabay** API key you also get **video clips** (shown first, marked ▶). Click a thumbnail
     to choose it, or upload your own photo or video. A recording per place is optional.
2. **Video.** Title, frame, drawing time per place, how long each place stays in colour (default
   3.2 s), music and the viewer hooks.

Keeping viewers watching (each can be switched off):

- *Intro* (6 s): a **pencil world map** (Natural Earth coastlines, public domain) with the headline;
  every destination is pinned in turn and the route is flown by a paper plane; "Let's go!".
- *Guess the place?* while the pencil draws; the name and "city, country" appear when it is done.
- At the end, a wall of **postcards** of all the drawings on the faded map, then "Which place
  should I draw next?" with Like / Subscribe / bell buttons and the photo credits.

The credits file (download it with the video) lists every photo and clip with its author, licence
and source: paste it into the video description, since Creative Commons photos must be credited.

```bash
python -m pencil_timelapse.travel trip.json -o trip.mp4 --music music.mp3
```

where `trip.json` is `{"title": "Wonders of the World", "places": [{"name": "Eiffel Tower",
"media": "eiffel.jpg", "location": "Paris, France", "country": "France", "lat": 48.8584, "lon":
2.2945, "facts": ["..."], "voice": "eiffel.m4a", "credit": "Photo by ..."}]}` (`media` may be a video
clip; paths relative to the file). Options: `--size`, `--seconds`, `--reveal`, `--no-guess`,
`--hook`, `--question`, `--no-intro`, `--no-outro`, `--no-sound-effects`.

## Pet Diary

The **Pet Diary** tab makes "a day in the life" videos told by the pet itself, in the first person.
Each moment of the day (a photo or a video clip) is drawn in a pencil sketchbook while the pet's
**thought** is written in a bubble and read aloud in the pet's voice (a macOS voice, pitched by
kind); the time is written on top with a little clock, and a **paw print walks along a day bar**
from sunrise to bedtime. When the drawing is done it **comes to life**: colour washes over the
graphite, the real clip grows to the whole screen as the pet's camera ("BISCUIT CAM", REC dot, the
time) and the pet's **punchline** appears in captions word by word; then paw prints walk across to
the next page.

Built from what keeps viewers watching pet videos (an animal "character" with a voice, a small
story from setup to payoff, a reason to stay until the end):

- *Cold open*: the day's big moment plays, freezes with a record scratch and turns to pencil; the
  pet asks "Wait... how did I end up here?" and "Let me explain.", and the day rewinds.
- *What happens next?* before the big moment: two to four answers, a 3-2-1 countdown, the right one
  turns green; then the moment is drawn and plays.
- *Day bar*: viewers always see how much of the day is left.
- *Ending*: the sketchbook of the whole day, then a question for the comments with Like /
  Subscribe / bell.
- Built-in music (happy or calm) is turned down whenever the pet speaks. In 9:16 (Shorts, the
  default) the text stays above the bottom 16 % of the screen, where Shorts shows its own buttons.

Moments come from a **ready-made diary** (Biscuit the corgi, Pepper the cat), from **AI** (GPT or
Gemini writes the times, titles, thoughts, punchlines, search words, teaser line and the guess; the
writer is asked for original jokes only) or are typed by hand. Pictures: **your own pet's photos and
clips work best** (one pet all day long); otherwise free photos from Openverse and, with a Pexels or
Pixabay key, video clips, all under licences that allow changes and reuse. Do not use other
creators' videos: re-uploading them, even redrawn, breaks their copyright and YouTube's reused
content policy.

```bash
python -m pencil_timelapse.pets diary.json -o diary.mp4 --size 1080x1920
python -m pencil_timelapse.pets --sample biscuit -o biscuit.mp4   # finds the photos online
```

where `diary.json` is `{"pet": "Biscuit", "kind": "dog", "title": "...", "hook": "...", "guess":
["...", "..."], "answer": "...", "question": "...", "moments": [{"time": "7:00 AM", "title":
"Breakfast", "thought": "...", "punch": "...", "media": "breakfast.mp4", "best": false, "credit":
"..."}]}` (paths relative to the file). Options: `--size`, `--seconds` (drawing, default 8),
`--reveal` (real clip, default 3), `--music`, `--tune happy|calm|none`, `--voice`, `--pitch`,
`--no-guess`, `--no-cold-open`, `--no-outro`, `--no-sound-effects`.

## Cartoon

The **Cartoon** tab makes "Peeko & Friends", a cartoon series for young children (about 3 to 6),
or stories starring your own character. Everything is made by the program: the characters, the
scenery, the music and the sound effects, so it is free to publish.

**The cast.** Four friends live in Sunny Garden. They are drawn frame by frame, so their
rubber-hose arms and legs move, they blink and look around, their mouths (or Kiko's beak) move with
their voices, they turn towards where they go, and each has its own voice and pitch.

| Character | Who | Voice |
|---|---|---|
| **Peeko** | a lilac sprout in yellow rain boots who wants to learn everything; its leaves show how it feels and spin like a propeller when it flies; a flower blooms on its sprout for every lesson | Samantha, pitched up |
| **Lulu** | a shy white bunny in a yellow beret who loves painting ("Oh, how pretty!") | Karen, pitched up |
| **Kiko** | a chatty bluebird in round glasses: "Did you know...?", and often a little bit wrong | Tessa, high |
| **Bobo** | a big, gentle bear cub with a red bow tie who is always hungry | Rishi, pitched down |

Your own character from a picture can star instead of Peeko (a PNG with a transparent background
works best; a plain background is removed automatically). It is cut out and moved as a whole,
like a paper puppet; the friends can still join it.

**Season 1: Sunny Garden** (12 episodes, about 2½ to 3 minutes each): Hello, Sunny Garden! ·
Bobo's Big Breakfast (healthy food) · Kiko Counts the Stars (counting to ten) · Lulu's Missing
Colors (mixing colours) · Brush, Brush, Bobo! · Bubble Trouble (washing hands) · Shapes in the Sky ·
My Turn, Your Turn (sharing) · Peeko Feels Grumpy (anger, calming down) · Seeds and Sunshine (how
plants grow) · Safe Steps to the Park (crossing the street) · The Sunny Garden Festival (a quiz about
everything, the finale). Each episode:

1. a **teaser** (a funny or surprising moment), then the show's **opening**: the logo's letters drop
   in on the beat of the theme tune, the friends pop up with their names, everyone waves;
2. the episode's **title**, read out;
3. the story, with a problem and a happy ending, and two or three moments where children join in:
   **counting** out loud (things pop up on a chalkboard one by one while the narrator says the
   numbers), a **picture quiz** with a 3-2-1 countdown (the right picture lights up), saying words;
4. "The End": the **lesson** read out, Peeko's newest **flower** joining the season's collection
   ("4 of 12"), and **next time**, read out; then like, subscribe and a question for comments.

Eleven shorter **mini episodes** (Peeko mostly alone, about a minute and a half) are there too.

**Writing your own.** Load an episode and change it, let Gemini or GPT write one from an idea (the
same keys as the Travel tab; the AI is told who everyone is), or write the scenes yourself. A scene
has a place (garden, meadow, forest, playground, beach, city, snow, desert, mountain, room,
kitchen, space, underwater), a time of day, weather, a narration, the hero's actions, the friends
in the scene ("Lulu, sad Bobo"), the conversation, one line each:

```
Lulu: Good morning! (wave)
I'm so hungry!            <- no name: the hero
Narrator: Later that day...
```

an effect, something for the hero to hold (30 props drawn by the program), and a chalkboard with
words (colour names in their colour), a picture, things to count, or a picture quiz. Actions: walk,
run, enter, exit, jump, dance, wave, clap, point, happy, sad, angry, scared, surprised, think,
sleep, yawn, float, spin, look, brush, wash, eat, read.

The music is a cheerful tune (plucked ukulele, bass, bells, light drums) or a calm lullaby, made by
the program and turned down while anyone speaks; you can use your own music instead. The scenery
has sunlight on the edges of things, haze in the distance, drifting pollen and grass in front.

```bash
python -m pencil_timelapse.cartoon --episode s1e04-lulus-missing-colors -o colors.mp4
python -m pencil_timelapse.cartoon --season -o season1/                 # all 12 episodes
python -m pencil_timelapse.cartoon --episode shapes --size 1080x1920 -o shapes.mp4  # vertical
python -m pencil_timelapse.cartoon --sample --character pip.png --name Pip -o pip.mp4
python -m pencil_timelapse.cartoon story.json -o story.mp4
```

where `story.json` is `{"character": "pip.png" (leave it out for Peeko), "name", "title", "lesson",
"next", "hook": true, "music": "happy", "scenes": [{"setting": "kitchen", "time": "day",
"narration": "...", "actions": ["think"], "cast": ["Bobo"], "lines": "Bobo: Snack time! (eat)",
"board": "Which one is healthy?", "quiz": ["candy", "carrot"], "answer": "carrot", "holds": "",
"item": "", "count": 0}]}`. Options: `--size`, `--narrator`, `--voice`, `--pitch`, `--music`,
`--no-series`, `--no-intro`, `--no-outro`, `--no-sound-effects`. Without the macOS speech
synthesiser the story is told with subtitles only.

## Input guidelines

| | |
|---|---|
| Formats | PNG, JPEG, WEBP, BMP, TIFF (8- or 16-bit). Transparency is placed on white. EXIF rotation is applied. |
| Size | Shorter side at least 128 px, at most 80 megapixels; at most 40 MB per upload in the web UI. |
| Framing | For best results the start and end images show the same sheet with the same framing (for example two photos or scans of the same drawing). Small shifts, rotations and scale differences are corrected automatically. |
| Start image | Optional. A sketch or partly finished drawing of the same sheet. Marks the end image doesn't have (for example guide lines) are erased during the shading stage. Don't use a photo of the person here: the video would morph the photo into the drawing (the app warns about this). |
| End image | The finished artwork. The video ends on exactly this image. A colour image works too: it is shown in graphite grey while being drawn and returns to its own colours in the final stage. |

## How it works

### Timeline (default 15 s)

| Time | Stage | What appears |
|---|---|---|
| 0–3 s | Construction strokes | The pencil comes onto the page and makes faint proportion ticks, then very light, loose guide lines. With a detected face: head oval, facial centreline, eye line, jaw, nose/mouth lines and neck. Otherwise: centreline and eye line. Plus straight block-in lines for the big shapes. Many guides are gone over a second time (exploratory, overlapping strokes) |
| 3–6 s | Contour strokes | Delicate contour lines (forehead, cheeks, jaw, hairline, shoulders), then the facial structure (eyes, nose, lips, ears), face first, with varying pencil pressure |
| 6–9 s | Feature strokes | Sharp lines in the eyes, nose and lips; short radial iris strokes when the eyes are detected; small hatching that gives the features their tone; long hair strands that follow the hair's flow |
| 9–12 s | Shading strokes | Area by area: parallel hatching, cross-hatching with more pressure in the darker areas, then broad tonal strokes for smooth transitions. Paper-white highlights are never drawn over |
| 12–15 s | Finishing strokes | Fine hair strands and the deepest shadows. Small highlights that the shading covered are lifted out again (as with an eraser) and the guide lines are erased. The pencil moves away and the finished artwork is held (from about 14.1 s) |

All marks are made by strokes, and graphite only appears where the pencil tip passes. The stages
keep their equal proportions for any duration. The optional extra hold (0 s by default) adds more
seconds on the exact end image after the finishing stage.

### Pipeline

1. **Normalize.** Both images are fitted onto the output canvas with their aspect ratio preserved
   and centred. Padding is filled with the estimated paper colour and a subtle procedural paper
   texture. The start image is registered to the end image (ORB features + RANSAC, similarity
   transform; only small, well-supported corrections are accepted), and its paper tone is matched.
2. **Graphite representation.** The end image is converted to neutral graphite tones over its paper
   colour. Line-integral convolution of noise along the stroke-direction field (structure-tensor
   edge tangents, with a diagonal default in flat areas) adds subtle simulated pencil streaks
   where new graphite goes.
3. **What is left to draw.** A per-pixel "ink still needed" map compares the start image with the
   graphite target. Everything that follows uses this map, so artwork that already exists in the
   start image is kept and never redrawn.
4. **Lines.** At three scales (coarse outlines, medium features, fine lines), a contrast-normalized
   ridge + edge detector with hysteresis finds lines. These are thinned (Zhang–Suen) and traced into
   polylines. At the fine scale, strokes that cut across strongly directional areas such as hair
   are dropped.
5. **Drawing order.** Line strokes are chained into a greedy "pen path" that balances travel
   distance, closeness to the focal points (OpenCV Haar face and eye detection, or the most detailed
   area as a fallback) and stroke length. Each stroke grows from one end to the other with tapered
   ends and its own pencil pressure.
6. **Shading as strokes.** The image is split into compact work areas (k-means on position + tone,
   initialised from a grid so the result is deterministic). In each area, hatching strokes are
   generated along the local stroke direction (hair flow, contours, or a comfortable diagonal):
   blocks of short side-by-side strokes going back and forth, of varied length, with ragged ends
   and a slight bow. Parallel hatching, cross-hatching and a broad tonal fill are drawn for one area
   before the pencil moves to the next. Shaded areas come from smoothed tone, so the strokes cover
   continuous areas instead of chasing paper grain.
7. **Pacing.** The passes inside each stage run one after another, with time split by how much
   graphite each adds (line passes get a minimum share so they are drawn at a believable pace). If a
   stage has almost nothing to draw (for example the start already has all the contours), its time
   goes to the next stage.
8. **Compositing.** Each frame is `start + p_draw·min(G − start, 0) + p_erase·max(G − start, 0)`.
   A pixel darkens within a fraction of a frame once the stroke over it is drawn. The progress maps
   only increase, and a static paper-tooth grain modulates partially drawn tone. Because nothing is
   random per frame, there is no flickering, and there is no camera movement.
9. **Pencil.** The tip traces the exact stroke path (stored at 1,920 samples per second), lifting
   slightly between strokes, so every new mark lies on the path the tip took during that frame. The
   pencil tilts a little as it moves. Frames average several pencil positions within the exposure
   (motion blur, capped by how far the tip actually travels so fast hatching never becomes a smear).
   The pencil enters from the lower right and moves away before the hold.
10. **Finish and encode.** The last specks of tone settle and the frame cross-fades to the exact end
   image, which is then held. Frames are rendered on a small thread pool and streamed straight into
   FFmpeg (H.264, yuv420p, BT.709, `+faststart`). Keyframes are placed so the final hold is
   perfectly still.

**Exactness.** Before encoding, the final frames are bit-for-bit identical to the normalized end
image (the end image resized to fit the canvas). H.264 is lossy, so the decoded MP4 is very close to
that image rather than bit-exact: about 43 dB PSNR at the default CRF 18. Lower the CRF for even
higher fidelity.

## Performance

Measured on an Apple Silicon Mac (10 cores) with the synthetic sample:

| Output | Time | Peak memory |
|---|---|---|
| 1080 × 1920, 30 fps, 15 s, with pencil | about 12 s | about 1.1 GB |
| 1080 × 1920, 30 fps, 15 s, no pencil | about 7 s | about 1.1 GB |
| 2160 × 3840, 30 fps, 15 s, no pencil | about 27 s | about 2.6 GB |

Analysis takes about 3 s at 1080p; the rest is rendering and encoding. The encoder preset trades
speed for file size: `ultrafast` is quickest, `slow` gives the smallest files.

## Project structure

```
app.py                       Flask web server (upload, job queue API, video streaming)
templates/index.html         Web page: portrait tab
templates/story.html         Web page: story video tab
templates/players.html       Web page: football players tab
templates/animals.html       Web page: animals tab
templates/travel.html        Web page: travel tab
templates/cartoon.html       Web page: cartoon tab
static/app.js, story.js,
  players.js, animals.js,
  travel.js, cartoon.js      Front-end logic for the tabs
static/style.css             Styles
pencil_timelapse/
  config.py                  RenderConfig, validation, five-stage timeline
  imaging.py                 Loading/validation, canvas fit, paper texture, ORB alignment
  analysis.py                Flow field, LIC, line detection, thinning, stroke ordering, regions, focus
  plan.py                    Builds the timed reveal layers (the drawing plan)
  pencil.py                  Simulated pencil: sprite, tip trajectory, compositing
  renderer.py                Per-frame compositing, threaded frame generation
  encoder.py                 FFmpeg discovery and streaming H.264 writer
  pipeline.py                High-level API: prepare → plan → render → encode
  story.py                   Story videos: script, timing, subtitles, cards, final mix (CLI)
  keywords.py                Offline Vietnamese → English image-keyword suggestions
  media.py                   Free-licence image search (Commons/Openverse), audio helpers
  sketch.py                  Photo → graphite drawing conversion
  players.py                 Football players: photos, framing, background, layout, video (CLI)
  showtime.py                Intro, "who's next?", highlights, call-to-action ending, sound effects
  animals.py                 Animal videos: drawing, name and facts per animal, gallery (CLI)
  travel.py                  Travel videos: drawing, colour reveal, stamp, plane, postcards (CLI)
  pets.py                    Pet diaries: the pet's thoughts, day bar, cold open, pet camera (CLI)
  worldmap.py                Pencil world map, pins, flight paths, map cards (assets/land.png)
  sources.py                 Photos and clips from Openverse, Pexels and Pixabay
  ai_writer.py               GPT / Gemini writing: travel scripts, pet diaries, cartoon stories
  cartoon.py                 Cartoon stories: actions, voices, bubbles, chalkboard, counting (CLI)
  cast.py                    Peeko, Lulu, Kiko and Bobo, drawn with moving arms, legs and faces
  props.py                   Props drawn by the program (fruit, toothbrush, shapes...)
  season1.py                 Peeko & Friends, season 1: twelve episodes
  episodes.py                The episode library (season 1 and the mini episodes)
  music.py                   Music made by the program: background tunes and the theme jingle
  scenery.py                 Cartoon backdrops: 13 places, time of day, weather, light, parallax
  jobs.py                    Background render queue, progress, cleanup
  cli.py                     Command-line interface
  samples.py                 Synthetic test portrait generator
tests/                       pytest suite (config, imaging, pipeline, story, players, web API)
examples/                    Sample images and a demo video (demo_pencil.mp4)
```

## Testing

```bash
python -m pytest -q
```

The suite renders small videos from a synthetic portrait (no real person's likeness is included)
and checks the following:

- the default timeline and input/setting validation
- that the video ends on exactly the end image, and that any extra hold is exactly the end image
- the artist's order of stages (only light guide lines during construction, line work before
  tonal work, guide lines erased by the end)
- that new marks lie on the path the pencil tip traced in that frame
- that the first frame is the start image (or a blank sheet in blank-paper mode)
- that the pencil is off the page at the start, is visible on the drawing while drawing,
  and is gone before the final hold
- that there are no abrupt jumps between frames and the error steadily decreases stage by stage
- that the outline stage draws lines rather than areas
- that existing artwork in a partial start is never wiped
- that stray marks are erased
- determinism, alignment, and frame count/fidelity of the encoded MP4
- the full web job lifecycle, including HTTP range requests and upload cleanup
- football players (offline, with the Wikimedia requests replaced by fixed answers): choosing the
  senior men's team, reading players and photo credits, picking the best photo with a fallback,
  the grid layout, background fading, and a complete small video with picture and credits

## Troubleshooting

| Problem | Fix |
|---|---|
| "FFmpeg was not found" | `pip install imageio-ffmpeg`, install FFmpeg system-wide, or set `PENCIL_FFMPEG`. |
| Port 5000/8000 already in use | Use `python app.py --port 9000`. On macOS, AirPlay Receiver uses port 5000. |
| Drawing looks shifted or doubled | The start and end images are framed differently and auto-alignment was rejected (see "Alignment" in the result notes). Crop both images the same way. |
| "nearly identical" note | The start already equals the end, so there is nothing to draw. |
| The video turns a photo into the drawing | The start image was a photo. Choose **Blank white paper** as the start. |
| Out of memory | Choose a smaller resolution or set `workers=1` (CLI `--workers 1`). |
| Video won't play in an old player | The output is standard H.264/yuv420p MP4. Try VLC, or re-encode with your own FFmpeg settings. |

## Limitations

- The drawing order is inferred from the finished image. Real artists may work in a completely
  different order, use different strokes, or erase and redraw.
- Face and eye detection uses classical Haar cascades. Stylised, profile or heavily shaded faces may
  not be detected, in which case the focus falls back to the most detailed area.
- Line detection is tuned for graphite drawings on light paper. Other media (charcoal on dark paper,
  colour paintings) are animated but look less like pencil work.
- The pencil is an overlay. It shows where graphite is being added but does not touch the
  drawing physically (no smudging, no hand), and it moves faster than a real pencil because the
  whole portrait is drawn in about 13 seconds.
- If the start and end images are not views of the same sheet, the start content is faded out in
  the shading stage rather than being drawn over convincingly.
