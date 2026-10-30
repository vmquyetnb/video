"""Command-line interface.

Examples::

    python -m pencil_timelapse.cli end.png -o drawing.mp4                 # blank paper + pencil
    python -m pencil_timelapse.cli start.png end.png -o drawing.mp4       # continue a partial drawing
    python -m pencil_timelapse.cli end.png -o out.mp4 --no-pencil --duration 20 --size 720x1280
    python -m pencil_timelapse.cli --make-samples samples/
"""

from __future__ import annotations

import argparse
import os
import sys
import time

from .config import PENCIL_STYLES, RESOLUTION_PRESETS, START_MODES, X264_PRESETS, RenderConfig
from .errors import PencilError


def _size(text: str):
    try:
        w, h = text.lower().replace("×", "x").split("x")
        return int(w), int(h)
    except ValueError:
        raise argparse.ArgumentTypeError("size must look like 1080x1920") from None


def build_parser() -> argparse.ArgumentParser:
    d = RenderConfig()
    p = argparse.ArgumentParser(
        prog="pencil-timelapse",
        description="Create a simulated pencil-drawing timelapse video from a start image and a "
                    "finished pencil portrait. The strokes are simulated with deterministic image "
                    "processing; they are not a reconstruction of the artist's real hand movements.")
    p.add_argument("images", nargs="*", metavar="IMAGE",
                   help="END (draw on blank paper) or START END (continue from a start image)")
    p.add_argument("-o", "--output", default="pencil_timelapse.mp4", help="output MP4 path")
    p.add_argument("--duration", type=float, default=d.duration, help="video length in seconds")
    p.add_argument("--fps", type=int, default=d.fps, help="frames per second")
    p.add_argument("--size", type=_size, default=(d.width, d.height),
                   help=f"output resolution WxH (presets: {', '.join(RESOLUTION_PRESETS)})")
    p.add_argument("--hold", type=float, default=d.hold, help="seconds to hold the exact end image")
    p.add_argument("--no-align", action="store_true", help="do not register the start image to the end")
    p.add_argument("--no-paper-match", action="store_true", help="keep the start image's paper tone")
    p.add_argument("--no-construction", action="store_true", help="omit the light guide lines")
    p.add_argument("--paper-texture", type=float, default=d.paper_texture,
                   help="procedural paper texture strength 0-1 (used for padding and flat starts)")
    p.add_argument("--stroke-strength", type=float, default=d.stroke_strength,
                   help="strength of simulated pencil stroke texture 0-1")
    p.add_argument("--crf", type=int, default=d.crf, help="H.264 quality (lower = better, 18 default)")
    p.add_argument("--preset", choices=X264_PRESETS, default=d.preset, help="x264 speed preset")
    p.add_argument("--seed", type=int, default=d.seed, help="random seed for textures (deterministic)")
    p.add_argument("--workers", type=int, default=0, help="frame-render threads (0 = auto)")
    p.add_argument("--start-mode", choices=START_MODES,
                   help="'blank' = draw on empty paper, 'image' = continue from START "
                        "(default: 'image' when two images are given, else 'blank')")
    p.add_argument("--no-pencil", action="store_true",
                   help="hide the simulated pencil that follows the strokes")
    p.add_argument("--pencil-scale", type=float, default=d.pencil_scale, help="pencil size, 0.4-2")
    p.add_argument("--pencil-style", choices=PENCIL_STYLES, default=d.pencil_style,
                   help="pencil look: black artist pencil, yellow, or natural wood")
    p.add_argument("--make-samples", metavar="DIR",
                   help="write synthetic sample images to DIR and exit")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.make_samples:
        from .samples import write_samples

        for name, path in write_samples(args.make_samples).items():
            print(f"{name:8s} {path}")
        return 0
    if len(args.images) not in (1, 2):
        print("error: give END, or START and END (or use --make-samples DIR)", file=sys.stderr)
        return 2
    start = args.images[0] if len(args.images) == 2 else None
    end = args.images[-1]
    start_mode = args.start_mode or ("image" if start else "blank")

    from .pipeline import render_video

    try:
        cfg = RenderConfig(
            width=args.size[0], height=args.size[1], fps=args.fps, duration=args.duration,
            hold=args.hold, align=not args.no_align, match_paper=not args.no_paper_match,
            paper_texture=args.paper_texture, stroke_strength=args.stroke_strength,
            construction_lines=not args.no_construction, seed=args.seed, crf=args.crf,
            preset=args.preset, workers=args.workers, start_mode=start_mode,
            show_pencil=not args.no_pencil, pencil_scale=args.pencil_scale,
            pencil_style=args.pencil_style).validate()
        started = time.time()
        interactive = sys.stderr.isatty()

        def progress(fraction: float, message: str) -> None:
            bar = "#" * int(fraction * 30)
            line = f"[{bar:<30}] {fraction * 100:5.1f}%  {message}"
            if interactive:
                sys.stderr.write("\r" + line[:110].ljust(110))
                sys.stderr.flush()

        result = render_video(start, end, args.output, cfg, progress)
        if interactive:
            sys.stderr.write("\n")
    except PencilError as exc:
        print(f"\nerror: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\ncancelled", file=sys.stderr)
        return 130

    print(f"Saved {os.path.abspath(result.output_path)}")
    print(f"  {result.width}x{result.height} @ {result.fps} fps, {result.frames} frames, "
          f"{result.duration:g} s, encoder {result.encoder}, rendered in {time.time() - started:.1f} s")
    focus = result.info.get("focus", {})
    print(f"  focus: {focus.get('method')}, alignment: {result.info.get('alignment')}")
    for w in result.warnings:
        print(f"  note: {w}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
