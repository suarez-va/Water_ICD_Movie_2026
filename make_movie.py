#!/usr/bin/env python
"""Stitch movie/*.png into an .mp4 with ffmpeg.

The frames already exist as PNGs on disk, so they go straight into ffmpeg.
Routing them back through matplotlib's FuncAnimation/FFMpegWriter (the approach
in 4_plot_rhot.py) is the right call when matplotlib is *drawing* the frames,
but here it would decode every PNG, blit it into a figure, and re-rasterise it
before encoding -- slower, and it resamples images that are already final.

Frames are piped to ffmpeg on stdin rather than matched with a `-i %d.png`
pattern.  The pattern form needs contiguous numbering starting at a known
index, which breaks the moment you render a subset (`--start 100`) or drop a
frame; piping puts the ordering entirely under this script's control.

Usage
-----
    python make_movie.py                          # pacceptor/frames/ -> hole_movie.mp4
    python make_movie.py --fps 60 -o fast.mp4
    python make_movie.py --start 0 --limit 200    # first 200 frames only
    python make_movie.py --stride 10 --fps 12     # every 10th frame, same length
    python make_movie.py --codec h265 --crf 20
    python make_movie.py --lossless -o archive.mkv

--stride N uses one frame and skips the next N-1.  Dropping --fps by the same
factor keeps both the running time and the apparent speed of the physics
identical to the un-strided movie -- it just shows the same evolution with
fewer, longer-held frames.  The reported "a.u. of simulation per second of
video" is the number to match if you want two movies to play at the same rate.
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import time

# Encoder settings.  yuv420p is what makes the file play in QuickTime,
# PowerPoint and browsers; without it x264 defaults to yuv444p, which many
# players silently refuse.
CODECS = {
    'h264': ['-c:v', 'libx264', '-pix_fmt', 'yuv420p'],
    'h265': ['-c:v', 'libx265', '-pix_fmt', 'yuv420p', '-tag:v', 'hvc1'],
    'vp9':  ['-c:v', 'libvpx-vp9', '-pix_fmt', 'yuv420p'],
}


def natural_key(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    return [int(p) if p.isdigit() else p for p in re.split(r'(\d+)', stem)]


def list_frames(framedir):
    """Every PNG in the directory, in numeric order."""
    if not os.path.isdir(framedir):
        raise SystemExit(f'make_movie.py: no such directory: {framedir}')
    frames = [os.path.join(framedir, f) for f in os.listdir(framedir)
              if f.lower().endswith('.png')]
    if not frames:
        raise SystemExit(f'make_movie.py: no .png files in {framedir}')
    frames.sort(key=natural_key)
    return frames


def select_frames(frames, start, limit, stride=1):
    """Offset by `start`, then take every `stride`-th.

    stride=10 means "use one frame, skip the next nine".  Note this counts by
    position in the list, not by frame number -- identical on a complete
    directory, but on one with holes the chosen frames will not be evenly
    spaced in time.  report_gaps() runs on the full listing first so an
    incomplete render is flagged before it can distort the spacing.

    `limit` is applied last, so it counts frames that actually reach the
    encoder rather than frames considered.
    """
    if stride < 1:
        raise SystemExit('make_movie.py: --stride must be at least 1')
    frames = frames[start::stride]
    if limit:
        frames = frames[:limit]
    if not frames:
        raise SystemExit('make_movie.py: --start/--stride/--limit selected '
                         'no frames')
    return frames


def frame_numbers(frames):
    """Numeric stems, or None if any filename is not a plain integer."""
    nums = []
    for f in frames:
        stem = os.path.splitext(os.path.basename(f))[0]
        if not stem.isdigit():
            return None
        nums.append(int(stem))
    return nums


def report_gaps(frames):
    """Warn if the source numbering has holes -- usually a died render.

    Deliberately runs on the complete listing, before any striding: a stride is
    *supposed* to skip numbers, so checking the strided output would either flag
    every skip or (worse) compare against an assumed spacing the source does not
    actually have.
    """
    nums = frame_numbers(frames)
    if not nums:
        return False
    missing = sorted(set(range(nums[0], nums[-1] + 1)) - set(nums))
    if missing:
        shown = ', '.join(str(m) for m in missing[:10])
        more = f' (+{len(missing) - 10} more)' if len(missing) > 10 else ''
        print(f'  WARNING: {len(missing)} frame numbers missing between '
              f'{nums[0]} and {nums[-1]}: {shown}{more}')
        print('           the movie will simply skip them, so motion will jump')
    return bool(missing)


def build_command(ffmpeg, fps, args, out):
    cmd = [ffmpeg, '-y', '-f', 'image2pipe', '-framerate', str(fps), '-i', '-']
    # libx264 needs even dimensions for yuv420p; round down rather than fail.
    cmd += ['-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2']
    if args.lossless:
        cmd += ['-c:v', 'ffv1', '-level', '3']
    else:
        cmd += CODECS[args.codec]
        cmd += ['-crf', str(args.crf), '-preset', args.preset]
    cmd += ['-r', str(fps)]
    if out.lower().endswith('.mp4'):
        cmd += ['-movflags', '+faststart']   # lets it start playing while loading
    cmd.append(out)
    return cmd


def main(argv=None):
    p = argparse.ArgumentParser(
        description='Encode a directory of PNG frames into a video with ffmpeg.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--frames', default='pacceptor/frames', help='directory of .png frames')
    p.add_argument('-o', '--output', default='hole_movie.mp4', help='output video')
    p.add_argument('--fps', type=int, default=120, help='playback frame rate')
    p.add_argument('--start', type=int, default=0, help='index of the first frame')
    p.add_argument('--limit', type=int, help='use only this many frames')
    p.add_argument('--stride', type=int, default=1, metavar='N',
                   help='use every Nth frame; N=10 means use one and skip nine. '
                        'Drop --fps by the same factor to keep the movie the '
                        'same length and the physics running at the same speed')
    p.add_argument('--codec', choices=sorted(CODECS), default='h264',
                   help='video codec')
    p.add_argument('--crf', type=int, default=18,
                   help='quality, lower is better; 18 is near-transparent, '
                        '23 is the ffmpeg default')
    p.add_argument('--preset', default='slow',
                   help='x264/x265 speed-vs-compression preset')
    p.add_argument('--lossless', action='store_true',
                   help='encode FFV1 in a .mkv instead (archival, very large)')
    p.add_argument('--timestep', type=float, default=1.0,
                   help='simulation time per frame, in atomic units, used only '
                        'to report how much physical time the movie covers')
    p.add_argument('--ffmpeg', default=shutil.which('ffmpeg') or 'ffmpeg')
    args = p.parse_args(argv)

    if not shutil.which(args.ffmpeg) and not os.path.exists(args.ffmpeg):
        raise SystemExit(f'make_movie.py: ffmpeg not found: {args.ffmpeg}')
    if args.lossless and args.output.lower().endswith('.mp4'):
        raise SystemExit('make_movie.py: FFV1 does not belong in an .mp4; '
                         'use -o something.mkv with --lossless')

    available = list_frames(args.frames)
    has_gaps = report_gaps(available)
    if has_gaps and args.stride > 1:
        print('           --stride counts by position, so with frames missing '
              'the\n           selected frames will not be evenly spaced in time')
    frames = select_frames(available, args.start, args.limit, args.stride)

    # 1 a.u. of time = 0.0241888 fs.  Each encoded frame now advances the
    # simulation by stride*timestep, so the span covered is unchanged by
    # striding -- only the number of frames used to show it drops.
    fs = len(frames) * args.stride * args.timestep * 0.02418884
    print(f'encoding {len(frames)} frames -> {args.output}')
    print(f'  source   {args.frames}/  ({os.path.basename(frames[0])} .. '
          f'{os.path.basename(frames[-1])})')
    if args.stride > 1:
        print(f'  stride   every {args.stride}th frame '
              f'(using 1, skipping {args.stride - 1})')
    print(f'  codec    {"ffv1 (lossless)" if args.lossless else args.codec} '
          f'@ {args.fps} fps'
          + ('' if args.lossless else f', crf {args.crf}, preset {args.preset}'))
    print(f'  duration {len(frames) / args.fps:.1f} s of video '
          f'covering {fs:.1f} fs of simulation')
    print(f'           {args.stride * args.timestep * args.fps:.0f} a.u. of '
          f'simulation per second of video\n')

    cmd = build_command(args.ffmpeg, args.fps, args, args.output)
    t0 = time.time()
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    try:
        for i, f in enumerate(frames, 1):
            with open(f, 'rb') as fh:
                proc.stdin.write(fh.read())
            if i % 25 == 0 or i == len(frames):
                el = time.time() - t0
                print(f'\r  {i}/{len(frames)} frames piped, {el:.0f}s elapsed',
                      end='', flush=True)
        proc.stdin.close()
    except BrokenPipeError:
        pass    # ffmpeg died; the stderr below explains why
    err = proc.stderr.read().decode(errors='replace')
    if proc.wait() != 0:
        sys.stderr.write('\nmake_movie.py: ffmpeg failed\n')
        sys.stderr.write('\n'.join(err.splitlines()[-25:]) + '\n')
        raise SystemExit(1)

    size = os.path.getsize(args.output)
    print(f'\n\ndone: {args.output}  ({size / 1e6:.1f} MB, '
          f'{time.time() - t0:.0f}s)')
    print(f'  {len(frames)} frames, {len(frames) / args.fps:.1f} s at {args.fps} fps')


if __name__ == '__main__':
    sys.exit(main())
