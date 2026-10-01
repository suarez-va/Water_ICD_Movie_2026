#!/usr/bin/env python
"""Solve for the VIEW_ZOOM / VIEW_SHIFT_X / VIEW_SHIFT_Y in vmd_settings.tcl.

VMD's `display resetview` fits the *nuclei* and knows nothing about how far the
isosurface extends, so the framing has to be set by hand -- and it goes stale
the moment you change ISOVALUE, the rotation, the atom representation, or the
aspect ratio.  Rather than eyeballing it, this renders a few probe frames,
measures where the drawn content actually lands, and solves for the values that
put it in the frame with the margin you asked for.

    python fit_view.py                     # report the values
    python fit_view.py --apply             # and write them into vmd_settings.tcl
    python fit_view.py --margin 0.15       # leave 15% empty on the tightest edge
    python fit_view.py --probes 8          # sample more frames

Probes are spread across the whole cube directory, because the isosurface grows
and shrinks over the trajectory and the framing has to accommodate its *widest*
moment -- fitting to frame 0 alone would clip later on.

How the solve works
-------------------
The projection is orthographic, so screen extent is exactly linear in zoom:

    zoom_new = zoom_now * target_fill / measured_fill

`translate by` writes VMD's global_matrix, which is applied after the scale
matrix, so its effect on screen position is independent of zoom.  That makes
the shift solvable in closed form too, once its sensitivity is calibrated -- the
script measures that by rendering one extra pass with a deliberate offset
rather than assuming a constant.
"""

import argparse
import os
import re
import shutil
import sys
import tempfile

import numpy as np

import render_movie

HERE = os.path.dirname(os.path.abspath(__file__))
SETTINGS = os.path.join(HERE, 'vmd_settings.tcl')
CAL_STEP = 0.5          # shift offset used to calibrate translate sensitivity


def read_values(path):
    """Pull the three framing variables out of the Tcl file."""
    text = open(path).read()
    out = {}
    for name in ('VIEW_ZOOM', 'VIEW_SHIFT_X', 'VIEW_SHIFT_Y'):
        m = re.search(rf'^\s*set\s+{name}\s+(-?[\d.eE+]+)', text, re.M)
        if not m:
            raise SystemExit(f'fit_view.py: {name} not found in {path}')
        out[name] = float(m.group(1))
    return out


def write_values(path, vals):
    text = open(path).read()
    for name, v in vals.items():
        text, n = re.subn(rf'^(\s*set\s+{name}\s+)(-?[\d.eE+]+)',
                          lambda m: f'{m.group(1)}{v:g}', text, count=1, flags=re.M)
        if not n:
            raise SystemExit(f'fit_view.py: could not rewrite {name}')
    open(path, 'w').write(text)


def settings_with(vals, tmpdir):
    """A copy of vmd_settings.tcl with the framing variables overridden."""
    out = os.path.join(tmpdir, 'probe_settings.tcl')
    shutil.copy(SETTINGS, out)
    write_values(out, vals)
    return out


def pick_probes(cubedir, n):
    """Frames spread evenly across the trajectory.

    The first probe is always the first cube: setup_view pins the camera at
    whichever frame VMD loads first, and that has to match the movie render.
    """
    cubes = render_movie.find_cubes(cubedir, 0, None)
    if n >= len(cubes):
        return cubes
    idx = np.linspace(0, len(cubes) - 1, n).round().astype(int)
    return [cubes[i] for i in sorted(set(idx.tolist()))]


def measure(paths):
    """Union content bounding box over the probes, as fractions of the frame.

    The union, not the per-frame worst case: the isosurface drifts as well as
    grows, so each frame has its own centre.  What the camera has to satisfy is
    the box that contains *every* frame, and centring that union is the
    well-posed target -- centring some per-frame maximum is not, since no single
    frame's offset describes the series.

    Returns (fill_x, fill_y, off_x, off_y, min_edge_margin_px).
    """
    from PIL import Image
    x0 = y0 = np.inf
    x1 = y1 = -np.inf
    W = H = None
    for p in paths:
        with Image.open(p) as im:
            a = np.asarray(im.convert('RGB'), dtype=np.int16)
        bg = a[0, 0]
        mask = (np.abs(a - bg) > 10).any(2)
        if not mask.any():
            raise SystemExit(f'fit_view.py: {p} is blank -- nothing was drawn')
        ys, xs = np.where(mask)
        H, W = a.shape[:2]
        x0, x1 = min(x0, xs.min()), max(x1, xs.max())
        y0, y1 = min(y0, ys.min()), max(y1, ys.max())
    fx = (x1 - x0 + 1) / W
    fy = (y1 - y0 + 1) / H
    ox = ((x0 + x1) / 2 - W / 2) / W
    oy = ((y0 + y1) / 2 - H / 2) / H
    margin = int(min(x0, y0, W - 1 - x1, H - 1 - y1))
    return fx, fy, ox, oy, margin


def render_probes(vals, cubes, size, vmd, tmpdir, tag, upsample):
    sfile = settings_with(vals, tmpdir)
    outdir = os.path.join(tmpdir, tag)
    os.makedirs(outdir, exist_ok=True)
    jobs = render_movie.render_chunk(vmd, cubes, outdir, sfile, size, quiet=True,
                                     upsample=upsample)
    missing = [s for s, t in jobs if not os.path.exists(t)]
    if missing:
        raise SystemExit(f'fit_view.py: VMD produced no output for {missing}')
    return [t for _, t in jobs]


def main(argv=None):
    p = argparse.ArgumentParser(
        description='Measure the rendered content and solve for the framing '
                    'variables in vmd_settings.tcl.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--cubes', default='pacceptor/holecubes', help='directory of .cube files')
    p.add_argument('--probes', type=int, default=5,
                   help='how many frames to sample across the trajectory')
    p.add_argument('--margin', type=float, default=0.05,
                   help='fraction of the frame left empty on the tightest edge')
    p.add_argument('--size', type=int, nargs=2, default=[1600, 900],
                   metavar=('W', 'H'),
                   help='frame size to fit for; must match how you will render')
    p.add_argument('--apply', action='store_true',
                   help='write the solved values back into vmd_settings.tcl')
    p.add_argument('--no-verify', action='store_true',
                   help='skip the confirming render after solving')
    p.add_argument('--upsample', type=int, default=1, metavar='N',
                   help='spline upsampling factor; must match render_movie.py')
    p.add_argument('--vmd', default=shutil.which('vmd') or 'vmd')
    args = p.parse_args(argv)

    if args.margin < 0 or args.margin >= 0.5:
        raise SystemExit('fit_view.py: --margin must be in [0, 0.5)')
    target = 1.0 - 2.0 * args.margin

    cur = read_values(SETTINGS)
    cubes = pick_probes(args.cubes, args.probes)
    print(f'fitting {os.path.basename(SETTINGS)} at {args.size[0]}x{args.size[1]}, '
          f'{args.margin:.0%} margin')
    print(f'  probes: {", ".join(os.path.basename(c) for c in cubes)}')
    print(f'  current: zoom {cur["VIEW_ZOOM"]:g}, shift '
          f'({cur["VIEW_SHIFT_X"]:g}, {cur["VIEW_SHIFT_Y"]:g})\n')

    tmpdir = tempfile.mkdtemp(prefix='fitview_')
    try:
        # Pass 1: where does the content sit now?
        f = render_probes(cur, cubes, args.size, args.vmd, tmpdir, 'p1', args.upsample)
        fx, fy, ox, oy, marg = measure(f)
        print(f'  measured: fill x={fx:.3f} y={fy:.3f}, '
              f'offset x={ox:+.3f} y={oy:+.3f}, edge margin {marg}px')

        # Pass 2: how far does one unit of translate actually move the image?
        cal = dict(cur, VIEW_SHIFT_X=cur['VIEW_SHIFT_X'] + CAL_STEP,
                        VIEW_SHIFT_Y=cur['VIEW_SHIFT_Y'] + CAL_STEP)
        f = render_probes(cal, cubes[:1], args.size, args.vmd, tmpdir, 'p2', args.upsample)
        _, _, ox2, oy2, _ = measure(f)
        # Recompute pass-1 offsets on the same single frame, so the difference
        # is not contaminated by which probe happened to be the worst case.
        f = render_probes(cur, cubes[:1], args.size, args.vmd, tmpdir, 'p1b', args.upsample)
        _, _, ox1, oy1, _ = measure(f)
        kx = (ox2 - ox1) / CAL_STEP
        ky = (oy2 - oy1) / CAL_STEP
        print(f'  translate sensitivity: x {kx:+.4f} y {ky:+.4f} frame per unit')
        if abs(kx) < 1e-4 or abs(ky) < 1e-4:
            raise SystemExit('fit_view.py: translate had no measurable effect; '
                             'cannot solve for the shift')

        # Zoom is exactly linear in screen extent under orthographic projection.
        ratio = target / max(fx, fy)
        zoom = cur['VIEW_ZOOM'] * ratio

        # Split the measured offset into the part the geometry contributes
        # (which scales with zoom) and the part the current translate
        # contributes (which does not), then solve for zero total offset.
        gx = ox - kx * cur['VIEW_SHIFT_X']
        gy = oy - ky * cur['VIEW_SHIFT_Y']
        sx = -gx * ratio / kx
        sy = -gy * ratio / ky

        new = {'VIEW_ZOOM': round(zoom, 3),
               'VIEW_SHIFT_X': round(sx, 3),
               'VIEW_SHIFT_Y': round(sy, 3)}
        print(f'\n  solved:  zoom {new["VIEW_ZOOM"]:g}, shift '
              f'({new["VIEW_SHIFT_X"]:g}, {new["VIEW_SHIFT_Y"]:g})')

        if not args.no_verify:
            f = render_probes(new, cubes, args.size, args.vmd, tmpdir, 'v', args.upsample)
            fx2, fy2, ox2, oy2, marg2 = measure(f)
            print(f'  verified: fill x={fx2:.3f} y={fy2:.3f}, '
                  f'offset x={ox2:+.3f} y={oy2:+.3f}, edge margin {marg2}px')
            if marg2 < 2:
                print('  WARNING: content still touches the frame edge; '
                      'raise --margin and rerun')
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    if args.apply:
        write_values(SETTINGS, new)
        print(f'\nwrote the three values into {os.path.relpath(SETTINGS, HERE)}')
    else:
        print('\nnot applied.  Rerun with --apply, or edit vmd_settings.tcl:')
        for k, v in new.items():
            print(f'    set {k:<13s} {v:g}')


if __name__ == '__main__':
    sys.exit(main())
