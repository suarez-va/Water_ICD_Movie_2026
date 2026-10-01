#!/usr/bin/env python
"""Render pacceptor/holecubes/*.cube to PNG frames with VMD.

VMD runs headless (`-dispdev text`) and Tachyon ray-traces each frame on the
CPU, so no X display is involved.  There is no GPU path: VMD's accelerated
renderer is NVIDIA OptiX/CUDA only, and this machine has no CUDA device.

By default VMD only writes a Tachyon *scene* here, which this script then
patches before handing to the standalone tachyon binary.  The reason is that
ambient occlusion dominates render time -- roughly 20 s of a 20.3 s frame at
1080p -- and its sample count is not reachable any other way: VMD 2.0.0's
`display ambientocclusion` is on/off only, and `render TachyonInternal` bakes in
12 AO and 12 antialiasing samples.  The scene file exposes both, and dropping
them to 4 is ~5x faster with no visible difference on smooth translucent
surfaces.  Pass --renderer internal to go back to VMD's own renderer.

--renderer optix ray-traces on an NVIDIA GPU with VMD's TachyonL-OptiX
(`render TachyonLOptiXInternal`).  It needs a VMD built with CUDA and OptiX --
VMD's startup prints "Detected 1 available TachyonL/OptiX ray tracing
accelerator" when it is -- such as the cluster's vmd/1.9.3; the local VMD 2.0
cannot load OptiX.  VMD writes the image directly and takes --ao-samples and
--aa-samples as settings.  It compiles its OptiX shaders at every launch, a
fixed per-chunk cost, so larger chunks pay off with this renderer.

Neither the bundled tachyon binary nor VMD's writer was compiled with PNG
support, so frames land as Targa and Pillow converts them.

Cubes can be upsampled with a cubic spline before VMD sees it (--upsample N,
default 1 = off).  This is for cubes made with cubegen's default grid; the
ehrenfest scripts now write CUBE_RESOLUTION = 0.06 Bohr directly, which needs
no upsampling.  The default writes only 80^3 points over a box that follows
the molecule, ~0.12-0.16 Bohr apart, and VMD's marching-cubes isosurface on that
grid shows visible facets on the small lobes.  It is the grid, not the basis --
a Gaussian-basis density is smooth everywhere -- and not the shading, which
already uses per-vertex normals.  A spline refines a smooth function faithfully
(unlike `voltool smooth`, a blur, or `voltool supersample`, which is
trilinear).  2x removes nearly all faceting for ~1.8x the time per frame; 3x is
barely better for another 2x.  The upsampled copies live only in scratch; the
cubes on disk are never touched.

Frames are rendered in chunks, one VMD launch per chunk, for two reasons:
launching VMD costs ~2 s and would dominate a per-frame launch, while holding
all 2000 uncompressed Targas at once would need ~3.4 GB of scratch space.

Files
-----
    vmd_settings.tcl    what the movie looks like -- edit this one
    render_frames.tcl   the render loop VMD runs -- machinery, leave alone
    render_movie.py     this driver

Usage
-----
    python render_movie.py --limit 5              # first 5 frames
    python render_movie.py                        # all of them
    python render_movie.py --start 100 --limit 20 # frames 100..119
    python render_movie.py --size 1920 1080 --limit 1
    python render_movie.py --cube pacceptor/holecubes/0.cube --outdir snapshots

To preview a single frame interactively while tuning vmd_settings.tcl:
    vmd -e vmd_settings.tcl pacceptor/holecubes/0.cube
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))

# Shipped with VMD.  Not compiled with PNG support ("-format PNG  XXX Not
# compiled into this binary XXX"), hence the Targa intermediate.
TACHYON = '/home/victorwsl/.local/lib/vmd/tachyon_LINUXAMD64'

# A 2x cube is ~58 MB of text and every cube in a chunk sits in scratch until
# VMD exits, so upsampled chunks are kept small.
UPSAMPLED_CHUNK = 20


def read_cube(path):
    """Gaussian cube -> (comments, origin line fields, axes, atom lines, data).

    `axes` is a list of (n, vector) and `data` is shaped (nx, ny, nz).
    """
    import numpy as np
    with open(path) as f:
        comments = [f.readline().rstrip('\n'), f.readline().rstrip('\n')]
        head = f.readline().split()
        natoms, origin = int(head[0]), [float(x) for x in head[1:4]]
        axes = []
        for _ in range(3):
            a = f.readline().split()
            axes.append((int(a[0]), np.array([float(x) for x in a[1:4]])))
        atoms = [f.readline().rstrip('\n') for _ in range(abs(natoms))]
        data = np.array(f.read().split(), dtype=float)
    shape = tuple(n for n, _ in axes)
    return comments, natoms, origin, axes, atoms, data.reshape(shape)


def write_cube(path, comments, natoms, origin, axes, atoms, data):
    """Inverse of read_cube; 6 values per line, rows along z, like PySCF."""
    with open(path, 'w') as f:
        f.write('\n'.join(comments) + '\n')
        f.write(f'{natoms:5d}' + ''.join(f'{x:12.6f}' for x in origin) + '\n')
        for n, v in axes:
            f.write(f'{n:5d}' + ''.join(f'{x:12.6f}' for x in v) + '\n')
        f.write('\n'.join(atoms) + '\n')
        rows = data.reshape(-1, data.shape[2])
        nz = rows.shape[1]
        fmt = '\n'.join(''.join(['%14.5e'] * min(6, nz - k))
                        for k in range(0, nz, 6)) + '\n'
        for row in rows:
            f.write(fmt % tuple(row))


def upsample_cube(src, dst, factor):
    """Write `src` resampled `factor`x per axis with a cubic spline to `dst`.

    zoom() maps first and last grid points onto themselves, so the new spacing
    is old * (N-1)/(N_new-1): same box, same origin, same atoms.
    """
    from scipy import ndimage
    comments, natoms, origin, axes, atoms, data = read_cube(src)
    fine = ndimage.zoom(data, factor, order=3, mode='nearest')
    new_axes = [(m, v * (n - 1) / (m - 1))
                for (n, v), m in zip(axes, fine.shape)]
    write_cube(dst, comments, natoms, origin, new_axes, atoms, fine)


def natural_key(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    return [int(p) if p.isdigit() else p for p in re.split(r'(\d+)', stem)]


def find_cubes(cubedir, start, limit):
    """Cube files in numeric order, offset by `start` and capped at `limit`."""
    if not os.path.isdir(cubedir):
        raise SystemExit(f'render_movie.py: no such directory: {cubedir}')
    cubes = [os.path.join(cubedir, f) for f in os.listdir(cubedir)
             if f.endswith('.cube')]
    if not cubes:
        raise SystemExit(f'render_movie.py: no .cube files in {cubedir}')
    cubes.sort(key=natural_key)
    cubes = cubes[start:]
    if limit:
        cubes = cubes[:limit]
    return cubes


_RE_SAMPLES = re.compile(r'^(\s*Samples\s+)\d+\s*$', re.M)
_RE_ANTIALIAS = re.compile(r'^(\s*Antialiasing\s+)\d+\s*$', re.M)
_RE_RESOLUTION = re.compile(r'^\s*Resolution\s+(\d+)\s+(\d+)\s*$', re.M)


def patch_scene(scene, ao_samples, aa_samples, size):
    """Lower the two sample counts in a Tachyon scene file, in place.

    These are the whole point of the external renderer.  VMD writes
    `Samples 12` (ambient occlusion) and `Antialiasing 12`, and both are
    multiplicative on render time -- dropping them to 4 is ~5x faster with no
    visible change on smooth translucent surfaces.  `display ambientocclusion`
    is on/off only in VMD 2.0.0, and TachyonInternal exposes neither.
    """
    text = open(scene).read()

    # A mismatch here would silently produce a stretched image rather than an
    # error, because the camera baked into the scene follows the display's
    # aspect ratio while -res would impose a different one.
    m = _RE_RESOLUTION.search(text)
    if not m:
        raise SystemExit(f'render_movie.py: no Resolution line in {scene}')
    got = (int(m.group(1)), int(m.group(2)))
    if got != tuple(size):
        raise SystemExit(
            f'render_movie.py: scene is {got[0]}x{got[1]} but {size[0]}x{size[1]} '
            f'was requested.  VMD could not resize its display to the target '
            f'size, so the camera aspect would not match the output.')

    text, n_ao = _RE_SAMPLES.subn(rf'\g<1>{ao_samples}', text)
    text, n_aa = _RE_ANTIALIAS.subn(rf'\g<1>{aa_samples}', text)
    if not n_ao and ao_samples is not None:
        # Not fatal: AO is off in the settings, so the scene has no AO block.
        pass
    if not n_aa:
        raise SystemExit(f'render_movie.py: no Antialiasing line in {scene}; '
                         f'the Tachyon scene format may have changed')
    open(scene, 'w').write(text)


def run_tachyon(tachyon, scene, tga):
    """Ray-trace a patched scene to Targa with the standalone binary."""
    proc = subprocess.run([tachyon, scene, '-format', 'TARGA', '-o', tga],
                          capture_output=True, text=True)
    if proc.returncode != 0 or not os.path.exists(tga):
        sys.stderr.write(f'\nrender_movie.py: tachyon failed on {scene}\n')
        sys.stderr.write('\n'.join((proc.stdout + proc.stderr).splitlines()[-20:]))
        sys.stderr.write('\n')
        raise SystemExit(1)


def render_chunk(vmd, cubes, tgadir, settings, size, quiet,
                 renderer='external', ao_samples=4, aa_samples=4,
                 tachyon=None, upsample=1):
    """One VMD launch: render every cube in `cubes` to a .tga in tgadir.

    With upsample > 1 each cube is first spline-resampled into tgadir and VMD
    loads that copy; the copies are deleted as soon as VMD exits.

    With renderer='external' VMD writes a Tachyon scene per frame, which is then
    patched to lower the sample counts and ray-traced by the standalone binary.
    Either way the contract is the same -- TGAs in tgadir, [(stem, tga)] back --
    so callers such as fit_view.py do not care which backend ran.
    """
    tachyon = tachyon or TACHYON
    # internal and optix: VMD writes the image itself.
    external = renderer == 'external'
    if external and not (shutil.which(tachyon) or os.path.exists(tachyon)):
        raise SystemExit(f'render_movie.py: tachyon binary not found: {tachyon}')

    jobs = []
    fine_cubes = []
    if upsample > 1:
        # ~2.5 s of numpy/scipy and text I/O per cube, independent across
        # cubes, so the whole chunk is prepared in parallel up front.
        from concurrent.futures import ProcessPoolExecutor
        srcs = list(cubes)
        cubes = [os.path.join(tgadir, os.path.splitext(os.path.basename(c))[0]
                              + '.cube') for c in srcs]
        fine_cubes = cubes
        with ProcessPoolExecutor(max_workers=min(len(srcs), os.cpu_count() or 1)) as ex:
            list(ex.map(upsample_cube, srcs, cubes, [upsample] * len(srcs)))
    with tempfile.NamedTemporaryFile('w', suffix='.txt', delete=False) as jf:
        for cube in cubes:
            stem = os.path.splitext(os.path.basename(cube))[0]
            tga = os.path.join(tgadir, stem + '.tga')
            # VMD writes here: the scene file for external, the image directly
            # for internal.
            target = os.path.join(tgadir, stem + '.dat') if external else tga
            jf.write(f'{os.path.abspath(cube)}\t{os.path.abspath(target)}\n')
            jobs.append((stem, tga, target))
        joblist = jf.name

    # The loop script goes in on stdin rather than via -e: a -e script runs
    # before VMD's display device exists, so `display resize` throws and VMD
    # then abandons the rest of the file silently with exit status 0.
    env = dict(os.environ,
               VMDMOVIE_JOBLIST=joblist,
               VMDMOVIE_SETTINGS=settings,
               VMDMOVIE_WIDTH=str(size[0]),
               VMDMOVIE_HEIGHT=str(size[1]),
               VMDMOVIE_RENDERER=renderer,
               VMDMOVIE_AO_SAMPLES=str(ao_samples),
               VMDMOVIE_AA_SAMPLES=str(aa_samples))
    try:
        with open(os.path.join(HERE, 'render_frames.tcl')) as script:
            proc = subprocess.run([vmd, '-dispdev', 'text'], stdin=script,
                                  capture_output=True, text=True, cwd=HERE, env=env)
    finally:
        os.unlink(joblist)
        for fine in fine_cubes:
            os.unlink(fine)

    if 'RENDER_LOOP_DONE' not in proc.stdout:
        sys.stderr.write('\nrender_movie.py: VMD did not finish the render loop.\n')
        sys.stderr.write('--- VMD stdout (tail) ---\n')
        sys.stderr.write('\n'.join(proc.stdout.splitlines()[-40:]) + '\n')
        if proc.stderr.strip():
            sys.stderr.write('--- VMD stderr (tail) ---\n')
            sys.stderr.write('\n'.join(proc.stderr.splitlines()[-20:]) + '\n')
        raise SystemExit(1)
    if not quiet:
        for line in proc.stdout.splitlines():
            # Surface real problems; VMD is otherwise extremely chatty.
            if line.startswith('ERROR)') and 'image file extension' not in line:
                print(f'  vmd: {line}')

    if external:
        for stem, tga, scene in jobs:
            if not os.path.exists(scene):
                raise SystemExit(f'render_movie.py: VMD wrote no scene for {stem}')
            patch_scene(scene, ao_samples, aa_samples, size)
            run_tachyon(tachyon, scene, tga)
            # Drop each scene as soon as it is consumed; a chunk's worth of
            # scenes plus TGAs together is a lot of scratch at large sizes.
            os.unlink(scene)

    return [(stem, tga) for stem, tga, _ in jobs]


def to_png(tga, png):
    from PIL import Image
    with Image.open(tga) as im:
        im.convert('RGB').save(png)


def main(argv=None):
    p = argparse.ArgumentParser(
        description='Render cube files to PNG frames with headless VMD.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--cubes', default='pacceptor/holecubes',
                   help='directory of .cube files')
    p.add_argument('--cube', action='append', metavar='FILE',
                   help='render just this cube (repeatable); overrides '
                        '--cubes/--start/--limit')
    p.add_argument('--outdir', default='pacceptor/frames', help='directory for the .png frames')
    p.add_argument('--settings', default='vmd_settings.tcl',
                   help='Tcl file defining setup_scene / setup_view / setup_frame')
    p.add_argument('--start', type=int, default=0, help='index of the first frame')
    p.add_argument('--limit', type=int, help='render only this many frames')
    p.add_argument('--size', type=int, nargs=2, default=[1920, 1080],
                   metavar=('W', 'H'),
                   help='image size in pixels; the camera in vmd_settings.tcl '
                        'is tuned for this 16:9 aspect, so changing the ratio '
                        '(not the resolution) means retuning its scale')
    p.add_argument('--renderer', choices=['external', 'internal', 'optix'],
                   default='external',
                   help='external: VMD writes a Tachyon scene, this script '
                        'lowers its sample counts and runs the tachyon binary. '
                        'internal: VMD renders directly, locked at 12/12 samples. '
                        'optix: GPU ray tracing with TachyonL-OptiX; needs a '
                        'CUDA/OptiX build of VMD')
    p.add_argument('--ao-samples', type=int, default=4, metavar='N',
                   help='ambient-occlusion rays per pixel (external and optix). '
                        'VMD writes 12; 4 is ~5x faster and visually equivalent '
                        'here, 1-2 is faster still but grainier')
    p.add_argument('--aa-samples', type=int, default=4, metavar='N',
                   help='antialiasing samples per pixel (external and optix)')
    p.add_argument('--tachyon', default=TACHYON,
                   help='standalone tachyon binary')
    p.add_argument('--upsample', type=int, default=1, metavar='N',
                   help='cubic-spline resample each cube Nx per axis before '
                        'VMD builds the isosurface.  1 = use the cube as is '
                        '(right for cubes written at CUBE_RESOLUTION 0.06); '
                        '2 = smooths old 80^3 cubes at ~1.8x the time')
    p.add_argument('--chunk', type=int, default=20,
                   help='frames per VMD launch (bounds scratch disk use); '
                        f'capped at {UPSAMPLED_CHUNK} when upsampling')
    p.add_argument('--vmd', default=shutil.which('vmd') or 'vmd', help='vmd executable')
    p.add_argument('--keep-tga', action='store_true',
                   help='keep the intermediate Targa files')
    p.add_argument('--quiet', action='store_true', help='suppress VMD warnings')
    args = p.parse_args(argv)

    settings = args.settings if os.path.isabs(args.settings) \
        else os.path.join(HERE, args.settings)
    if not os.path.exists(settings):
        raise SystemExit(f'render_movie.py: no settings file: {settings}')
    if not shutil.which(args.vmd) and not os.path.exists(args.vmd):
        raise SystemExit(f'render_movie.py: vmd not found: {args.vmd}')

    if args.cube:
        missing = [c for c in args.cube if not os.path.isfile(c)]
        if missing:
            raise SystemExit(f'render_movie.py: no such cube: {missing[0]}')
        cubes = args.cube
    else:
        cubes = find_cubes(args.cubes, args.start, args.limit)
    if args.upsample < 1:
        raise SystemExit('render_movie.py: --upsample must be at least 1')
    chunk = min(args.chunk, UPSAMPLED_CHUNK) if args.upsample > 1 else args.chunk
    os.makedirs(args.outdir, exist_ok=True)
    tgadir = args.outdir if args.keep_tga else tempfile.mkdtemp(prefix='vmdtga_')

    print(f'rendering {len(cubes)} frames at {args.size[0]}x{args.size[1]}')
    print(f'  cubes    {os.path.dirname(cubes[0])}/  ({os.path.basename(cubes[0])} .. '
          f'{os.path.basename(cubes[-1])})')
    print(f'  settings {os.path.relpath(settings, HERE)}')
    if args.renderer == 'external':
        print(f'  renderer external tachyon, AO {args.ao_samples} / '
              f'AA {args.aa_samples} samples')
    elif args.renderer == 'optix':
        print(f'  renderer TachyonL-OptiX (GPU), AO {args.ao_samples} / '
              f'AA {args.aa_samples} samples')
    else:
        print('  renderer TachyonInternal (12/12 samples, not adjustable)')
    print(f'  grid     {"raw" if args.upsample == 1 else f"{args.upsample}x spline upsampled"}, '
          f'{chunk} frames per VMD launch')
    print(f'  output   {args.outdir}/\n')

    t0 = time.time()
    done = 0
    try:
        for c0 in range(0, len(cubes), chunk):
            batch = cubes[c0:c0 + chunk]
            jobs = render_chunk(args.vmd, batch, tgadir, settings, args.size,
                                args.quiet, renderer=args.renderer,
                                ao_samples=args.ao_samples,
                                aa_samples=args.aa_samples,
                                tachyon=args.tachyon,
                                upsample=args.upsample)
            for stem, tga in jobs:
                if not os.path.exists(tga):
                    hint = ''
                    if args.renderer == 'optix':
                        hint = ('.  Check that this VMD supports the GPU '
                                'renderer: `echo "render list; quit" | vmd '
                                '-dispdev text` must list TachyonLOptiXInternal')
                    raise SystemExit(f'render_movie.py: VMD produced no output '
                                     f'for {stem}.cube{hint}')
                to_png(tga, os.path.join(args.outdir, stem + '.png'))
                if not args.keep_tga:
                    os.unlink(tga)
                done += 1
            el = time.time() - t0
            print(f'\r  {done}/{len(cubes)} frames, {el:.0f}s elapsed, '
                  f'{el / done:.2f}s/frame, ~{el / done * (len(cubes) - done):.0f}s left',
                  end='', flush=True)
    finally:
        if not args.keep_tga and os.path.isdir(tgadir):
            shutil.rmtree(tgadir, ignore_errors=True)

    print(f'\n\ndone: {done} PNGs in {args.outdir}/ in {time.time() - t0:.0f}s')
    print('  stitch into a movie with:')
    print(f'    ffmpeg -framerate 30 -i {args.outdir}/%d.png '
          f'-c:v libx264 -pix_fmt yuv420p movie.mp4')


if __name__ == '__main__':
    sys.exit(main())
