# ---------------------------------------------------------------------------
# Graphics settings for the hole-density movie.
#
# This is the file to edit when tuning how the movie looks.  render_movie.py
# sources it and calls the three procs below; nothing else in the pipeline
# needs to change.
#
#   setup_scene           once per VMD launch  -- display, background, lighting
#   setup_view   <molid>  once per frame       -- camera orientation and zoom
#   setup_frame  <molid>  once per frame       -- representations for that cube
#
# Quick preview of a single frame while you tune:
#   vmd -e vmd_settings.tcl pacceptor/holecubes/0.cube
# (the trailing block at the bottom of this file handles that case)
# ---------------------------------------------------------------------------

# Isosurface level, in e/Bohr^3.  The hole density peaks at ~2.0 right on the
# oxygen nucleus, but the interesting charge redistribution between the two
# waters lives around 1e-2 .. 1e-3.  Raise this to see only the core, lower it
# to bring out the diffuse ICD features.
set ISOVALUE 0.005

# The hole density is signed: positive = density removed relative to the
# ground state, negative = density gained.
set COLOR_POSITIVE  red
set COLOR_NEGATIVE  blue
set ISO_MATERIAL    Transparent
set ATOM_MATERIAL   Opaque

# CPK atom representation: sphere scale (x vdW radius) and bond radius (Å).
set ATOM_SPHERE_SCALE 0.8
set ATOM_BOND_RADIUS  0.3

# --- framing ---------------------------------------------------------------
# How tightly the subject sits in the frame.  These three are what you retune
# whenever the rendered extent changes -- a new ISOVALUE, a different ROT_*,
# a heavier atom representation, or a different aspect ratio.
#
# VIEW_ZOOM     multiplies VMD's `display resetview`, which fits the *nuclei*
#               with a wide margin and knows nothing about the isosurface.
#               Larger = tighter crop.
# VIEW_SHIFT_X  recentres horizontally; the hole sits on one water, so the
#               rendered content is offset from the nuclear centre.  Units are
#               odd: 0.1 moves the image about 1.4% of the frame width.
# VIEW_SHIFT_Y  same, vertically.
#
# Don't tune these by eye -- fit_view.py measures the rendered content and
# solves for them:
#     python fit_view.py                # report what they should be
#     python fit_view.py --apply        # rewrite the three values below
#
# Current values: fit_view.py --margin 0.08, probes 0..870, ROT 180/0/90.
set VIEW_ZOOM     1.74
set VIEW_SHIFT_X  0.35
set VIEW_SHIFT_Y  -0.35

# The cube the camera is derived from.  `display resetview` runs on this cube
# once and its result is reused for every frame, so the camera never depends on
# which frame is being rendered or which frame a VMD launch happened to load
# first.  The nuclei move in this run, so any other choice makes the view jump.
# Relative paths resolve against this file's directory, then the cwd.
set VIEW_REFERENCE pacceptor/holecubes/0.cube
set SETTINGS_DIR [file dirname [file normalize [info script]]]

# --- camera orientation ----------------------------------------------------
# Fixed for the whole movie -- the camera never moves.  Degrees about the
# screen axes, applied x, then y, then z.
set ROT_X        180
set ROT_Y        180
set ROT_Z        100


proc setup_scene {} {
    display projection Orthographic
    display depthcue   off
    display ambientocclusion on
    display aoambient  0.9
    display aodirect   0.4
    display shadows    on
    display antialias  on
    axes location Off
    color Display Background white
    # Keep the isosurface from being clipped when it extends past the atoms.
    display nearclip set 0.01
}


proc setup_view {molid} {
    global LOCKED_RESETVIEW VIEW_ZOOM VIEW_SHIFT_X VIEW_SHIFT_Y VIEW_REFERENCE
    global ROT_X ROT_Y ROT_Z

    # The camera must be pixel-identical on every frame, or the picture
    # visibly breathes as the isosurface grows and shrinks.
    #
    # `display resetview` is the one step that inspects the molecule, and the
    # nuclei move in this run, so it must never run on the frame itself: frames
    # 0 and 870 give a ~10% different zoom and a visible pan.  It runs once, on
    # VIEW_REFERENCE, and the matrices are cached for the life of this VMD
    # process.  Every launch (render_movie.py starts one per chunk) re-derives
    # the same matrices from the same cube, so chunk boundaries are seamless.
    # Cube atom coordinates are absolute, so the reference's centre matrix is
    # valid for any frame -- the molecule simply moves within a still shot.
    if {![info exists LOCKED_RESETVIEW]} {
        set ref [view_reference_path]
        set refid [mol new $ref type cube waitfor all]
        display resetview
        set LOCKED_RESETVIEW [molinfo $refid get \
            {center_matrix rotate_matrix scale_matrix global_matrix}]
        mol delete $refid
        mol top $molid
    }
    molinfo $molid set {center_matrix rotate_matrix scale_matrix global_matrix} \
            $LOCKED_RESETVIEW
    # Everything after this is arithmetic on constants.

    # ---- tune the camera here -------------------------------------------
    # Keep the x/y/z order -- VMD applies each rotation about the screen axes,
    # so reordering them changes the result.
    rotate x by $ROT_X
    rotate y by $ROT_Y
    rotate z by $ROT_Z

    # Zoom and centring live at the top of this file; fit_view.py sets them.
    scale by $VIEW_ZOOM
    translate by $VIEW_SHIFT_X $VIEW_SHIFT_Y 0.0
    # ---------------------------------------------------------------------
}


proc view_reference_path {} {
    global VIEW_REFERENCE SETTINGS_DIR
    if {[file pathtype $VIEW_REFERENCE] ne "relative"} { return $VIEW_REFERENCE }
    # fit_view.py sources a temp copy of this file, so SETTINGS_DIR is not
    # always the repo; VMD's cwd (the repo root) is the fallback.
    foreach cand [list [file join $SETTINGS_DIR $VIEW_REFERENCE] $VIEW_REFERENCE] {
        if {[file exists $cand]} { return $cand }
    }
    error "vmd_settings.tcl: VIEW_REFERENCE not found: $VIEW_REFERENCE"
}


# Forget the cached camera so the next setup_view re-derives it.  Useful when
# tuning interactively: edit the numbers above, then
#     source vmd_settings.tcl; unlock_view; setup_view [molinfo top]
proc unlock_view {} {
    global LOCKED_RESETVIEW
    catch {unset LOCKED_RESETVIEW}
}


proc setup_frame {molid} {
    global ISOVALUE COLOR_POSITIVE COLOR_NEGATIVE ISO_MATERIAL ATOM_MATERIAL
    global ATOM_SPHERE_SCALE ATOM_BOND_RADIUS

    # Drop the default Lines representation that `mol new` creates.
    mol delrep 0 $molid

    # --- the nuclei -------------------------------------------------------
    mol representation CPK $ATOM_SPHERE_SCALE $ATOM_BOND_RADIUS 30 30
    mol color Name
    mol selection {all}
    mol material $ATOM_MATERIAL
    mol addrep $molid

    # --- positive lobe: density removed (the hole) ------------------------
    # Isosurface <value> <volID> <show> <draw> <step> <size>
    #   show: 0 isosurface only, 1 +box, 2 box only
    #   draw: 0 solid, 1 wireframe, 2 points
    mol representation Isosurface $ISOVALUE 0 0 0 1 1
    mol color ColorID [colorid $COLOR_POSITIVE]
    mol selection {all}
    mol material $ISO_MATERIAL
    mol addrep $molid

    # --- negative lobe: density gained ------------------------------------
    mol representation Isosurface [expr {-1.0 * $ISOVALUE}] 0 0 0 1 1
    mol color ColorID [colorid $COLOR_NEGATIVE]
    mol selection {all}
    mol material $ISO_MATERIAL
    mol addrep $molid
}


# Map a colour name to the ColorID index VMD wants in `mol color ColorID N`.
proc colorid {name} {
    set i [lsearch [colorinfo colors] $name]
    if {$i < 0} {
        puts "vmd_settings.tcl: unknown colour '$name', falling back to blue"
        return 0
    }
    return $i
}


# ---------------------------------------------------------------------------
# Interactive preview: if this file was given straight to vmd alongside a cube
# (`vmd -e vmd_settings.tcl pacceptor/holecubes/0.cube`), set the scene up right away.
# render_movie.py sets NO_AUTO_APPLY first so this block stays out of its way.
# ---------------------------------------------------------------------------
if {![info exists NO_AUTO_APPLY] && [molinfo num] > 0} {
    setup_scene
    setup_frame [molinfo top]
    setup_view  [molinfo top]
}
