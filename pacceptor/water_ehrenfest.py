import os
n_threads = os.environ.get('SLURM_CPUS_PER_TASK', '2')
os.environ['MKL_NUM_THREADS'] = n_threads
os.environ['OPENBLAS_NUM_THREADS'] = n_threads
os.environ['OMP_NUM_THREADS'] = n_threads
os.environ['MKL_DYNAMIC'] = 'FALSE'

import numpy as np
from pyscf import lib
lib.num_threads(int(n_threads))

from pyscf import gto, dft
from tides.utils import basis_utils, rt_utils
from tides.potentials import rt_cap
from tides.methods import rt_ehrenfest

dimer = gto.Mole()
water_donor = gto.Mole()
water_acceptor = gto.Mole()

dimer.atom = '''
 O               -0.33775152    -1.46032090    -0.01334192
 H               -1.28517880    -1.53293364    -0.13253239
 H               -0.14613489    -0.50477300    -0.04545198
 O                0.30554734     1.30417058     0.06747124
 H                0.85907109     1.51153823     0.81419560
 H                0.74987713     1.62842427    -0.72567413
'''
water_donor.atom = '''
 O               -0.33775152    -1.46032090    -0.01334192
 H               -1.28517880    -1.53293364    -0.13253239
 H               -0.14613489    -0.50477300    -0.04545198
'''
water_acceptor.atom = '''
 O                0.30554734     1.30417058     0.06747124
 H                0.85907109     1.51153823     0.81419560
 H                0.74987713     1.62842427    -0.72567413
'''
water_donor.spin = 0; water_donor.charge = 0
water_acceptor.spin = 0; water_acceptor.charge = 0

dimer.basis = 'augccpvdz'
water_donor.basis = 'augccpvdz'
water_acceptor.basis = 'augccpvdz'

dimer.build()
water_donor.build()
water_acceptor.build()

# LC-PBE 
dimer = dft.UKS(dimer); dimer.xc = 'HYB_GGA_XC_LC_PBEOP, -1 * GGA_C_OP_PBE + PBE'; dimer._numint.omega = 0.516; dimer._numint.alpha = 0.0; dimer._numint.beta = 1.0
water_donor = dft.UKS(water_donor); water_donor.xc = 'HYB_GGA_XC_LC_PBEOP, -1 * GGA_C_OP_PBE + PBE'; water_donor._numint.omega = 0.516; water_donor._numint.alpha = 0.0; water_donor._numint.beta = 1.0
water_acceptor = dft.UKS(water_acceptor); water_acceptor.xc = 'HYB_GGA_XC_LC_PBEOP, -1 * GGA_C_OP_PBE + PBE'; water_acceptor._numint.omega = 0.516; water_acceptor._numint.alpha = 0.0; water_acceptor._numint.beta = 1.0

dimer.kernel()
water_donor.kernel()
water_acceptor.kernel()

rt_water = rt_ehrenfest.RT_Ehrenfest(dimer, 1.0, 1230,
                                     filename="acceptor_excite.out", prop="magnus_interpol",
                                     frequency=1, chkfile="acceptor_excite.chk", verbose=6,
                                     Ne_step=1, N_step=1, get_mo_coeff_print = rt_utils.get_noscf_orbitals)
rt_water.nuc.mass[0] = 29156.945034195
rt_water.nuc.mass[1] = 3670.483014130 
rt_water.nuc.mass[2] = 3670.483014130 
rt_water.nuc.mass[3] = 29156.945034195
rt_water.nuc.mass[4] = 3670.483014130 
rt_water.nuc.mass[5] = 3670.483014130 
np.random.seed(4)
rt_water.nuc.sample_vel(beta = 10526.25)
# Declare which observables to be calculated/printed
rt_water.observables.update(energy=True, mo_occ=True, charge=True, atom_charge=True, nuclei=True)

# Create object for complex absorbing potential and add to rt object
CAP = rt_cap.MOCAP(0.5, 0.0477, 1.0, 10.0)
rt_water.add_potential(CAP)

# Remove electron from molecular orbital (in SCF basis)
# Input the two water fragments for their charge to be calculated
rt_utils.excite(rt_water, 3)
rt_water._scf.mo_occ = rt_water.occ
rt_utils.input_fragments(rt_water, water_donor, water_acceptor)


from pyscf.tools import cubegen

# Cube grid for the movie.  cubegen's default is a fixed 80 points per axis,
# which over this dimer's box is only ~0.12-0.16 Bohr spacing (and different
# per axis) -- coarse enough that VMD's isosurfaces come out visibly faceted.
# Setting the spacing instead gives the same fine grid in every direction and
# every frame; the box still follows the molecule with a CUBE_MARGIN border.
# At 0.06 Bohr: ~168x200x149 points, ~66 MB and ~4 s per cube.
CUBE_RESOLUTION = 0.06   # Bohr
CUBE_MARGIN = 3.0        # Bohr; |hole density| > 0.002 reaches ~3 Bohr out

def get_cube_holedensity(rt_obj):
    '''
    Will create Gaussian cube file for molecule electron density
    for every propagation time given in rt_obj.cube_density_indices.
    '''
    if np.rint(rt_obj.current_time/rt_obj.timestep) in np.rint(np.array(rt_obj.cube_density_indices)*rt_obj.frequency):
        cube_name = f'{rt_obj.cube_filename}{round(rt_obj.current_time)}.cube'
        mo_coeff = np.copy(rt_obj._scf.mo_coeff)
        mo_occ = np.copy(rt_obj._scf.mo_occ)
        basis, labels, pos = basis_utils._read_mol(rt_obj._scf.mol)
        spin, charge = rt_obj._scf.mol.spin, rt_obj._scf.mol.charge
        rt_obj._scf.reset(basis_utils._write_mol(basis, labels, pos, spin, charge))
        rt_obj._scf.kernel()
        dm_scf = rt_obj._scf.make_rdm1()
        rt_obj._scf.mo_coeff = mo_coeff
        rt_obj._scf.mo_occ = mo_occ
        dm_hole = dm_scf - rt_obj.den_ao
        cubegen.density(rt_obj._scf.mol, cube_name, dm_hole[0] + dm_hole[1],
                        resolution=CUBE_RESOLUTION, margin=CUBE_MARGIN)

rt_water.observables.update({'cube_holedensity' : True})
rt_water._observables_functions.update({'cube_holedensity' : [get_cube_holedensity, lambda *args: None]})
rt_water.cube_filename = 'holecubes/'
rt_water.cube_density_indices = np.arange(1230)

rt_water.kernel()

