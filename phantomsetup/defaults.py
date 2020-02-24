"""Default values for all Phantom values."""

import datetime

import numpy as np
import phantomconfig

PHANTOM_VERSION = '0.0.0'
PHANTOM_GIT_HASH = 'xxxxxxx'

COMPILE_OPTIONS = {
    'DISC_VISCOSITY': False,
    'DRIVING': False,
    'DUST': False,
    'DUSTGROWTH': False,
    'GRAVITY': False,
    'H2CHEM': False,
    'IND_TIMESTEPS': True,
    'INJECT_PARTICLES': False,
    'ISOTHERMAL': False,
    'KERNEL': 'cubic',
    'LIGHTCURVE': False,
    'MAXDUSTSMALL': 11,
    'MAXDUSTLARGE': 11,
    'MCFOST': False,
    'MHD': False,
    'NONIDEALMHD': False,
    'PERIODIC': False,
    'PHOTO': False,
}

KERNEL_HFACT = {
    'cubic': 1.2,
    'quartic': 1.1,
    'quintic': 1.0,
    'WendlandC2': 1.3,
    'WendlandC4': 1.5,
    'WendlandC6': 1.6,
}

KERNELS = KERNEL_HFACT.keys()

PARTICLE_TYPE = {
    'igas': 1,
    'iboundary': 3,
    'istar': 4,
    'idarkmatter': 5,
    'ibulge': 6,
    'idust': 7,
    'idustlast': 7 + COMPILE_OPTIONS['MAXDUSTLARGE'] - 1,
    'iunknown': 0,
}

EXTERNAL_FORCES = (
    'star',
    'corotate',
    'binary',
    'prdrag',
    'torus',
    'toystar',
    'external',
    'spiral',
    'Lense',
    'neutronstar',
    'Einstein',
    'generalised',
    'static',
    'grav',
    'disc',
    'corotating',
)

_EXTERNAL_FORCES_COMMENT = ''
for idx, ext in enumerate(EXTERNAL_FORCES):
    _EXTERNAL_FORCES_COMMENT += f'{idx}={ext[:4]},'
_EXTERNAL_FORCES_COMMENT = _EXTERNAL_FORCES_COMMENT[:-1]

MAXTYPES = 7 + COMPILE_OPTIONS['MAXDUSTLARGE'] - 1
MAXDUST = COMPILE_OPTIONS['MAXDUSTSMALL'] + COMPILE_OPTIONS['MAXDUSTLARGE']

RUN_OPTION_BLOCK_LABEL = {
    'accuracy': 'options controlling accuracy',
    'cooling': 'options controlling cooling',
    'damping': 'options controlling damping',
    'driving': 'options controlling forcing of turbulence',
    'dust': 'options controlling dust',
    'dustgrowth': 'options controlling dust growth',
    'eos': 'options controlling equation of state',
    'external forces': 'options relating to external forces',
    'hydrodynamics': 'options controlling hydrodynamics, artificial dissipation',
    'inject': 'options for injecting particles',
    'io': 'options controlling run time and input/output',
    'io supplementary': (
        'options controlling run time and input/output: supplementary features'
    ),
    'job': 'job name',
    'MCFOST': 'options controlling MCFOST',
    'non-ideal MHD': 'options controlling non-ideal MHD',
    'photoevaporation': 'options controlling photoevaporation',
    'sinks': 'options controlling sink particles',
    'viscosity': 'options controlling physical viscosity',
}

_RUN_OPTIONS = {
    # ------------------------------------------------
    # job name
    'job name': {
        'logfile': ('prefix01.log', 'file to which output is directed'),
        'dumpfile': ('prefix_00000.tmp', 'dump file to start from'),
    },
    # ------------------------------------------------
    # options controlling run time and input/output
    'options controlling run time and input/output': {
        'tmax': (10.0, 'end time'),
        'dtmax': (1.0, 'time between dumps'),
        'nmax': (-1, 'maximum number of timesteps (0=just get derivs and stop)'),
        'nout': (-1, 'number of steps between dumps (-ve=ignore)'),
        'nmaxdumps': (-1, 'stop after n full dumps (-ve=ignore)'),
        'twallmax': (
            datetime.timedelta(hours=0),
            'maximum wall time (hhh:mm, 000:00=ignore)',
        ),
        'dtwallmax': (
            datetime.timedelta(hours=12),
            'maximum wall time between dumps (hhh:mm, 000:00=ignore)',
        ),
        'nfulldump': (10, 'full dump every n dumps'),
        'iverbose': (
            0,
            'verboseness of log (-1=quiet 0=default 1=allsteps 2=debug 5=max)',
        ),
    },
    # ------------------------------------------------
    # options controlling run time and input/output: supplementary features
    'options controlling run time and input/output: supplementary features': {
        'rhofinal_cgs': (0.0, 'maximum allowed density (cgs) (<=0 to ignore)'),
        'dtmax_dratio': (
            0.0,
            'dynamic dtmax: density ratio controlling decrease (<=0 to ignore)',
        ),
        'dtmax_max': (-1.0, 'dynamic dtmax: maximum allowed dtmax (=dtmax if <= 0)'),
        'dtmax_min': (0.0, 'dynamic dtmax: minimum allowed dtmax'),
        'calc_erot': (False, 'include E_rot in the ev_file'),
    },
    # ------------------------------------------------
    # options controlling accuracy
    'options controlling accuracy': {
        'C_cour': (0.3, 'Courant number'),
        'C_force': (0.25, 'dt_force number'),
        'tolv': (1.0e-2, 'tolerance on v iterations in timestepping'),
        'hfact': (1.2, 'h in units of particle spacing [h = hfact(m/rho)^(1/3)]'),
        'tolh': (1.0e-4, 'tolerance on h-rho iterations'),
        'tree_accuracy': (0.5, 'tree opening criterion (0.0-1.0)'),
    },
    # ------------------------------------------------
    # options controlling hydrodynamics, artificial dissipation
    'options controlling hydrodynamics, artificial dissipation': {
        'alpha': (1.0, 'MINIMUM art. viscosity parameter'),
        'alphamax': (1.0, 'MAXIMUM art. viscosity parameter'),
        'alphau': (1.0, 'art. conductivity parameter'),
        'beta': (2.0, 'beta viscosity'),
        'alphaB': (1.0, 'art. resistivity parameter'),
        'psidecayfac': (1.0, 'div B diffusion parameter'),
        'overcleanfac': (
            1.0,
            'factor to increase cleaning speed (decreases time step)',
        ),
        'avdecayconst': (0.1, 'decay time constant for viscosity switches'),
    },
    # ------------------------------------------------
    # options controlling damping
    'options controlling damping': {
        'idamp': (0, 'artificial damping of velocities (0=off, 1=constant, 2=star)'),
        'damp': (0.0, 'artificial damping of velocities (if on, v=0 initially)'),
        'tdyn_s': (
            0.0,
            'dynamical timescale of star in seconds - damping is dependent on it',
        ),
    },
    # ------------------------------------------------
    # options controlling equation of state
    'options controlling equation of state': {
        'ieos': (1, 'eqn of state (1=isoth;2=adiab;3=locally iso;8=barotropic)'),
        'mu': (2.381, 'mean molecular weight'),
        'ipdv_heating': (1, 'heating from PdV work (0=off, 1=on)'),
        'ishock_heating': (1, 'shock heating (0=off, 1=on)'),
        'iresistive_heating': (1, 'resistive heating (0=off, 1=on)'),
        'drhocrit': (
            0.50,
            'transition size between rhocrit0 & 1 '
            '(fraction of rhocrit0; barotropic eos)',
        ),
        'rhocrit0': (1.0e-18, 'critical density 0 in g/cm^3 (barotropic eos)'),
        'rhocrit1': (1.0e-14, 'critical density 1 in g/cm^3 (barotropic eos)'),
        'rhocrit2': (1.0e-10, 'critical density 2 in g/cm^3 (barotropic eos)'),
        'rhocrit3': (1.0e-3, 'critical density 3 in g/cm^3 (barotropic eos)'),
        'gamma1': (1.4, 'adiabatic index 1 (barotropic eos)'),
        'gamma2': (1.1, 'adiabatic index 2 (barotropic eos)'),
        'gamma3': (5 / 3, 'adiabatic index 3 (barotropic eos)'),
        'rhocrit0pwp': (
            2.62780e12,
            'critical density 0 in g/cm^3 (piecewise polytropic eos)',
        ),
        'rhocrit1pwp': (
            5.01187e14,
            'critical density 1 in g/cm^3 (piecewise polytropic eos)',
        ),
        'rhocrit2pwp': (
            1.0e15,
            'critical density 2 in g/cm^3 (piecewise polytropic eos)',
        ),
        'gamma0pwp': (5 / 3, 'adiabatic index 0 (piecewise polytropic eos)'),
        'gamma1pwp': (3.166, 'adiabatic index 1 (piecewise polytropic eos)'),
        'gamma2pwp': (3.573, 'adiabatic index 2 (piecewise polytropic eos)'),
        'gamma3pwp': (3.281, 'adiabatic index 3 (piecewise polytropic eos)'),
        'p1pwp': (
            2.46604e34,
            'pressure at cutoff density rhocrit1pwp (piecewise polytropic eos)',
        ),
        'X': (0.74, 'hydrogen mass fraction'),
        'Z': (0.02, 'metallicity'),
        'relaxflag': (0, '0=evolve, 1=relaxation on (keep T const)'),
    },
    # ------------------------------------------------
    # options controlling cooling
    'options controlling cooling': {
        'icooling': (0, 'cooling function (0=off, 1=on)'),
        'C_cool': (0.05, 'factor controlling cooling timestep'),
        'beta_cool': (3.0, 'beta factor in Gammie (2001) cooling'),
        'dlq': (3.086e19, 'distance for column density in cooling function'),
        'dphot': (1.0801e20, 'photodissociation distance used for CO/H2'),
        'dphotflag': (
            0,
            'photodissociation distance static or radially adaptive (0/1)',
        ),
        'dchem': (3.086e20, 'distance for chemistry of HI'),
        'abundc': (1.4e-4, 'Carbon abundance'),
        'abundo': (3.2e-4, 'Oxygen abundance'),
        'abundsi': (1.5e-5, 'Silicon abundance'),
        'abunde': (2.0e-4, 'electron abundance'),
        'uv_field_strength': (1.0, 'Strength of UV field (in Habing units)'),
        'dust_to_gas_ratio': (1.0, 'dust to gas ratio'),
        'AV_conversion_factor': (
            5.348e-22,
            'Extinction per unit column density (cm^-2)',
        ),
        'cosmic_ray_ion_rate': (1.0e-17, 'Cosmic ray ionisation rate of H1 (in s^-1)'),
        'iphoto': (
            1,
            'Photoelectric heating treatment (0=optically thin, 1=w/extinction)',
        ),
        'iflag_atom': (1, 'Which atomic cooling (1:Gal ISM, 2:Z=0 gas)'),
        'cooltable': ('cooltable.dat', 'data file containing cooling function'),
        'habund': (0.7, 'Hydrogen abundance assumed in cooling function'),
        'temp_floor': (1.0e4, 'Minimum allowed temperature in K'),
    },
    # ------------------------------------------------
    # options controlling MCFOST
    'options controlling MCFOST': {
        'use_mcfost': (False, 'use the mcfost library'),
        'Voronoi_limits_file': ('limits', 'Limit file for the Voronoi tesselation'),
        'use_mcfost_stars': (
            False,
            'Fix the stellar parameters to mcfost values or update using sink mass',
        ),
    },
    # ------------------------------------------------
    # options controlling sink particles
    'options controlling sink particles': {
        'icreate_sinks': (0, 'allow automatic sink particle creation'),
        'rho_crit_cgs': (
            1.0e-10,
            'density above which sink particles are created (g/cm^3)',
        ),
        'r_crit': (
            5.0e-3,
            'critical radius for point mass creation (no new sinks < r_crit from '
            'existing sink)',
        ),
        'h_acc': (1.0e-3, 'accretion radius for new sink particles'),
        'h_soft_sinkgas': (0.0, 'softening length for new sink particles'),
        'h_soft_sinksink': (0.0, 'softening length between sink particles'),
        'f_acc': (0.8, 'particles < f_acc*h_acc accreted without checks'),
    },
    # ------------------------------------------------
    # options relating to external forces
    # TODO: this is unfinished
    'options relating to external forces': {
        'iexternalforce': (0, _EXTERNAL_FORCES_COMMENT)
    },
    # ------------------------------------------------
    # options controlling physical viscosity
    'options controlling physical viscosity': {
        'irealvisc': (0, 'physical viscosity type (0=none,1=const,2=Shakura/Sunyaev)'),
        'shearparam': (
            0.1,
            'magnitude of shear viscosity (irealvisc=1) or alpha_SS (irealvisc=2)',
        ),
        'bulkvisc': (0.0, 'magnitude of bulk viscosity'),
    },
    # ------------------------------------------------
    # options controlling forcing of turbulence
    'options controlling forcing of turbulence': {
        'istir': (1, 'switch to turn stirring on or off at runtime'),
        'st_spectform': (1, 'spectral form of stirring'),
        'st_stirmax': (18.86, 'maximum stirring wavenumber'),
        'st_stirmin': (6.28, 'minimum stirring wavenumber'),
        'st_energy': (2.0, 'energy input/mode'),
        'st_decay': (0.05, 'correlation time for driving'),
        'st_solweight': (1.0, 'solenoidal weight'),
        'st_dtfreq': (0.01, 'frequency of stirring'),
        'st_seed': (1, 'random number generator seed'),
        'st_amplfac': (1.0, 'amplitude factor for stirring of turbulence'),
    },
    # ------------------------------------------------
    # options controlling dust
    'options controlling dust': {
        'idrag': (2, 'gas/dust drag (0=off,1=Epstein/Stokes,2=const K,3=const ts)'),
        'K_code': (1.0, 'drag constant when constant drag is used'),
        'icut_backreaction': (0, 'cut the drag on the gas phase (0=no, 1=yes)'),
        'ilimitdustflux': (False, 'limit the dust flux using Ballabio et al. (2018)'),
        'grainsize': (1.0, 'Grain size in cm'),
        'graindens': (3.0, 'Intrinsic grain density in g/cm^3'),
    },
    # ------------------------------------------------
    # options controlling growth
    'options controlling growth': {
        'ifrag': (0, 'dust fragmentation (0=off,1=on,2=Kobayashi)'),
        'grainsizemin': (1.0e-3, 'minimum grain size in cm'),
        'isnow': (0, 'snow line (0=off,1=position based,2=temperature based)'),
        'rsnow': (100.0, 'position of the snow line in AU'),
        'Tsnow': (20.0, 'snow line condensation temperature in K'),
        'vfrag': (15.0, 'uniform fragmentation threshold in m/s'),
        'vfragin': (5.0, 'inward fragmentation threshold in m/s'),
        'vfragout': (15.0, 'outward fragmentation threshold in m/s'),
    },
    # ------------------------------------------------
    # options controlling photoevaporation
    'options controlling photoevaporation': {
        'mu_cgs': (1.26, 'Mean molecular weight'),
        'recombrate_cgs': (2.6e-13, 'Recombination rate (alpha)'),
        'ionflux_cgs': (1.0e41, 'Stellar EUV flux rate'),
    },
    # ------------------------------------------------
    # options for injecting particles
    # TODO
    # ------------------------------------------------
    # options controlling non-ideal MHD
    # TODO
}

RUN_OPTIONS = phantomconfig.read_dict(_RUN_OPTIONS, 'nested')

_RUN_OPTIONS = dict(zip(RUN_OPTIONS.variables, RUN_OPTIONS.values))

HEADER = {
    'Bextx': 0.0,
    'Bexty': 0.0,
    'Bextz': 0.0,
    'C_cour': _RUN_OPTIONS['C_cour'],
    'C_force': _RUN_OPTIONS['C_force'],
    'RK2': 1.5,
    'alpha': _RUN_OPTIONS['alpha'],
    'alphaB': _RUN_OPTIONS['alphaB'],
    'alphau': _RUN_OPTIONS['alphau'],
    'angtot_in': 0.0,
    'dtmax': _RUN_OPTIONS['dtmax'],
    'dum': 0.0,
    'etot_in': 0.0,
    'fileident': '',
    'gamma': 1.0,
    'get_conserv': 1.0,
    'graindens': np.zeros(MAXDUST),
    'grainsize': np.zeros(MAXDUST),
    'hfact': _RUN_OPTIONS['hfact'],
    'idust': PARTICLE_TYPE['idust'],
    'ieos': _RUN_OPTIONS['ieos'],
    'iexternalforce': _RUN_OPTIONS['iexternalforce'],
    'isink': 0,
    'majorv': PHANTOM_VERSION.split('.')[0],
    'massoftype': np.zeros(MAXTYPES),
    'microv': PHANTOM_VERSION.split('.')[1],
    'minorv': PHANTOM_VERSION.split('.')[2],
    'mdust_in': np.zeros(MAXDUST),
    'ndustlarge': 0,
    'ndustsmall': 0,
    'npartoftype': np.zeros(MAXTYPES, dtype=np.int),
    'nparttot': 0,
    'nptmass': 0,
    'ntypes': 0,
    'polyk2': 0.0,
    'qfacdisc': 0.75,
    'rhozero': 1.0,
    'time': 0.0,
    'tolh': _RUN_OPTIONS['tolh'],
    'totmom_in': 0.0,
    'udist': 1.0,
    'umass': 1.0,
    'umagfd': 1.0,
    'utime': 1.0,
    'xmax': 0.5,
    'xmin': -0.5,
    'ymax': 0.5,
    'ymin': -0.5,
    'zmax': 0.5,
    'zmin': -0.5,
}
