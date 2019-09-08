import datetime

import numpy as np

MAJORV = 0
MINORV = 0
MICROV = 0

options = {
    # ------------------------------------------------
    # job name
    'logfile': 'prefix01.log',
    'dumpfile': 'prefix_00000.tmp',
    # ------------------------------------------------
    # options controlling run time and input/output
    'tmax': 0.1,
    'dtmax': 0.001,
    'nmax': -1,
    'nout': -1,
    'nmaxdumps': -1,
    'twallmax': datetime.timedelta(hours=0),
    'dtwallmax': datetime.timedelta(hours=12),
    'nfulldump': 1,
    'iverbose': 0,
    # ------------------------------------------------
    # options controlling accuracy
    'C_cour': 0.3,
    'C_force': 0.25,
    'tolv': 1.0e-2,
    'hfact': 1.0,
    'tolh': 1.0e-4,
    # ------------------------------------------------
    # options controlling hydrodynamics, artificial dissipation
    'alpha': 1.0,
    'alphamax': 1.0,
    'alphau': 1.0,
    'beta': 2.0,
    'alphaB': 1.0,
    'psidecayfac': 1.0,
    'overcleanfac': 1.0,
    'avdecayconst': 0.1,
    # ------------------------------------------------
    # options controlling damping
    'idamp': 0,
    'damp': 0.0,
    'tdyn_s': 0.0,
    # ------------------------------------------------
    # options controlling equation of state
    'ieos': 1,
    'mu': 2.381,
    'drhocrit': 0.50,
    'rhocrit0': 1.0e-18,
    'rhocrit1': 1.0e-14,
    'rhocrit2': 1.0e-10,
    'rhocrit3': 1.0e-3,
    'gamma1': 1.4,
    'gamma2': 1.1,
    'gamma3': 5 / 3,
    'rhocrit0pwp': 2.62780e12,
    'rhocrit1pwp': 5.01187e14,
    'rhocrit2pwp': 1.0e15,
    'gamma0pwp': 5 / 3,
    'gamma1pwp': 3.166,
    'gamma2pwp': 3.573,
    'gamma3pwp': 3.281,
    'p1pwp': 2.46604e34,
    'X': 0.74,
    'Z': 0.02,
    'relaxflag': 0,
    'ipdv_heating': 1,
    'ishock_heating': 1,
    'iresistive_heating': 1,
    'isink': 0,
    # ------------------------------------------------
    # options controlling cooling
    'icooling': 0,
    'C_cool': 0.05,
    'beta_cool': 3.0,
    'dlq': 3.086e19,
    'dphot': 1.0801e20,
    'dphotflag': 0,
    'dchem': 3.086e20,
    'abundc': 1.4e-4,
    'abundo': 3.2e-4,
    'abundsi': 1.5e-5,
    'abunde': 2.0e-4,
    'uv_field_strength': 1.0,
    'dust_to_gas_ratio': 1.0,
    'AV_conversion_factor': 5.348e-22,
    'cosmic_ray_ion_rate': 1.0e-17,
    'iphoto': 1,
    'iflag_atom': 1,
    'cooltable': 'cooltable.dat',
    'habund': 0.7,
    'temp_floor': 1.0e4,
    # ------------------------------------------------
    # options controlling magnetic fields
    'Bextx': 0.0,
    'Bexty': 0.0,
    'Bextz': 0.0,
    # ------------------------------------------------
    # options controlling MCFOST
    'use_mcfost': False,
    'Voronoi_limits_file': 'limits',
    'use_mcfost_stars': False,
    # ------------------------------------------------
    # options controlling sink particles
    'icreate_sinks': 0,
    'rho_crit_cgs': 1.0e-10,
    'r_crit': 5.0e-3,
    'h_acc': 1.0e-3,
    'h_soft_sinkgas': 0.0,
    'h_soft_sinksink': 0.0,
    'f_acc': 0.8,
    # ------------------------------------------------
    # options relating to external forces
    # TODO: this is unfinished
    'iexternalforce': 0,
    # ------------------------------------------------
    # options controlling physical viscosity
    'irealvisc': 0,
    'shearparam': 0.1,
    'bulkvisc': 0.0,
    # ------------------------------------------------
    # options controlling forcing of turbulence
    # TODO
    # ------------------------------------------------
    # options controlling dust
    'idrag': 2,
    'K_code': 1.0,
    'ndustsmall': 0,
    'ndustlarge': 0,
    'icut_backreaction': 0,
    'ilimitdustflux': False,
    # ------------------------------------------------
    # options controlling growth
    'ifrag': 0,
    'grainsizemin': 1.0e-3,
    'isnow': 0,
    'rsnow': 100.0,
    'Tsnow': 20.0,
    'vfrag': 15.0,
    'vfragin': 5.0,
    'vfragout': 15.0,
    # ------------------------------------------------
    # options controlling photoevaporation
    'mu_cgs': 1.26,
    'recombrate_cgs': 2.6e-13,
    'ionflux_cgs': 1.0e41,
    # ------------------------------------------------
    # options for injecting particles
    # TODO
    # ------------------------------------------------
    # options controlling non-ideal MHD
    # TODO
    # ------------------------------------------------
    # options from initial params
    'angtot_in': 0.0,
    'get_conserv': 1.0,
    'etot_in': 0.0,
    'totmom_in': 0.0,
    'mdust_in': 0.0,
}

header = {
    'Bextx': options['Bextx'],
    'Bexty': options['Bexty'],
    'Bextz': options['Bextz'],
    'C_cour': options['C_cour'],
    'C_force': options['C_force'],
    'RK2': 1.5,
    'alpha': options['alpha'],
    'alphaB': options['alphaB'],
    'alphau': options['alphau'],
    'angtot_in': options['angtot_in'],
    'dtmax': options['dtmax'],
    'dum': 0.0,
    'etot_in': options['etot_in'],
    'gamma': 1.0,
    'get_conserv': options['get_conserv'],
    'graindens': np.zeros(100),
    'grainsize': np.zeros(100),
    'hfact': options['hfact'],
    'ieos': options['ieos'],
    'iexternalforce': options['iexternalforce'],
    'isink': options['isink'],
    'majorv': MAJORV,
    'massoftype': np.zeros(100),
    'microv': MICROV,
    'minorv': MINORV,
    'mdust_in': np.zeros(100),
    'ndustlarge': options['ndustlarge'],
    'ndustsmall': options['ndustsmall'],
    'npartoftype': np.zeros(100, dtype=np.int),
    'nparttot': 0,
    'nptmass': 0,
    'ntypes': 0,
    'polyk2': 1.0,
    'qfacdisc': 1.0,
    'rhozero': 1.0,
    'time': 0.0,
    'tolh': options['tolh'],
    'totmom_in': options['totmom_in'],
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
