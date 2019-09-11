import datetime
from typing import Dict, Tuple

from .defaults import run_option_block_label as block_label


class _InFile:
    """
    Phantom in file.

    For internal use.

    Parameters
    ----------
    compile_options : Dict
        TODO: ...
    run_options : Dict
        TODO: ...
    header : Dict
        TODO: ...
    """

    def __init__(self, compile_options, run_options, header):

        self.compile_options = compile_options
        self.run_options = run_options
        self.header = header
        self.infile_dictionary: Dict[str, tuple] = None
        self._make_infile_dictionary()

    def _make_header_and_datetime(self) -> Tuple[str]:

        now = datetime.datetime.strftime(
            datetime.datetime.today(), '%d/%m/%Y %H:%M:%S.%f'
        )[:-5]

        header = (
            f'Runtime options file for Phantom, written {now}',
            'Options not present assume their default values',
            'This file is updated automatically after a full dump',
        )

        datetime_ = now

        return header, datetime_

    def _blocks_to_add(self):
        """
        Determine which run time blocks to add.

        This method and _get_required_values_from_block contain the
        logic of writing a Phantom in file which is contained in
        readwrite_infile.F90 in the Phantom source.

        See Also
        --------
        _get_required_values_from_block
        """

        blocks_to_add = list()
        blocks_to_add.append(block_label['job'])
        blocks_to_add.append(block_label['io'])

        if (
            self.run_options.config['rhofinal_cgs'].value > 0.0
            or self.run_options.config['dtmax_dratio'].value > 1.0
            or self.run_options.config['calc_erot'].value
        ):
            blocks_to_add.append(block_label[('io supplementary')])

        blocks_to_add.append(block_label['accuracy'])
        blocks_to_add.append(block_label['hydrodynamics'])
        blocks_to_add.append(block_label['damping'])
        blocks_to_add.append(block_label['eos'])

        if not self.compile_options['ISOTHERMAL']:
            blocks_to_add.append(block_label['cooling'])

        if self.compile_options['MCFOST']:
            blocks_to_add.append(block_label['MCFOST'])

        if self.header['nptmass'] > 0 or self.compile_options['GRAVITY']:
            blocks_to_add.append(block_label['sinks'])

        blocks_to_add.append(block_label['external forces'])
        blocks_to_add.append(block_label['viscosity'])

        if self.compile_options['DRIVING']:
            blocks_to_add.append(block_label['driving'])

        if self.compile_options['DUST']:
            blocks_to_add.append(block_label['dust'])

        if self.compile_options['DUSTGROWTH']:
            blocks_to_add.append(block_label['dustgrowth'])

        if self.compile_options['PHOTO']:
            blocks_to_add.append(block_label['photoevaporation'])

        if self.compile_options['INJECT_PARTICLES']:
            blocks_to_add.append(block_label['inject'])

        if self.compile_options['NONIDEALMHD']:
            blocks_to_add.append(block_label['non-ideal MHD'])

        return blocks_to_add

    def _get_required_values_from_block(self, block: str) -> Dict:
        """
        Determine which parameters within a block to add.

        This method and _blocks_to_add contain the logic of writing a
        Phantom in file which is contained in readwrite_infile.F90 in
        the Phantom source.

        See Also
        --------
        _blocks_to_add
        """

        # TODO: add more checks for
        #  - external_forces
        #  - driving by turbulence
        #  - non-ideal MHD
        #  - particle injection

        block_dict = self.run_options.to_dict()[block]

        if block == block_label['accuracy']:
            if not self.compile_options['GRAVITY']:
                block_dict.pop('tree_accuracy')

        if block == block_label['hydrodynamics']:
            if self.compile_options['ISOTHERMAL']:
                block_dict.pop('alphau')
            if not (self.compile_options['MHD'] or self.compile_options['NONIDEALMHD']):
                block_dict.pop('alphaB')
                block_dict.pop('psidecayfac')
                block_dict.pop('overcleanfac')

        if block == block_label['damping']:
            if self.run_options.config['idamp'].value != 1:
                block_dict.pop('damp')
            if self.run_options.config['idamp'].value != 2:
                block_dict.pop('tdyn_s')

        if block == block_label['eos']:
            if self.compile_options['ISOTHERMAL']:
                block_dict.pop('ipdv_heating')
                block_dict.pop('ishock_heating')
                block_dict.pop('iresistive_heating')
                block_dict.pop('drhocrit')
                block_dict.pop('rhocrit0')
                block_dict.pop('rhocrit1')
                block_dict.pop('rhocrit2')
                block_dict.pop('rhocrit3')
                block_dict.pop('gamma1')
                block_dict.pop('gamma2')
                block_dict.pop('gamma3')
                block_dict.pop('rhocrit0pwp')
                block_dict.pop('rhocrit1pwp')
                block_dict.pop('rhocrit2pwp')
                block_dict.pop('gamma0pwp')
                block_dict.pop('gamma1pwp')
                block_dict.pop('gamma2pwp')
                block_dict.pop('gamma3pwp')
                block_dict.pop('p1pwp')
                block_dict.pop('X')
                block_dict.pop('Z')
                block_dict.pop('relaxflag')

        if block == block_label['sinks']:
            if self.compile_options['GRAVITY']:
                if not (self.run_options.config['icreate_sinks'].value > 0):
                    block_dict.pop('rho_crit_cgs')
                    block_dict.pop('r_crit')
                    block_dict.pop('h_acc')
                    block_dict.pop('h_soft_sinkgas')
            else:
                block_dict.pop('icreate_sinks')
                block_dict.pop('rho_crit_cgs')
                block_dict.pop('r_crit')
                block_dict.pop('h_acc')
                block_dict.pop('h_soft_sinkgas')

        if block == block_label['dust']:
            if self.run_options.config['idrag'].value != 1:
                block_dict.pop('grainsize')
                block_dict.pop('graindens')
            if self.run_options.config['idrag'].value == 1:
                block_dict.pop('K_code')
            if self.header['ndustsmall'] == 0:
                block_dict.pop('ilimitdustflux')

        if block == block_label['dustgrowth']:
            if self.run_options.config['ifrag'].value == 0:
                block_dict.pop('grainsizemin')
                block_dict.pop('isnow')
            if self.run_options.config['isnow'].value == 0:
                block_dict.pop('vfragin')
                block_dict.pop('vfragout')
            if self.run_options.config['isnow'].value != 0:
                block_dict.pop('vfrag')
            if self.run_options.config['isnow'].value != 1:
                block_dict.pop('rsnow')
            if self.run_options.config['isnow'].value != 2:
                block_dict.pop('Tsnow')

        return block_dict

    def _make_infile_dictionary(self):

        infile_dictionary = dict()

        header, datetime_ = self._make_header_and_datetime()
        infile_dictionary['__header__'] = header
        infile_dictionary['__datetime__'] = datetime_

        blocks_to_add = self._blocks_to_add()
        for block in blocks_to_add:
            block_dict = self._get_required_values_from_block(block)
            infile_dictionary[block] = block_dict
        self.infile_dictionary = infile_dictionary


def generate_infile(compile_options, run_options, header):
    """
    TODO: write docstring.
    """
    infile = _InFile(compile_options, run_options, header)
    return infile.infile_dictionary
