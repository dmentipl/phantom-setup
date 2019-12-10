import datetime
from typing import Any, Dict, Tuple

from .defaults import RUN_OPTION_BLOCK_LABEL as block_label


class _InFile:
    """Phantom in file.

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

    def get_compile_option(self, option: str) -> Any:
        """Get the value of a Phantom compile time option.

        Parameters
        ----------
        option
            The compile time option to get.

        Returns
        -------
        The value of the option.
        """
        if option in self.compile_options:
            return self.compile_options[option]
        else:
            raise ValueError(f'Compile time option={option} does not exist')

    def get_run_option(self, option: str) -> Any:
        """Get the value of a Phantom run time option.

        Parameters
        ----------
        option
            The run time option to get.

        Returns
        -------
        The value of the option.
        """
        if option in self.run_options.config:
            return self.run_options.config[option].value
        else:
            raise ValueError(f'Run time option={option} does not exist')

    def _make_header_and_datetime(self) -> Tuple[Tuple[str, ...], str]:

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
        """Determine which run time blocks to add.

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
            self.get_run_option('rhofinal_cgs') > 0.0
            or self.get_run_option('dtmax_dratio') > 1.0
            or self.get_run_option('calc_erot')
        ):
            blocks_to_add.append(block_label[('io supplementary')])

        blocks_to_add.append(block_label['accuracy'])
        blocks_to_add.append(block_label['hydrodynamics'])
        blocks_to_add.append(block_label['damping'])
        blocks_to_add.append(block_label['eos'])

        if not self.get_compile_option('ISOTHERMAL'):
            blocks_to_add.append(block_label['cooling'])

        if self.get_compile_option('MCFOST'):
            blocks_to_add.append(block_label['MCFOST'])

        if self.header['nptmass'] > 0 or self.get_compile_option('GRAVITY'):
            blocks_to_add.append(block_label['sinks'])

        blocks_to_add.append(block_label['external forces'])
        blocks_to_add.append(block_label['viscosity'])

        if self.get_compile_option('DRIVING'):
            blocks_to_add.append(block_label['driving'])

        if self.header['ndustsmall'] > 0 or self.header['ndustlarge'] > 0:
            blocks_to_add.append(block_label['dust'])

        if self.get_compile_option('DUSTGROWTH'):
            blocks_to_add.append(block_label['dustgrowth'])

        if self.get_compile_option('PHOTO'):
            blocks_to_add.append(block_label['photoevaporation'])

        if self.get_compile_option('INJECT_PARTICLES'):
            blocks_to_add.append(block_label['inject'])

        if self.get_compile_option('NONIDEALMHD'):
            blocks_to_add.append(block_label['non-ideal MHD'])

        return blocks_to_add

    def _get_required_values_from_block(self, block: str) -> Dict:
        """Determine which parameters within a block to add.

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
            if not self.get_compile_option('GRAVITY'):
                block_dict.pop('tree_accuracy')

        if block == block_label['hydrodynamics']:
            if self.get_compile_option('ISOTHERMAL'):
                block_dict.pop('alphau')
            if not (
                self.get_compile_option('MHD') or self.get_compile_option('NONIDEALMHD')
            ):
                block_dict.pop('alphaB')
                block_dict.pop('psidecayfac')
                block_dict.pop('overcleanfac')
            if self.get_compile_option('DISC_VISCOSITY'):
                block_dict.pop('alphamax')

        if block == block_label['damping']:
            if self.get_run_option('idamp') != 1:
                block_dict.pop('damp')
            if self.get_run_option('idamp') != 2:
                block_dict.pop('tdyn_s')

        if block == block_label['eos']:
            if self.get_compile_option('ISOTHERMAL'):
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
            if self.get_compile_option('GRAVITY'):
                if not (self.get_run_option('icreate_sinks') > 0):
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
            if self.get_run_option('idrag') != 1:
                block_dict.pop('grainsize')
                block_dict.pop('graindens')
            if self.get_run_option('idrag') == 1:
                block_dict.pop('K_code')
                if self.header['ndustsmall'] > 1 or self.header['ndustlarge'] > 1:
                    block_dict.pop('grainsize')
                    block_dict.pop('graindens')
            if self.header['ndustsmall'] == 0:
                block_dict.pop('ilimitdustflux')

        if block == block_label['dustgrowth']:
            if self.get_run_option('ifrag') == 0:
                block_dict.pop('grainsizemin')
                block_dict.pop('isnow')
            if self.get_run_option('isnow') == 0:
                block_dict.pop('vfragin')
                block_dict.pop('vfragout')
            if self.get_run_option('isnow') != 0:
                block_dict.pop('vfrag')
            if self.get_run_option('isnow') != 1:
                block_dict.pop('rsnow')
            if self.get_run_option('isnow') != 2:
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
    """TODO: write docstring."""
    infile = _InFile(compile_options, run_options, header)
    return infile.infile_dictionary
