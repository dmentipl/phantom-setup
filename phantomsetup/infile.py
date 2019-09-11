import datetime
from typing import Dict, Tuple


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

        blocks_to_add = list()
        blocks_to_add.append('job name')
        blocks_to_add.append('options controlling run time and input/output')

        if (
            self.run_options.config['rhofinal_cgs'].value > 0.0
            or self.run_options.config['dtmax_dratio'].value > 1.0
            or self.run_options.config['calc_erot'].value
        ):
            blocks_to_add.append(
                (
                    'options controlling run time and input/output: '
                    'supplementary features'
                )
            )

        blocks_to_add.append('options controlling accuracy')
        blocks_to_add.append(
            'options controlling hydrodynamics, artificial dissipation'
        )
        blocks_to_add.append('options controlling damping')
        blocks_to_add.append('options controlling equation of state')

        if self.compile_options['ISOTHERMAL']:
            blocks_to_add.append('options controlling cooling')

        if self.compile_options['MCFOST']:
            blocks_to_add.append('options controlling MCFOST')

        if self.header['nptmass'] > 0 or self.compile_options['GRAVITY']:
            blocks_to_add.append('options controlling sink particles')

        blocks_to_add.append('options relating to external forces')
        blocks_to_add.append('options controlling physical viscosity')

        if self.compile_options['DRIVING']:
            blocks_to_add.append('options controlling forcing of turbulence')

        if self.compile_options['DUST']:
            blocks_to_add.append('options controlling dust')

        if self.compile_options['DUSTGROWTH']:
            blocks_to_add.append('options controlling dust growth')

        if self.compile_options['PHOTO']:
            blocks_to_add.append('options controlling photoevaporation')

        if self.compile_options['INJECT_PARTICLES']:
            blocks_to_add.append('options controlling injecting particles')

        if self.compile_options['NONIDEALMHD']:
            blocks_to_add.append('options controlling non-ideal MHD')

        return blocks_to_add

    def _get_required_values_from_block(self, block: str) -> Dict:

        block_dict = self.run_options.to_dict()[block]

        if block == 'options controlling accuracy':
            if not self.compile_options['GRAVITY']:
                block_dict.pop('tree_accuracy')

        if block == 'options controlling hydrodynamics, artificial dissipation':
            if self.compile_options['ISOTHERMAL']:
                block_dict.pop('alphau')

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
