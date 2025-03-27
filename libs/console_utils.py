#!/usr/bin/env python
#
#
# v0.1 -- Initial Implementation -- <mcaimi@redhat.com>
#
""" Various utilities to ease dealing with console output
 from colored text to formatting to everything else """

from typing import Union


class ANSIColors(object):
    """ Class managing ansi colors in console output """

    def __init__(self) -> None:
        self.escapecode: str = '\033[%s;%sm'

        self.colors: dict = {'RED': 1,
                             'GREEN': 2,
                             'YELLOW': 3,
                             'BLUE': 4,
                             'PURPLE': 5,
                             'CYAN': 6,
                             'WHITE': 7,
                             'BLACK': 0}

        self.modifiers: dict = {'BRIGHT': 1, 'NORMAL': 0}

        # build color codes hashes
        self.compile_ansicolors_hash()

    def compile_ansicolors_hash(self) -> None:
        ''' Compile ansi colors hash table '''
        colorlist: dict = {}
        for intensity, intensity_value in self.modifiers.items():
            colorlist[intensity]: dict = {}
            for color, color_value in self.colors.items():
                colorlist[intensity][color] = self.escapecode % (intensity_value, 30 + color_value)
        colorlist['RESET'] = '\033[0m'
        # update ansi colorlist hash
        self.ansi_escapes: dict = colorlist

    def get_ansicolors_hash(self) -> dict:
        ''' retrieve the hash table containing ANSI colors escape codes '''
        return self.ansi_escapes

    def get_code_for_color(self, mod, color) -> Union[None, str]:
        """ get the escape sequence with parameters

            mod: escape modifier
            color: color key """

        if mod in self.ansi_escapes:
            if color in self.ansi_escapes[mod]:
                return self.ansi_escapes[mod][color]
        return None

    def color_write(self, text, mod="NORMAL", color="WHITE") -> str:
        ''' generate an escapecode-colored string '''
        return f"{self.get_code_for_color(mod, color)}{text}{self.ansi_escapes['RESET']}"

    def color_print(self, text, mod="NORMAL", color="WHITE") -> None:
        ''' print colored text '''
        print(self.color_write(text, mod, color))

    def error(self, text) -> str:
        ''' generate an error string (red text) '''
        return self.color_write(text, 'BRIGHT', 'RED')

    def print_error(self, text) -> None:
        ''' print an error message (red text) '''
        print(self.error(text))

    def warning(self, text) -> str:
        ''' generate a warning string (bright yellow text) '''
        return self.color_write(text, 'BRIGHT', 'YELLOW')

    def print_warning(self, text) -> None:
        ''' print a warning message '''
        print(self.warning(text))

    def success(self, text) -> str:
        ''' generate a success string (green color) '''
        return self.color_write(text, 'BRIGHT', 'GREEN')

    def print_success(self, text) -> None:
        ''' print a success message '''
        print(self.success(text))
