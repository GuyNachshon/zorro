# vim: set filetype=python fileencoding=utf-8:
# -*- coding: utf-8 -*-

#============================================================================#
#                                                                            #
#  Licensed under the Apache License, Version 2.0 (the "License");           #
#  you may not use this file except in compliance with the License.          #
#  You may obtain a copy of the License at                                   #
#                                                                            #
#      http://www.apache.org/licenses/LICENSE-2.0                            #
#                                                                            #
#  Unless required by applicable law or agreed to in writing, software       #
#  distributed under the License is distributed on an "AS IS" BASIS,         #
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  #
#  See the License for the specific language governing permissions and       #
#  limitations under the License.                                            #
#                                                                            #
#============================================================================#


''' Misspelling of "librovore". See the Librovore package instead. '''


import warnings as _warnings

from librovore import *  # pyright: ignore

from . import __
# --- BEGIN: Injected by Copier ---
# --- END: Injected by Copier ---


__version__ = '1.0'


_warnings.warn(
    "You have imported 'libravore' which is a misspelling of "
    "'librovore'. Consider updating your code to use 'librovore' (with an "
    "'o') instead.",
    UserWarning,
    stacklevel=2
)


# TODO: Reclassify package modules as immutable and concealed.
