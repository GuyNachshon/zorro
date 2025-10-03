.. vim: set fileencoding=utf-8:
.. -*- coding: utf-8 -*-
.. +--------------------------------------------------------------------------+
   |                                                                          |
   | Licensed under the Apache License, Version 2.0 (the "License");          |
   | you may not use this file except in compliance with the License.         |
   | You may obtain a copy of the License at                                  |
   |                                                                          |
   |     http://www.apache.org/licenses/LICENSE-2.0                           |
   |                                                                          |
   | Unless required by applicable law or agreed to in writing, software      |
   | distributed under the License is distributed on an "AS IS" BASIS,        |
   | WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. |
   | See the License for the specific language governing permissions and      |
   | limitations under the License.                                           |
   |                                                                          |
   +--------------------------------------------------------------------------+

*******************************************************************************
                                   libravore                                   
*******************************************************************************

.. image:: https://img.shields.io/pypi/v/libravore
   :alt: Package Version
   :target: https://pypi.org/project/libravore/

.. image:: https://img.shields.io/pypi/status/libravore
   :alt: PyPI - Status
   :target: https://pypi.org/project/libravore/

.. image:: https://github.com/emcd/python-libravore/actions/workflows/tester.yaml/badge.svg?branch=master&event=push
   :alt: Tests Status
   :target: https://github.com/emcd/python-libravore/actions/workflows/tester.yaml

.. image:: https://emcd.github.io/python-libravore/coverage.svg
   :alt: Code Coverage Percentage
   :target: https://github.com/emcd/python-libravore/actions/workflows/tester.yaml

.. image:: https://img.shields.io/github/license/emcd/python-libravore
   :alt: Project License
   :target: https://github.com/emcd/python-libravore/blob/master/LICENSE.txt

.. image:: https://img.shields.io/pypi/pyversions/libravore
   :alt: Python Versions
   :target: https://pypi.org/project/libravore/


🚩 **Misspelling Redirect Package**

This package exists because "libravore" is a common misspelling of 
`librovore <https://pypi.org/project/librovore/>`_ (note the "o"). While this 
package name is reserved for a future "devourer of [model] weights", it is not 
the "devourer of documentation" that you are looking for!

**You probably want** `librovore <https://pypi.org/project/librovore/>`_ **instead.**

What This Package Does ⭐
===============================================================================

* Issues a warning when imported to alert about the misspelling
* Automatically redirects all imports to the correct ``librovore`` package
* Prevents confusion from the common "library" → "libravore" misspelling


Installation 📦
===============================================================================

Method: Install Python Package
-------------------------------------------------------------------------------

Install via `uv <https://github.com/astral-sh/uv/blob/main/README.md>`_ ``pip``
command:

::

    uv pip install libravore

Or, install via ``pip``:

::

    pip install libravore


Examples 💡
===============================================================================

**If you accidentally imported this package:**

::

    import libravore  # Shows warning, redirects to librovore

**Better - use the correct package directly:**

::

    pip install librovore  # Note the "o"
    import librovore       # The actual package you want


Contribution 🤝
===============================================================================

Contribution to this project is welcome! However, it must follow the `code of
conduct
<https://emcd.github.io/python-project-common/stable/sphinx-html/common/conduct.html>`_
for the project.

Please file bug reports and feature requests in the `issue tracker
<https://github.com/emcd/python-libravore/issues>`_ or submit `pull
requests <https://github.com/emcd/python-libravore/pulls>`_ to
improve the source code or documentation.

For development guidance and standards, please see the `development guide
<https://emcd.github.io/python-libravore/stable/sphinx-html/contribution.html#development>`_.


`More Flair <https://www.imdb.com/title/tt0151804/characters/nm0431918>`_
===============================================================================

.. image:: https://img.shields.io/github/last-commit/emcd/python-libravore
   :alt: GitHub last commit
   :target: https://github.com/emcd/python-libravore

.. image:: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/copier-org/copier/master/img/badge/badge-grayscale-inverted-border-orange.json
   :alt: Copier
   :target: https://github.com/copier-org/copier

.. image:: https://img.shields.io/badge/%F0%9F%A5%9A-Hatch-4051b5.svg
   :alt: Hatch
   :target: https://github.com/pypa/hatch

.. image:: https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit
   :alt: pre-commit
   :target: https://github.com/pre-commit/pre-commit

.. image:: https://microsoft.github.io/pyright/img/pyright_badge.svg
   :alt: Pyright
   :target: https://microsoft.github.io/pyright

.. image:: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json
   :alt: Ruff
   :target: https://github.com/astral-sh/ruff

.. image:: https://img.shields.io/pypi/implementation/libravore
   :alt: PyPI - Implementation
   :target: https://pypi.org/project/libravore/

.. image:: https://img.shields.io/pypi/wheel/libravore
   :alt: PyPI - Wheel
   :target: https://pypi.org/project/libravore/


Other Projects by This Author 🌟
===============================================================================


* `python-absence <https://github.com/emcd/python-absence>`_ (`absence <https://pypi.org/project/absence/>`_ on PyPI) 

  🕳️ A Python library package which provides a **sentinel for absent values** - a falsey, immutable singleton that represents the absence of a value in contexts where ``None`` or ``False`` may be valid values.
* `python-accretive <https://github.com/emcd/python-accretive>`_ (`accretive <https://pypi.org/project/accretive/>`_ on PyPI) 

  🌌 A Python library package which provides **accretive data structures** - collections which can grow but never shrink.
* `python-classcore <https://github.com/emcd/python-classcore>`_ (`classcore <https://pypi.org/project/classcore/>`_ on PyPI) 

  🏭 A Python library package which provides **foundational class factories and decorators** for providing classes with attributes immutability and concealment and other custom behaviors.
* `python-dynadoc <https://github.com/emcd/python-dynadoc>`_ (`dynadoc <https://pypi.org/project/dynadoc/>`_ on PyPI) 

  📝 A Python library package which bridges the gap between **rich annotations** and **automatic documentation generation** with configurable renderers and support for reusable fragments.
* `python-falsifier <https://github.com/emcd/python-falsifier>`_ (`falsifier <https://pypi.org/project/falsifier/>`_ on PyPI) 

  🎭 A very simple Python library package which provides a **base class for falsey objects** - objects that evaluate to ``False`` in boolean contexts.
* `python-frigid <https://github.com/emcd/python-frigid>`_ (`frigid <https://pypi.org/project/frigid/>`_ on PyPI) 

  🔒 A Python library package which provides **immutable data structures** - collections which cannot be modified after creation.
* `python-icecream-truck <https://github.com/emcd/python-icecream-truck>`_ (`icecream-truck <https://pypi.org/project/icecream-truck/>`_ on PyPI) 

  🍦 **Flavorful Debugging** - A Python library which enhances the powerful and well-known ``icecream`` package with flavored traces, configuration hierarchies, customized outputs, ready-made recipes, and more.
* `python-mimeogram <https://github.com/emcd/python-mimeogram>`_ (`mimeogram <https://pypi.org/project/mimeogram/>`_ on PyPI) 

  📨 A command-line tool for **exchanging collections of files with Large Language Models** - bundle multiple files into a single clipboard-ready document while preserving directory structure and metadata... good for code reviews, project sharing, and LLM interactions.
