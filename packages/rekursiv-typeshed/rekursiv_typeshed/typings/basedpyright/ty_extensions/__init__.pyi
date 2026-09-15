# Polyfill for ty's built-in ``ty_extensions.Intersection``. basedpyright has no
# intersection type, so ``Intersection[A, B]`` approximates as ``A``. This file
# replaces ``stdlib/ty_extensions/__init__.pyi`` in basedpyright's tree only; ty's
# tree keeps the real ``_SpecialForm``.
type Intersection[_First, _Second] = _First
