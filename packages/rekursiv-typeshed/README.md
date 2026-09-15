# rekursiv-typeshed

A copy of the standard-library stubs the `ty` and `basedpyright` type checkers
bundle, with one change: `int.__pow__`, `int.__rpow__`, `float.__pow__` and
`float.__rpow__` return `float` instead of typeshed's `Any`. Under
`reportAny`, typeshed's spelling turns every `x ** 0.5` into a diagnostic.

Neither checker accepts that as a partial override -- basedpyright only flags
a `builtins.pyi` as builtin when it sits inside a full typeshed tree, and ty
binds `builtins` to its bundled stdlib -- so the patch is applied to a whole
copy and both checkers are pointed at the copy. The wheel is built from the
exact checker versions it pins, so the trees and the checkers never drift.

Installing the wheel places the trees at `<prefix>/share/rekursiv-typeshed`
(`.venv/share/rekursiv-typeshed` under uv), one per checker:

```toml
[tool.basedpyright]
typeshedPath = ".venv/share/rekursiv-typeshed/basedpyright"

[tool.ty.environment]
typeshed = ".venv/share/rekursiv-typeshed/ty"
```

Both trees start from ty's `stdlib/` (the newer of the two bundles; it
already carries `ty_extensions/`), patched, then overlaid with the whole-file
replacements in `rekursiv_typeshed/typings/stdlib/` (`unittest/mock.pyi`,
typed so `Mock` attributes are not `Any`). They differ in:

- `basedpyright/` adds `stubs/` (basedpyright's third-party stubs, which ty
  does not bundle) and replaces `stdlib/ty_extensions/` with
  `rekursiv_typeshed/typings/basedpyright/`: `Intersection[A, B] = A`, since
  basedpyright has no intersection type.
- `ty/` keeps ty's real `ty_extensions.Intersection` (`_SpecialForm`).

Your own `stubPath` / `extra-paths` stay free for hand-written stubs.
