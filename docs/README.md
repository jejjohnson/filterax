# Documentation

The site is built by two generators.

- **mystmd** builds the prose half: the home page, the maths reference in `math/` and the design docs. It reads `myst.yml` and writes `_build/html`.
- **MkDocs + mkdocstrings** builds the API reference from the docstrings, with its pages in `api/`. It reads `../mkdocs.yml` and writes `../site`, along with the `objects.inv` the prose links against.

`scripts/build_docs.py` runs both, mounts the API reference under `/reference/`, repairs the cross-references between them, and checks that every internal link in the assembled site resolves.

## Commands

mystmd is a Node package, so it needs a separate install.

```bash
npm install -g mystmd
```

Build the whole site into `public/`:

```bash
make docs
```

Validate the sources without rendering the theme, which needs network access to fetch its template:

```bash
make docs-check
```

Serve the built site at <http://127.0.0.1:8000>:

```bash
make docs-serve
```

## Linking between the two halves

- A chapter links to an API symbol through the inventory: `` [`ETKF`](xref:api#filterax.ETKF) ``.
- A chapter links to a whole API page, and an API page links to a chapter, with an absolute URL to the deployed site.
- A local preview's "API reference" button points at the deployed site, because the theme needs a full URL there.
