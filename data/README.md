# data/

Nothing in this directory except this file and `.gitkeep` is tracked. DENTEX, the
Tufts Dental Database, ACTA-Bw25-RefStd and HUNT4 are released under data use
agreements that forbid redistribution: never commit images, annotations, or
derived artefacts (crops, embeddings, cached record manifests).

Expected layout once access is granted (loaders will read from here):

```
data/
├── dentex/
├── tufts/
├── acta_bw25/      # external test ONLY: never train, tune or threshold on it
└── hunt4/
```

Check before any commit:

```bash
git status --ignored data/
```
