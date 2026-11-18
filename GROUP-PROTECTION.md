# Group portrait integration — 17 November 2026

Superseded by the internal extract-clean-composite workflow documented in
`SEPARATED-WORKFLOW.md`. This file records the earlier group-fusion approach.

Auto Edit now runs person and foreground models on each detected person when
there are two or more people. It fuses those masks with the full-frame result.
A separate head crop recovers graduation caps and thin coloured tassels. The
same protection is used for reconstruction, backdrop finishing and saved export.
Single-person images keep their existing path. No model weights were trained
or downloaded for this change; all processing stays local.

Regression cases: `D:/test22/IMG_0164.JPG` (four people) and
`D:/test22/IMG_0203.JPG` (two people). Before integration both were rejected
because the global foreground model missed part of the group. Outputs, plans,
source hashes and comparisons are in `qa/group-release-20261117`.

This resolves the group detection rejection, not every background defect. Both
examples must retain their **Needs review** status. Visual issues still observed:

- IMG_0164: pale borders around caps and clothing; some side wallpaper/panel
  remnants; blue cloth at bottom left; uneven reconstructed backdrop colour;
  floor scratches and the frayed front edge are not fully repaired.
- IMG_0203: pale borders around caps/hair and sleeves; remnants of side panels;
  uneven patches in the reconstructed backdrop; residual blue cloth/colour at
  bottom left; floor marks remain.

The certificate, books and pedestal are intentional props and are retained.
The original source photos are never overwritten. Reimport an original and
run Auto Edit to generate the new masks; existing edited previews are not
automatically reprocessed.
