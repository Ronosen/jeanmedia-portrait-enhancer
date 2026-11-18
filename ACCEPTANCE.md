# Acceptance gates

A checked implementation item is not a photographic quality certificate. Inspect original-resolution results before approving a client export.

| Gate | Pass condition | Current evidence |
|---|---|---|
| Source safety | Original bytes unchanged; existing outputs refused; cancellation cannot publish an incomplete final export | Worker export regression tests, atomic new-file publication |
| Protected geometry | Before tonal finishing, protected source pixels are unchanged except explicitly planned furniture-surface cleaning | Synthetic unit tests and per-photo `protectedPixelChanges` in QA-REPORT.json |
| Repair containment | Zero changed pixels outside the saved combined change map before permitted global finishing | `outsideMaskChanges` in QA-REPORT.json and floor/prop regression tests |
| Repair coherence | At 100%, no seams, copied marks, repeated floor texture, halos or visibly different material; wall/floor perspective and contact shadows remain plausible | Human acceptance pending; automatic containment does not prove this |
| Prop retention | Shape, location, edges, hand/seat contacts and shadows retained; only approved surface defects changed | Limited fixture tests; broad real-photo acceptance pending |
| Identity | No changes to facial/body geometry, expression, clothing design or accessories; no invented anatomy | Algorithm excludes subject from repairs; independent full-image human acceptance pending |
| Review | Uncertain results visibly flagged; save blocked until explicitly approved; skipped photos never counted as successful cleanups | Backend review gate; frontend approval, errors and queue states |
| Print | Requested dimensions, sRGB profile, intended PPI, no unintended crop or resampling | Original-resolution export fixtures and real worker exports; custom sizing/soft proof pending |
| Release | 30 varied consented originals, CPU and GPU-machine evaluation, human scores, license audit, signed package, clean-computer install | Not yet passed |

The local dataset has 16 original photographs, not 30. QA-REPORT.json records automated checks on those inputs. A skipped image is a failure to complete cleanup even though preserving the original is the correct safety outcome. No fabricated human approval scores or object-detection ground truth are recorded.

Known unresolved real cases: IMG_0731 and IMG_1030 have supporting seats that the small models cannot safely protect; IMG_1232 has no sufficiently consistent backdrop fit. Large/textured floor edges can remain unchanged and require review. These prevent marking the whole checklist complete.

The model decision document separates desk evaluation from installed runtime benchmarks. A portable runtime self-test on this machine does not count as an installation test on a second computer. No signing certificate has been supplied.
