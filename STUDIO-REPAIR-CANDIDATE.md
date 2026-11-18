# Studio repair verification — 2026-11-12

The user-approved LaMa model is installed and SHA256 verified. It replaces the colour-fit renderer for new automatic studio edits. Two real 4000 × 6000 exports passed cleanup containment and protected-region checks; see QA-INPAINTING.json. The original source files remain unchanged. The seat, hands, hair, coat, boots and small seat support were also checked explicitly.

The single Auto Edit action selects masks automatically, reconstructs background/floor holes locally, and heals small smooth-surface defects. The full equipment mask is retained across the detected floor line. Exposed tiles and tape are reconstructed as a localized mat extension. Mask inspection uses outlines instead of coloured blocks. Large fills require review before saving.

The backend and native build are validated. Final UI/bundle verification is recorded in VERIFICATION.json. General photographic acceptance, ambiguous supporting seats, commercial weight provenance, manual tools and signed distribution remain open. No client photos were uploaded.
