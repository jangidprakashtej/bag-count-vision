# Bag Counter Project — Summary

**For:** [Your Name / Company Name]
**Prepared:** August 2026

---

## What we're building

An automated, camera-based system that watches your bag-packing nozzle,
**recognizes which product category was just packed** (Gold ISI, Gold
NISI, Gold Plus, Sak 25kg ISI, Sak 25kg NISI, Suntara Gypsum, etc.),
**counts each bag automatically**, and **logs it to Google Sheets** with
a timestamp — no manual tally-keeping required.

From that raw count, the system automatically calculates:
- **Tonnage per category** (bag count × weight per bag, since not all
  bags are 25kg — some are 40kg)
- **A daily production report**, resetting cleanly every 24 hours
- **Traceability to QC sample testing** — every quality-control test
  taken gets linked back to exactly which bag count/category it was
  sampled from

It's being packaged as an actual **installable desktop app** (Windows
and Mac) with a simple point-and-click interface, so factory staff can
run it without touching code — designed to be deployed the same way
across **multiple factory sites**, each with one camera at one nozzle.

## How this helps your company

- **Eliminates manual counting** — staff previously hand-tallied bags
  per category to compute daily tonnage; this removes that entire task
  and its error risk.
- **Accurate, auditable records** — every bag is timestamped and logged
  automatically, creating a permanent, tamper-resistant production log
  instead of paper tallies.
- **Power-outage resilient** — if the system restarts mid-day, it
  automatically resumes counting from where it left off by reading
  today's existing log rows, rather than losing the day's count.
- **QC traceability** — when a quality dispute or audit comes up, you
  can trace exactly which bag count/category a given QC sample
  corresponds to, not just "sometime that day."
- **Scales across factories** — one shared reporting structure (with a
  Factory column) means you can compare production across sites from
  one place, without separate manual systems per location.
- **Foundation for further automation** — once reliable, this same
  camera+logging pipeline can extend to other stations, other QC checks,
  or feed into larger inventory/ERP systems later.

## Phase-wise development journey

### ✅ Phase 1 — Prototype & core pipeline
Built the first working version: camera capture → motion detection →
image recognition (CLIP, matched against your own reference photos) →
counting → Google Sheets logging with timestamps. Proved the core
concept end-to-end using a phone camera.

### ✅ Phase 2 — Fitting the real production process
Learned the actual constraints (bags take 20-35 seconds each, thrown
onto the floor, one category runs for a long stretch at a time) and
redesigned the counting logic around it — moving from an initial
"fast line crossing" assumption to "stable track confirmation," which
matches how bags actually arrive. Added per-category bag weights (not
all bags are 25kg) and a daily tonnage report that resets every 24 hours.

### ✅ Phase 3 — Reliability hardening
Fixed real-world issues that showed up in live testing:
- False counts from a handheld camera (background subtraction mistook
  camera shake for motion) — solved by requiring a stationary mount and
  tightening motion sensitivity.
- Auto-exposure/white-balance flicker on phone cameras causing false
  triggers.
- Camera stream drop-outs (network hiccups) with no automatic recovery
  — added reconnect logic.
- Power-outage data loss risk — added resume-from-Sheet logic so a
  restart never loses the day's count.

### ✅ Phase 4 — Traceability & reporting features
- Daily report script: aggregates a day's bags into tonnage per category.
- QC sample pegging: manual QC test entries now link to the exact
  category/bag-count active at the time of sampling.

### ✅ Phase 5 — Recognition accuracy upgrade
As more real bag photos accumulated (200+ per category, busier real-world
scenes with multiple bags), moved from CLIP similarity-matching to a
**custom-trained YOLOv8 object detector** — better suited to cluttered
scenes and more accurate on visually similar categories (e.g. Gold ISI
vs Gold NISI). Dataset annotated via Roboflow; model trained and
currently being evaluated/refined for these close look-alike categories.

### ✅ Phase 6 — Desktop app packaging
Rebuilt the pipeline into an actual installable GUI app (Start/Stop
button, live camera view, live running counts, a Settings screen) so
non-technical factory staff can operate it — settings now live in a
simple config file instead of requiring code edits, ready to be packaged
as a standalone installer for Windows and Mac.

### ✅ Phase 7 — IP protection groundwork
Put safeguards in place before bringing in outside developers: private
GitHub repository (with full commit history as a dated ownership trail),
a proprietary/all-rights-reserved LICENSE file, and guidance on
NDA + IP-assignment agreements for any external developer work.

### 🔄 Phase 8 — Current: refinement & parallel evaluation
- Tuning detection speed/accuracy on the trained YOLOv8 model (frame
  resizing, inference frequency, camera hardware upgrade guidance for
  moving off a phone onto a proper mounted IP camera).
- Resolving the Gold ISI / Gold NISI look-alike confusion specifically.
- Separately evaluating an alternative build (a TensorFlow-based version
  with per-factory local training) as a comparison point — being tested
  independently without affecting the main system.

### ⏭️ Phase 9 — Ahead
- Finalize accuracy on look-alike categories (more targeted training
  images, possibly an OCR tie-breaker for text-based distinctions).
- Move from phone camera to a properly mounted, dust-rated (IP66/67) IP
  camera at the actual nozzle location.
- Package the app as a distributable installer (PyInstaller/Inno Setup)
  for deployment to additional factory sites.
- Decide remaining open design questions: shared vs. per-factory Google
  Sheets, staff-editable vs. locked settings, model version tracking per
  log entry.
- Ongoing: collect real-world misclassifications from daily use and
  periodically retrain the model — turning live deployment into a
  continuous accuracy-improvement loop rather than a one-time build.

---

*For full technical detail (architecture decisions, known issues already
fixed, file-by-file breakdown), see `PROJECT_NOTES.md` in the project
repository — this document is the high-level version for sharing outside
the technical work itself.*
