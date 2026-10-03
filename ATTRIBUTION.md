# Attribution and reuse review

Original learning experiment, derivation, Deming implementation, tests and synthetic outputs: © 2026 ahmetalpkilic, MIT (see LICENSE).

The Gaussian-pupil pixel model and position fitter in `optics.py` are adapted from ahmetalpkilic/reference-clock-camera-timing-learning-lab at cccab81d3d15a8bb05352138cfdff6e88e9bf8cb, MIT. The predecessor notice and complete license are preserved in licenses/. This project changes the calibration model to noisy master/reference axes and uses a new errors-in-variables experiment.

Reviewed current unarchived public projects on the date recorded in sources.json:

- brandondube/prysm: physical-optics context, MIT. No implementation copied.
- scipy/scipy: Gaussian CDF, nonlinear least squares, and independent numerical validation; BSD-3-Clause (primary source license). Dependency, no vendored code.
- numpy/numpy: numerical arrays and algebra; BSD-3-Clause (primary source license). Dependency, no vendored code.
- matplotlib/matplotlib: plotted output; Matplotlib license agreement, full snapshot retained. Dependency, no vendored code.
- The predecessor optical clock lab: MIT; adapted optical functions as noted above.

Each record includes exact current source commit, commit date/message, last push, archived state, and pinned license URL. Reviewed upstream heads are provenance context, not dependency lock revisions: requirements.txt pins the actually installed releases. Installed distributions retain their own release-specific license files. No upstream images, datasets, weights, or nonpermissive code were reused.
