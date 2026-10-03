# POD filter for streaks and flashes in PIV images

Synthetic PIV dataset with exact ground truth, a PIV processor, a POD based
artifact filter, and a benchmark that measures how much the streaks and flashes
hurt the velocity field and how much of that the filter recovers.

## Files

| File | What it does |
|---|---|
| `synth_piv.py` | Generates the dataset: 1000 double frame pairs (2000 images) at 256 x 256, 12 bit, in a clean copy and a corrupted copy that share identical particles. |
| `piv.py` | PIV processor: FFT cross correlation, 4 passes (64 px, then 3 x 32 px at 50% overlap) with symmetric window deformation, cubic B spline image interpolation, Gaussian subpixel fit, normalized median test, plane fit replacement. |
| `pod_filter.py` | The POD filter. Library class `PODFilter` plus a command line tool for real TIFF sequences. |
| `benchmark.py` | Runs PIV on four conditions for all pairs, scores against the analytic truth, sweeps the number of removed modes, writes `results/`. |
| `experiment_every_frame.py` | Every image corrupted, with the flashes fixed, drifting, or at random positions, plus random thin streaks. Results in `results/every_frame.md`. |
| `crosscheck/` | Runs the same TIFFs through OpenPIV and PIVlab and scores them against the truth; compares four automatic rank rules. Results in `results/crosscheck.md`. |
| `LITERATURE.md` | Prior work on POD background removal and automatic rank selection, and what is and isn't new here. |

Requirements: numpy, scipy, matplotlib, pillow (TIFF only).

```bash
python synth_piv.py --out data --pairs 1000            # ~80 s; add --tiff for PIVlab/DaVis
python benchmark.py --data data --out results          # ~12 min on one core
python benchmark.py --data data --out results --max-seconds 250   # resumable chunks
```

Filtering your own images (one glob per laser cavity, filtered separately):

```bash
python pod_filter.py "run1/*_A.tif" "run1/*_B.tif" --out run1_filtered --modes auto
```

## Test case

**Flow.** A Karman vortex street (two staggered rows of Lamb-Oseen vortices,
core radius 24 px, spacing 160 px) convecting on a 6 px uniform stream.
Displacements range from 2.8 to 6.9 px with gradients up to about 0.13 px/px.
Each infinite vortex row is summed in closed form with a regularized cot
kernel, so the field is exact (checked against a brute force sum of 16,000
vortices to 1e-4 px), divergence free to 1e-10, and continuous as the street
convects. Particles are advected with RK4 through the time dependent field.
The truth at each node is the displacement of the particle whose spatial
midpoint lies on the node, which is what a central difference PIV estimate
measures.

**Images.** Pixel integrated Gaussian particles (d = 2.8 ± 0.4 px,
0.05 ppp), Gaussian light sheet with 10% out of plane motion, shot noise,
read noise, dark offset, 12 bit quantization and saturation.

**Artifacts** (corrupted copy only, independent draw per frame):

1. static glow along the bottom wall
2. three fixed reflection streaks, log normal amplitude (35% scatter) and 0.3 px rms jitter
3. two hotspot flashes at fixed locations, each present in 25% of frames, up to 2800 counts
4. global flash with a gradient in 8% of frames
5. random streaks at random position and angle in 3% of frames (not low rank, included to probe the method's limit)

## POD filter in one paragraph

Stack the N images of one cavity as rows of X. Artifacts that repeat in space
are low rank even when their amplitude flickers, jitters, or switches on and
off. Particle images from independent double frame recordings are uncorrelated
from snapshot to snapshot, so their energy spreads evenly over all N modes and
forms a flat bulk. The filter removes the modes above that bulk and
reconstructs X − Ψᵣ Ψᵣᵀ X, then clips negatives. The automatic rank fits a
Marchenko-Pastur law to the lower 75% of the spectrum and removes modes above
1.05 times its upper edge. The effective aspect ratio is fitted, not assumed:
it came out as 0.055 instead of the nominal N/Npix = 0.015, because each
particle spans about 9 correlated pixels. On clean data the rule selects one
mode (the mean image); on the corrupted data it selects 10 for both cavities.
The fit uses the snapshot Gram matrix accumulated in float64 by pixel chunks,
about 2 s per 1000 images here.

The rule is not new in substance: Epps & Krivitzky (2019) fit a
Marchenko-Pastur law with a correlation correction to PIV velocity fields, and
Mendez et al. (2017) already gave an automatic rank for this filter. On this
dataset Gavish-Donoho and ScreeNOT pick slightly fewer modes with the same
accuracy. See `LITERATURE.md`.

## Results (1000 pairs, 225,000 vectors per condition)

| Condition | RMS error (px) | Median (px) | Vectors > 1 px | Flagged by median test |
|---|---|---|---|---|
| Clean images (floor) | 0.080 | 0.032 | 0.02% | 0.89% |
| Corrupted, unfiltered | 2.000 | 0.060 | 1.91% | 2.69% |
| Corrupted, ensemble min subtraction | 1.389 | 0.050 | 0.88% | 1.72% |
| Corrupted, POD filtered (10 modes) | 0.088 | 0.039 | 0.02% | 0.85% |

Where the artifacts sit:

| Subset | Clean | Unfiltered | Min subtraction | POD |
|---|---|---|---|---|
| Windows on fixed streaks | 0.076 | 2.733 | 1.929 | 0.085 |
| Hotspot windows while flashing | 0.069 | 3.003 | 1.926 | 0.080 |
| Artifact free windows, quiet pairs | 0.086 | 0.244 | 0.091 | 0.094 |

Full tables, including the artifact induced error |d_condition − d_clean|, are
in `results/summary.md` and `results/metrics.json`.

The unfiltered and min subtraction numbers above are specific to this
package's processor. It has no guard against runaway vectors, so when it fails
it fails badly (9 px average error among bad vectors, worst 67 px). OpenPIV and
PIVlab on the same images give 0.27 to 0.30 px unfiltered; see the cross check
below. The POD filtered result holds for all three.

**What the numbers say**

* POD brings the corrupted data to within about 10% of the clean RMS, and the
  error tail (fig 4) lies on top of the clean one. The remaining gap is the
  extra shot noise from the flashes, which no background model can subtract.
* The damage from artifacts is almost entirely in the tail. The unfiltered
  median error only doubles, but about 2% of vectors are off by more than 1 px:
  blocks of windows lock onto the stationary pattern (fig 6). The median test
  misses most of them because the bad vectors come in spatially coherent
  clusters that agree with their neighbours.
* Ensemble min subtraction, the usual preprocessing default, only removes what
  is present in every frame. Intermittent flashes have a minimum of zero, so it
  barely helps where it matters.
* Where there is nothing to remove, POD costs essentially nothing: the
  artifact induced error is 0.051 px versus 0.053 px for min subtraction, which
  is the floor set by independent noise realizations of the two copies.

**Mode count sensitivity (fig 5, 100 pairs).** Error is flat from 4 to about
17 removed modes; the automatic choice of 10 sits in the middle. Removing 30,
50, 80 modes raises the RMS only slightly (0.090, 0.092, 0.095 px), since each
bulk mode carries about 1/N of the particle energy. Removing only 1 or 2 modes
is worse than no filtering (2.53 and 1.77 px versus 1.60 px). One mode can only
scale a fixed mix of all the artifacts, so it cannot track sources that
fluctuate independently: for example, with one mode a frame's streaks were
removed if a hotspot happened to flash (13 counts left) and mostly kept if not
(246 counts left). The error collapses once each independent source has its own
mode. With 1000 images, removing extra modes is nearly free, so when unsure
remove more. That stops being true for small ensembles: with 50 images each
mode carries 2% of the particle signal, and removing 25 modes doubled the error
(`LITERATURE.md` has the sweeps for 50, 100 and 200 images).

**Random streaks.** POD does not remove them (the residual near the streak
stays about 3 times higher than elsewhere), as expected. They also barely
affected PIV here, because each one appeared in only one frame of its pair,
and a pattern present in one frame cannot produce a coherent correlation peak.
The harmful case is a pattern that sits in the same place in both frames,
which is exactly what POD removes. Thin streaks that land somewhere new every
pair but appear in both frames are the one case POD only partly fixes; see
the next section.

## What if every image is corrupted?

`experiment_every_frame.py` puts streaks, both hotspots and a global flash in
every frame (no artifact free images at all), 400 pairs for the POD ensemble,
150 pairs evaluated. The moving artifacts sit at the same place in frames A
and B of a pair, since A and B are microseconds apart.

| Scenario | Unfiltered | POD (modes) | POD + high pass | Clean |
|---|---|---|---|---|
| Flashes fixed in place | 0.669 | 0.090 (8) | 0.091 | 0.080 |
| Flashes drift ~6 px between pairs | 0.890 | 0.093 (17) | 0.094 | 0.081 |
| Broad flashes at a random place every pair | 1.205 | 0.102 (57) | 0.103 | 0.078 |
| Thin streaks at a random place every pair | 0.979 | 0.165 (8) | 0.131 | 0.077 |

RMS error in px. Spatial high pass alone (image minus a 5 px Gaussian blur)
was worse than POD in every scenario (0.13 to 0.30 px).

Clean frames are not required. What POD needs is for the artifacts to be low
dimensional across the ensemble. Two kinds qualify:

* artifacts that repeat in the same place, whatever their intensity does
* artifacts that are spatially smooth, like broad glare, wherever they land:
  all smooth blobs are spanned by a limited set of low spatial frequency
  modes, where particles carry little energy

What does not qualify is sharp structure that lands somewhere new every time,
such as thin streaks at random positions and angles. POD removes everything
else around them, and a high pass afterwards helps a little, but the error
stays about twice the clean level. If such streaks come from a moving model,
registering the images to the model before filtering would make them fixed in
place again (not tested here).

## Cross check with OpenPIV and PIVlab

To make sure the conclusions don't depend on this package's processor, the
TIFFs were filtered once with the command line tool (10 modes per cavity) and
then fed to two independent processors with the same pass schedule (64 px,
then three 32 px passes at 50% overlap). All 1000 pairs, RMS error against the
exact truth, with the share of vectors off by more than 1 px:

| Condition | This package | OpenPIV 0.26.1 | PIVlab 3.09 |
|---|---|---|---|
| Clean images | 0.080 (0.02%) | 0.087 (0.00%) | 0.162 (0.03%) |
| Corrupted, unfiltered | 2.000 (1.91%) | 0.297 (1.15%) | 0.270 (0.50%) |
| Corrupted, min subtraction | 1.389 (0.88%) | 0.222 (0.53%) | 0.204 (0.20%) |
| Corrupted, PIVlab high-pass filter | | | 0.213 (0.19%) |
| Corrupted, POD filtered TIFFs | 0.088 (0.02%) | 0.097 (0.00%) | 0.165 (0.03%) |

* POD brings every processor back to within 2 to 12% of its own clean result,
  with the same share of bad vectors as clean images. Min subtraction and
  PIVlab's high-pass filter help but leave several times more bad vectors.
* How much the artifacts hurt depends strongly on the processor: 2.0 px RMS
  here versus under 0.3 px for the mature codes, which cap runaway vectors.
  The share of bad vectors is the fairer comparison (0.5 to 1.9%).
* Each processor is scored against the truth defined its own way: OpenPIV and
  this package deform both frames (midpoint displacement), PIVlab 3.09 deforms
  only frame B (displacement of the particle starting at the node). Using the
  wrong one adds about 0.015 to 0.05 px on clean images.

Caveats: PIVlab ran in GNU Octave 8.4 with two compatibility patches that do
not change results (`crosscheck/OCTAVE_PATCHES.txt`) and linear instead of
spline image deformation, because Octave's `interp2` can't spline onto a
deformed grid. PIVlab's CLAHE preprocessing could not run (Octave has no
`adapthisteq`); `crosscheck/run_crosscheck.sh` includes it for MATLAB, where
the driver should work but was not tested. DaVis is commercial and was not run.
Full tables: `results/crosscheck.md`.

## Limitations to keep in mind with real data

* Snapshots should be statistically independent. Mendez et al. note the
  method works on non time resolved sequences. For time resolved sequences,
  consecutive particle images are correlated, so particle energy should leak
  into the leading modes (expected, not tested here); subsampling in time is
  the simple fix.
* Filter each laser cavity separately (the CLI takes one glob per cavity).
* Ensemble size sets what can be detected. Artifact eigenvalues grow with N
  while the particle bulk does not, so weak artifacts sink into the bulk for
  small ensembles: on the first 200 images the automatic rule removed 7 modes
  instead of 10 (still inside the flat region of fig 5 here).
* Saturated pixels are lost and cannot be recovered.
* Real reflections move with model vibration; the jitter here showed up as a
  separate derivative mode (mode 7 in fig 2). Larger motion needs more modes,
  or registration before filtering.
* The PIV processor's own floor (0.08 px RMS) is dominated by the vortex cores
  and the image edges; it is identical in every condition and cancels in the
  artifact induced metric.

## Reference

M.A. Mendez, M. Raiola, A. Masullo, S. Discetti, A. Ianiro, R. Theunissen,
J.-M. Buchlin (2017). POD based background removal for particle image
velocimetry. Experimental Thermal and Fluid Science 80, 181–192.
