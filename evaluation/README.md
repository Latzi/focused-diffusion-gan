# Evaluation record

Everything here is what stands behind the numbers in the paper and the supplement.

## notebooks/

One Colab notebook per dataset for the distributional metrics (`Step14_dgm_eval_*_200k_sets.ipynb`) and one per
dataset for the object-level metrics (`Step16_object_eval_*.ipynb`). The dgm-eval notebooks install the public
dgm-eval code, unpack the sample sets, compute DINOv2 representations once per set and then every metric (FD, FD
infinity, KD, precision, recall, density, coverage, FLS and the overfit variants, authenticity, coverage test). The
object notebooks crop the conditioned box of every generated image and of the real image with the same box, score the
crops, and run the detectors for object presence and box adherence. `Step10_dgm_eval_FDGAN_4ch_COCO.ipynb` is the
earlier COCO notebook kept for provenance.

## colab_cells/

Cells we paste into the notebooks: unpacking a transfer bundle (`Step33`), filling the coverage-test column
(`Step14_CT_fill_cell*`), the paired bootstrap (`Step14_bootstrap_cell*`, `Step14_bootstrap_finish_cell`), the custom
detectors for Traffic Signs and COCO (`Step16_*_custom_detector_cell`), the guidance-curve scoring (`Step24`) and the
extraction of the samples used in the figures (`Step41`).

## sheets/

- `Step14_<dataset>_200k_metrics.xlsx`: one row per sample set, every metric, as read from the raw dgm-eval output.
- `Step14_<dataset>_bootstrap_FD.xlsx`: the per-model bootstrap FD and the paired differences with their intervals.
- `Step14_<dataset>_bootstrap_replicates.csv`: the 200 replicate FD values of every set, from one shared resample of
  the real set per replicate, so that any pair can be re-derived.
- `Step16_<dataset>_object_metrics.xlsx`: the object-level metrics per set.

The set names encode the model, the checkpoint, the number of images, the respacing and the sampling options, in the
same way as the sampling job's output folder.

## tables/

`build_step23_tables.py` reads the sheets and the replicate files and writes the LaTeX of Tables 1 to 3 and a JSON
with every interval quoted in the text. Paired differences are point differences between the two sets; the intervals
come from the replicates.

## audit/

`check_bundle*_sheets.py` compare every value of a metric sheet with the raw dgm-eval output file it came from.
`file_sheets_*.py` file the canonical sheets and keep the superseded ones under a dated name. `traffic_box_check.py`
is the check that found the Traffic Signs box-format problem (boxes stored as x, y, width, height).
