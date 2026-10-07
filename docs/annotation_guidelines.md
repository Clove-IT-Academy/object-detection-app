# Annotation preparation guide

Finalize exact class definitions before annotation. Document whether truncated, partially
occluded, overlapping, and ambiguous objects are included, and how their visible/full extents
are boxed. Use one tight, consistent box and stable class ID per in-scope instance; exclude
uncertain images or review them rather than guessing a class. Review all annotated images
for missing in-scope objects before designating a negative image.

Selected detection format: one `class_id center_x center_y width height` row, normalized by
image width/height, in a matching `.txt` file. IDs are contiguous from zero and follow the
dataset/model class map. Empty files explicitly represent reviewed negatives. The validator
requires finite values, positive in-image geometry and no duplicate annotation rows.

Before training, review representative examples from every class and scenario plus all flagged
labels. Record annotation tool/version, rights, review findings and dataset revision. Group
frames by source/session before splitting, review near-duplicates manually, and preserve an
independent final test set. No class-specific edge policy is finalized in this template yet.
