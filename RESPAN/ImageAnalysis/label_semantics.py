# -*- coding: utf-8 -*-
"""Segmentation label semantics: what each integer in a label volume means.

This is data about the trained models rather than analysis code, and it is kept in
its own module for two reasons. It imports nothing, so a test can exercise the
SHIPPED map on a machine with no GPU stack installed; and the model-type table is
findable here instead of buried in the middle of the analysis module.

Keeping the test on the real map matters: a test that copies the table cannot
notice when the table changes, which is the same class of mistake as reading the
map in the wrong direction.
"""

# {model type: {label value: class name}}
#
# Model type 2 lists spines at label 10, which no model emits. nnU-Net labels
# sequentially from 0, so 10 can never match; it is a deliberate placeholder that
# yields an empty spine mask for a model that segments dendrites and soma only.
LABEL_CHANNEL_MAP = {
    1: {1: 'spines', 2: 'dendrites', 3: 'soma'},
    2: {1: 'dendrites', 2: 'soma', 10: 'spines'},
    3: {1: 'spines', 2: 'dendrites', 3: 'soma', 4: 'necks'},
    4: {1: 'dendrites', 2: 'spine_cores', 3: 'spine_membranes', 4: 'necks', 5: 'soma'},
}


def soma_label_from_channel_map(channel_map):
    """Return the label VALUE whose class name is 'soma', or None if there is none.

    The map is keyed by label value and holds class names, so indexing it by an
    assumed position returns a NAME. Comparing a label array against a name yields
    all-False with no error, which is how every chunked run reported "no soma" even
    with over a million soma voxels segmented. Look the label up by name instead.
    """
    return next((k for k, v in channel_map.items() if v == 'soma'), None)
