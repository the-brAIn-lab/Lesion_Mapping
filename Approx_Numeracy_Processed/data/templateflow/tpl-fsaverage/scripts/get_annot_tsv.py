# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# Copyright 2024 The NiPreps Developers <nipreps@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# We support and encourage derived works from this project, please read
# about our expectations at
#
#     https://www.nipreps.org/community/licensing/
#
from pathlib import Path
import re
import click
import numpy as np
import pandas as pd
import nibabel as nb

FREESURFER_DEFAULT_ANNOT_MAPPING = {
    "lh.aparc.annot": 1000,
    "rh.aparc.annot": 2000,
    "lh.aparc.a2005s.annot": 1100,
    "rh.aparc.a2005s.annot": 2100,
    "lh.aparc.a2009s.annot": 11100,
    "rh.aparc.a2009s.annot": 12100,
}


def annot2tsv(input_annot, hemi=None, encoding="utf-8"):
    input_annot = Path(input_annot)

    hemi = hemi[:2].lower() if hemi is not None else None
    if hemi not in ('lh', 'rh'):
        if (hemi_match := re.search(r'([lr]h)\.', input_annot.name)) is not None:
            hemi = hemi_match.group()[:2].lower()
        else:
            raise RuntimeError('Hemisphere could not be determined')

    vert_lab, reg_ctable, reg_names = nb.freesurfer.read_annot(input_annot)

    df = pd.DataFrame(
        reg_ctable.byteswap().newbyteorder(),
        columns={
            "red": np.uint8,
            "green": np.uint8,
            "blue": np.uint8,
            "transparency": np.uint8,
            "index": np.uint32,
        }
    )

    df["hemi"] = hemi[0].upper()
    df["alpha"] = 255 - df.transparency
    df["color"] = df.loc[:, ("red", "green", "blue", "alpha")].apply(
        lambda r: "#{:02x}{:02x}{:02x}{:02x}".format(*r),
        axis=1
    )
    df["name"] = [n.decode(encoding) for n in reg_names]

    retcols = ["index", "hemi", "name", "color"]
    if (
        offset := FREESURFER_DEFAULT_ANNOT_MAPPING.get(input_annot.name, None)
    ) is not None:
        df["fs_index"] = np.arange(len(df), dtype=np.uint32) + offset

        sep = "_" if input_annot.name.endswith("a2009s.annot") else "-"
        df["fs_name"] = [sep.join(('ctx', hemi, n)) for n in df["name"]]
        retcols += ["fs_index", "fs_name"]

    return df[retcols]


@click.command()
@click.argument("input_annot", type=click.Path(exists=True))
@click.argument("output_tsv", type=click.Path())
def run(input_annot, output_tsv, encoding="utf-8"):

    input_annot = Path(input_annot)
    lh_annot = (
        input_annot if input_annot.name.startswith('lh')
        else input_annot.parent / input_annot.name.replace('rh', 'lh')
    )
    rh_annot = (
        input_annot if input_annot.name.startswith('rh')
        else input_annot.parent / input_annot.name.replace('lh', 'rh')
    )

    df = pd.concat([
        annot2tsv(a, encoding=encoding) for a in (lh_annot, rh_annot)
    ])

    df.to_csv(
        output_tsv, sep="\t", index=None
    )


if __name__ == '__main__':
    run()
