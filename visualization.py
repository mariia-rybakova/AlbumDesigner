"""Stage-visualizer orchestrator.

Reads `CONFIGS['save_files']` flags to decide which stage PDFs to produce:

    save_files.spreads -> stages_visualizer.spreads.render() -> spreads_layouts.pdf
    save_files.groups  -> stages_visualizer.splits.render()  -> split.pdf
                          stages_visualizer.merges.render()  -> merge.pdf

All output goes into `<output_dir>/<projectId>/album1_analysis/` (a folder)
next to the existing `<output_dir>/<projectId>/album1.pdf` produced by
`process_gallery.py`.

This script holds only argument parsing, path resolution, flag-driven
dispatch and dir creation. All rendering lives under `stages_visualizer/`.

Usage:
    python visualization.py <input_dir> <output_dir>
        [--request files/test_requests/request1.json]
        [--stages-info-dir files/stages_info]
"""

from __future__ import annotations

import argparse
import json
import os

from utils.configs import CONFIGS
from stages_visualizer import spreads as spreads_visualizer
from stages_visualizer import splits as splits_visualizer
from stages_visualizer import merges as merges_visualizer


DEFAULT_STAGES_INFO_DIR = os.path.join('files', 'stages_info')
ANALYSIS_DIR_NAME = 'album1_analysis'

# Stage flag -> (subdir under stages_info, output filename, renderer module)
STAGE_RENDERERS = (
    ('spreads', 'spreads', 'spreads_layouts.pdf', spreads_visualizer),
    ('groups',  'groups',  'split.pdf',          splits_visualizer),
    ('groups',  'groups',  'merge.pdf',          merges_visualizer),
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render analysis PDFs from files/stages_info/.")
    parser.add_argument("input_dir",
                        help="Directory holding per-project image subdirs (matches process_gallery.py).")
    parser.add_argument("output_dir",
                        help="Where the analysis folder should be written.")
    parser.add_argument("--request", default=os.path.join('files', 'test_requests', 'request1.json'),
                        help="Request file used to look up projectId (default: request1.json).")
    parser.add_argument("--stages-info-dir", default=DEFAULT_STAGES_INFO_DIR,
                        help=f"Stages-info root dir (default: {DEFAULT_STAGES_INFO_DIR}).")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    with open(args.request, 'r', encoding='utf-8') as f:
        request = json.load(f)
    project_id = str(request['projectId'])

    images_path = os.path.join(args.input_dir, project_id)
    project_out_dir = os.path.join(args.output_dir, project_id)
    analysis_dir = os.path.join(project_out_dir, ANALYSIS_DIR_NAME)
    os.makedirs(analysis_dir, exist_ok=True)

    save_files = CONFIGS.get('save_files', {}) or {}

    any_rendered = False
    for flag_name, subdir, out_name, renderer in STAGE_RENDERERS:
        if not save_files.get(flag_name, False):
            print(f"[skip] save_files[{flag_name!r}] is off -> {out_name} not generated")
            continue
        stages_subdir = os.path.join(args.stages_info_dir, subdir)
        if not os.path.isdir(stages_subdir):
            print(f"[skip] {stages_subdir} missing -> {out_name} not generated")
            continue
        out_path = os.path.join(analysis_dir, out_name)
        try:
            renderer.render(stages_subdir, images_path, out_path)
            print(f"[ok] wrote {out_path}")
            any_rendered = True
        except Exception as ex:
            print(f"[error] {renderer.__name__}.render failed: {ex}")

    if not any_rendered:
        print(f"No PDFs were rendered. Check CONFIGS['save_files'] and that "
              f"{args.stages_info_dir} contains stage data.")


if __name__ == '__main__':
    main()
