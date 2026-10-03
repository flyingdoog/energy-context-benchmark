"""Prepare the EPFL query packet without fitting or importing a model library."""
import argparse
import json
from pathlib import Path
import run_local
import numpy as np
from field_control.epfl_preparation import build_packet,TARGET_NAMES

ROOT=Path(__file__).resolve().parents[1]


def main(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument("--source",type=Path,default=ROOT/"results/benchmark_source_feasibility_v1/epfl_level3")
    parser.add_argument("--output",type=Path,default=ROOT/"results/epfl_dataset_v1")
    args=parser.parse_args(argv);dest=args.output.resolve()
    if (dest/"receipt.json").exists():
        raise ValueError("Packet already exists; reuse it or choose a new output directory")
    meta,x,y=build_packet(args.source)
    if meta.empty:raise ValueError("No eligible source sessions")
    dest.mkdir(parents=True,exist_ok=True)
    if (dest/"packet.npz").exists() or (dest/"cases.csv").exists():
        raise ValueError("Existing incomplete packet requires inspection before replacement")
    meta.to_csv(dest/"cases.csv",index=False)
    np.savez_compressed(dest/"packet.npz",x=x,y=y,case_id=meta.case_id.to_numpy(dtype=str))
    receipt=dict(completed=True,source="EPFL Level-3 EV charging dataset",
        source_url="https://github.com/DESL-EPFL/Level-3-EV-charging-dataset",
        query_minutes=10,history_minutes=10,target_bins_minutes=5,spans_minutes=[5,15,30],
        target_names=list(TARGET_NAMES),source_clock="Naive recorded timestamps; physical timezone unresolved",
        end_rule="Zero only after source-confirmed departure of the same session; no interior fill",
        support=meta.groupby("split").agg(sessions=("case_id","size"),dates=("date","nunique")).reset_index().to_dict(orient="records"),
        source_modified=False,new_model_fits=0)
    (dest/"receipt.json").write_text(json.dumps(receipt,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(receipt),flush=True)


if __name__=="__main__":main()
