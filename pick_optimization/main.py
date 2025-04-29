import sys
import json
import os
from tour_formation.tf_entry import run_tour_formation_entrypoint
from tour_allocation.ta_entry import run_tour_allocation_entrypoint

def main():
    # if len(sys.argv) < 2:
    #     print("Usage: python main.py '<json_params>'")
    #     sys.exit(1)

    # try:
    #     params = json.loads(sys.argv[1])
    # except json.JSONDecodeError:
    #     print("Invalid JSON input")
    #     sys.exit(1)

    # entry_type = params.get("entry_point_type")
    # if entry_type == "tour_formation":
    #     run_tour_formation_entrypoint(params)
    # elif entry_type == "tour_allocation":
    #     run_tour_allocation_entrypoint(params)
    # else:
    #     print(f"Unknown entry_point_type: {entry_type}")
    #     sys.exit(1)
    input_str = os.environ.get("ENTRY_PARAMS") or (sys.argv[1] if len(sys.argv) > 1 else None)
    if not input_str:
        print("Usage: pass JSON as env var ENTRY_PARAMS or as first argument.")
        sys.exit(1)

    try:
        print(f"input_str: {input_str}")
        params = json.loads(input_str)
    except json.JSONDecodeError:
        print("Invalid JSON input")
        sys.exit(1)

    entry_type = params.get("entry_point_type")
    if entry_type == "tour_formation":
        # from tour_formation.tf_entry import run_tour_formation_entrypoint
        print(f"params: {params}")
        entry_point_params = params.get("params")
        run_tour_formation_entrypoint(
            mode = entry_point_params.get("mode"),
            fc_id=entry_point_params.get("fc_id"),
            planning_timestamp=entry_point_params.get("planning_timestamp"),
            input_dir=entry_point_params.get("input_dir"),
            output_dir=entry_point_params.get("output_dir"),
            working_dir=entry_point_params.get("working_dir"),
            labor_headcount=entry_point_params.get("labor_headcount"),
            cluster_id=entry_point_params.get("cluster_id")
        )
    elif entry_type == "tour_allocation":
        # from tour_allocation.ta_entry import run_tour_allocation_entrypoint
        print(f"params: {params}")
        entry_point_params = params.get("params")
        run_tour_allocation_entrypoint(
            fc_id=entry_point_params.get("fc_id"),
            planning_timestamp=entry_point_params.get("planning_timestamp"),
            input_dir=entry_point_params.get("input_dir"),
            output_dir=entry_point_params.get("output_dir"),
            target_tours=entry_point_params.get("target_tours")
        )
    else:
        print(f"Unknown entry_point_type: {entry_type}")
        sys.exit(1)

if __name__ == "__main__":
    main()
