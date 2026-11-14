export type TrackedObject = {
  droneIndex: number;
  pos: number[];
  heading: number;
  vel?: number[];
  acc?: number[];
  position_reference?: "body_origin" | "base_midpoint";
  base_midpoint_measured?: number[];
  pose_r?: number[][];
  pose_t?: number[];
  timestamp?: number;
};
