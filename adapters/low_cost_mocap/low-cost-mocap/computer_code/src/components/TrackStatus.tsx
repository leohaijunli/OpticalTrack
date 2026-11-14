import { Card, Table } from "react-bootstrap";

type TrackBody = {
  body_id: number;
  name: string;
  frame: string;
  marker_count: number;
  marker_ids: number[];
  reference_ids: number[];
  extra_ids: number[];
  non_coplanar: boolean;
  state: string;
  misses: number;
  age_s: number | null;
  markers_used: number;
  pose_mode: string | null;
  pose_disambiguation?: string;
  pose_rmse_m?: number;
  pose_tilt_deg?: number;
  pose_euler_zyx_deg?: number[];
  used_marker_ids?: number[];
  pos?: number[];
  vel?: number[];
  acc?: number[];
};

type Tracking = {
  timestamp: number;
  bodies: TrackBody[];
  motion_gate_m: number | null;
  extra_tolerance_m: number;
  max_reprojection_mse_px2: number | null;
  position_reference: string;
};

const stateVariant: Record<string, string> = {
  locked: "table-success",
  coasting: "table-warning",
  lost: "table-secondary",
};

export default function TrackStatus({ tracking }: { tracking: Tracking }) {
  return (
    <Card className="mt-2">
      <Card.Header className="py-1">
        <strong>Rig identity &amp; pose</strong>{" "}
        <span className="text-muted small">
          t={tracking.timestamp.toFixed(3)} s, gate{" "}
          {tracking.motion_gate_m === null ? "off" : tracking.motion_gate_m.toFixed(2) + " m"}
        </span>
      </Card.Header>
      <Card.Body className="py-2">
        <div className="small mb-2">
          Position reference: <strong>{tracking.position_reference === "body_origin"
            ? "Body origin (geometric center reference)" : "Base-marker midpoint"}</strong>
          <div className="text-muted">Kalman: position + velocity + acceleration (9 states)</div>
        </div>
        <Table size="sm" bordered responsive className="mb-0">
          <thead>
            <tr>
              <th>Body</th>
              <th>Layout</th>
              <th>State</th>
              <th>Markers</th>
              <th>Pose</th>
              <th>Tilt</th>
            </tr>
          </thead>
          <tbody>
            {tracking.bodies.map((body) => (
              <tr key={body.body_id} className={stateVariant[body.state] ?? ""}>
                <td>
                  <strong>{body.name}</strong>
                  <div className="text-muted small">id {body.body_id}</div>
                  <div className="small">{body.pos?.map(v => v.toFixed(3)).join(", ") ?? "-"} m</div>
                </td>
                <td className="small">
                  {body.frame}
                  <div className="text-muted">
                    ref [{body.reference_ids.join(", ")}]
                    {body.extra_ids.length > 0
                      ? " + extra [" + body.extra_ids.join(", ") + "]"
                      : ""}
                  </div>
                </td>
                <td className="small">
                  {body.state}
                  {body.misses > 0 ? ` (${body.misses})` : ""}
                </td>
                <td className="small">
                  {body.markers_used}/{body.marker_count}
                  <div className="text-muted">
                    {body.pose_mode === "rigid" ? "4-marker / 6-DOF" :
                      body.pose_disambiguation === "temporal_prior" ? "triangle + prior" :
                      body.pose_mode === "triangle3" ? "triangle / ambiguous" : "no pose"}
                  </div>
                </td>
                <td className="small">
                  {body.pose_rmse_m === undefined
                    ? "-"
                    : `${(body.pose_rmse_m * 1000).toFixed(1)} mm rms`}
                </td>
                <td className="small">
                  {body.pose_tilt_deg === undefined
                    ? "-"
                    : `${body.pose_tilt_deg.toFixed(1)}°`}
                </td>
              </tr>
            ))}
          </tbody>
        </Table>
        <div className="text-muted small mt-1">
          Extra-marker budget {tracking.extra_tolerance_m.toFixed(3)} m. Identity is
          geometric; no ground-truth pose is used.
        </div>
      </Card.Body>
    </Card>
  );
}
