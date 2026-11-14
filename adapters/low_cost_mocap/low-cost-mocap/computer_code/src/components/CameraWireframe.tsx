import { useMemo } from "react";

type Props = {
  R: number[][];
  t: number[];
  K?: number[][];
  imageSize?: number[];
  toWorldCoordsMatrix: number[][] | null;
};

export default function CameraWireframe({ R, t, K, imageSize, toWorldCoordsMatrix }: Props) {
  const vertices = useMemo(() => {
    const [width, height] = imageSize ?? [1280, 800];
    const fx = K?.[0][0] ?? 933.333;
    const fy = K?.[1][1] ?? fx;
    const cx = K?.[0][2] ?? (width - 1) / 2;
    const cy = K?.[1][2] ?? (height - 1) / 2;
    const depth = 0.4;
    const cameraPoints = [[0, 0, 0], ...[
      [-0.5, -0.5], [width - 0.5, -0.5],
      [width - 0.5, height - 0.5], [-0.5, height - 0.5],
    ].map(([u, v]) => [(u - cx) / fx * depth, (v - cy) / fy * depth, depth])];
    const worldPoints = cameraPoints.map((point) => {
      // OpenCV extrinsics are world -> camera: Xw = R' * (Xc - t).
      const world = [0, 1, 2].map(axis =>
        R.reduce((sum, row, i) => sum + row[axis] * (point[i] - t[i]), 0));
      if (toWorldCoordsMatrix !== null) {
        const input = [-world[0], -world[1], world[2], 1];
        const output = toWorldCoordsMatrix.map(row =>
          row.reduce((sum, value, i) => sum + value * input[i], 0));
        // The legacy backend and Three.js each swap Y/Z, cancelling here.
        return output.slice(0, 3).map(value => value / output[3]);
      }
      return [world[0], world[2], world[1]];
    });
    const edges = [0, 1, 0, 2, 0, 3, 0, 4, 1, 2, 2, 3, 3, 4, 4, 1];
    return new Float32Array(edges.flatMap(index => worldPoints[index]));
  }, [R, t, K, imageSize, toWorldCoordsMatrix]);

  return (
    <lineSegments>
      <bufferGeometry>
        <bufferAttribute attach="attributes-position" args={[vertices, 3]} />
      </bufferGeometry>
      <lineBasicMaterial color="#64748b" />
    </lineSegments>
  );
}
