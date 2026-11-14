import { MutableRefObject, useEffect, useRef } from "react";
import { Color, InstancedMesh, Matrix4, Object3D, Vector3 } from "three";
import { numberToHexColor } from "../shared/styles/scripts/helpers";
import { TrackedObject } from "../types/tracking";

export default function Objects({filteredObjectsRef, count}: {filteredObjectsRef: MutableRefObject<TrackedObject[][]>, count: number}) {
  const objects = filteredObjectsRef.current[filteredObjectsRef.current.length - 1] ?? []

  const instancedMeshRef = useRef<InstancedMesh>(null)
  const temp = new Object3D()
  const tempColour = new Color()

  useEffect(() => {
    objects.forEach(({pos, heading, droneIndex, pose_r}, i) => {
      temp.position.set(pos[0], pos[2], pos[1]) // y is up in threejs
      if (pose_r) {
        const forward = new Vector3(pose_r[0][0], pose_r[2][0], pose_r[1][0]).normalize()
        const up = new Vector3(pose_r[0][2], pose_r[2][2], pose_r[1][2]).normalize()
        const side = new Vector3().crossVectors(forward, up).normalize()
        // Cone local +Y points along body +X; retain its measured bank/tilt.
        temp.setRotationFromMatrix(new Matrix4().makeBasis(side, forward, up))
      } else {
        temp.setRotationFromMatrix(new Matrix4().makeRotationY(heading)
          .multiply(new Matrix4().makeRotationZ(Math.PI / 2)))
      }
      temp.updateMatrix()
      instancedMeshRef.current!.setMatrixAt(i, temp.matrix)
      instancedMeshRef.current!.setColorAt(i, tempColour.set(numberToHexColor(droneIndex, 2))) 
    })
    instancedMeshRef.current!.instanceMatrix.needsUpdate = true 
    if (instancedMeshRef.current!.instanceColor) instancedMeshRef.current!.instanceColor!.needsUpdate = true
    instancedMeshRef.current!.computeBoundingSphere()
  }, [count])
  return (
    <instancedMesh ref={instancedMeshRef} args={[undefined, undefined, objects.length]}>
        <coneGeometry args={[0.035, 0.16, 5, 1]}/>
        <meshPhongMaterial/>
    </instancedMesh>
  )
}
