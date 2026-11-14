"""Build a self-contained, interactive 3-D view of the mocap rig + hangar.

Reads the same ground-truth files the renderer uses, then writes one single
HTML file with no external dependencies (no CDN, no web server, no CORS
issues): open it by double-clicking.

    python tools/make_rig_viz.py
    -> out/viz/camera_rig_3d.html

Left-drag orbits, wheel zooms, right-drag (or shift-drag) pans, double-click
a camera to focus it. Each camera's field of view can be toggled separately
in the right-hand panel or the legend without hiding the camera itself.
"""
import argparse
import csv
import json
import math
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
RIG = os.path.join(ROOT, "out", "rig3", "rig_blender.json")
TRAJ = os.path.join(ROOT, "out", "scenario3.csv")
CALIB = os.path.join(ROOT, "out", "rig3", "cameras_calib.json")
OUT = os.path.join(ROOT, "out", "viz", "camera_rig_3d.html")


def collect():
    rig = json.load(open(RIG))
    calib = json.load(open(CALIB))
    sx, sy = rig["resolution"]

    # The hangar, door and lamp grid come from the layout that produced the rig,
    # exactly like build_room()/build_lights() in blender/mocap_scene.py.
    width, depth, height = rig["hangar"]["size"]
    door = rig["hangar"].get("door") or {"x": 0.0, "width": 6.0, "height": 4.0}
    lamps = [[sx_ * width * 0.23, sy_ * depth * 0.28]
             for sx_, sy_ in ((-1, -1), (1, -1), (-1, 1), (1, 1))]
    center = rig["volume"]["center"]
    mount_z = max(c["pos"][2] for c in rig["cameras"])

    cams = []
    for c in rig["cameras"]:
        q = c["quat"]  # Blender order: w, x, y, z
        pos = c["pos"]
        fwd = quat_rotate(q, (0.0, 0.0, -1.0))
        right = quat_rotate(q, (1.0, 0.0, 0.0))
        up = quat_rotate(q, (0.0, 1.0, 0.0))
        K = c["intrinsics"]
        cams.append({
            "name": c["name"],
            "id": int(c["name"][-2:]),          # CAM01 -> 1
            "pos": pos,
            "target": c["target"],
            "az": math.degrees(math.atan2(pos[1], pos[0])) % 360.0,
            "radius": math.hypot(pos[0], pos[1]),
            "fwd": fwd,
            "right": right,
            "up": up,
            "fx": K["fx"],
            "tanH": (sx * 0.5) / K["fx"],
            "tanV": (sy * 0.5) / K["fx"],
            "hfov": 2.0 * math.degrees(math.atan((sx * 0.5) / K["fx"])),
            "vfov": 2.0 * math.degrees(math.atan((sy * 0.5) / K["fx"])),
            "lens": c["lens_mm"],
            "tilt": math.degrees(math.acos(max(-1.0, min(1.0, -fwd[2])))),
            "dist": math.dist(pos, center),
        })

    names = {b["body_id"]: b.get("name", f"body {b['body_id']}")
             for b in rig.get("bodies", [])}
    raw = {}
    with open(TRAJ, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            raw.setdefault(int(r["body_id"]), []).append(
                [float(r["x"]), float(r["y"]), float(r["z"])])
    trajectories = []
    for bid in sorted(raw):
        pts = raw[bid]
        step = max(1, len(pts) // 300)
        pts = pts[::step]
        if pts[-1] != raw[bid][-1]:
            pts.append(raw[bid][-1])
        trajectories.append({"body_id": bid, "name": names.get(bid, f"body {bid}"),
                             "points": pts})

    return {
        "hangar": {"x": width, "y": depth, "z": height,
                   "door": {"x": float(door["x"]), "w": float(door["width"]),
                            "h": float(door["height"])},
                   "lamps": lamps},
        "resolution": rig["resolution"],
        "volume": rig["volume"],
        "mount_z": mount_z,
        "rail_z": mount_z + 0.45,
        "roof_z": height - 0.30,
        "markers": rig.get("markers", []),
        "cameras": cams,
        "trajectories": trajectories,
        "fps": rig.get("render_fps", 30),
        "time": calib.get("time", {}),
    }


def quat_rotate(q, v):
    w, x, y, z = q
    qv = (x, y, z)
    t = cross(qv, v)
    t = [2.0 * c for c in t]
    t2 = cross(qv, t)
    return [v[i] + w * t[i] + t2[i] for i in range(3)]


def cross(a, b):
    return [a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0]]


HTML = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>mocap-sim · 机库 + 6 机位 3D 全景</title>
<style>
  :root{
    --bg:#07090d; --panel:#10141bcc; --line:#1e2634;
    --fg:#dce3ee; --dim:#7d8798;
  }
  *{box-sizing:border-box}
  html,body{margin:0;height:100%;background:var(--bg);color:var(--fg);
    font:13px/1.5 "Segoe UI",system-ui,-apple-system,sans-serif;overflow:hidden}
  #stage{position:fixed;inset:0}
  canvas{display:block;width:100%;height:100%;cursor:grab}
  canvas.drag{cursor:grabbing}

  .panel{position:fixed;background:var(--panel);border:1px solid var(--line);
    border-radius:10px;backdrop-filter:blur(10px)}
  #head{top:16px;left:16px;padding:14px 16px;max-width:352px}
  #head h1{margin:0 0 3px;font-size:15px;letter-spacing:.4px}
  #head .sub{color:var(--dim);font-size:11.5px;line-height:1.6}
  #head .meta{margin-top:8px;padding-top:8px;border-top:1px solid var(--line);
    color:var(--dim);font-size:11px;font-family:Consolas,monospace}
  #head .meta b{color:var(--fg);font-weight:600}

  #ui{top:16px;right:16px;padding:12px;width:242px;
    max-height:calc(100vh - 32px);overflow-y:auto;scrollbar-width:thin}
  #ui .row{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:7px}
  #ui .row:last-child{margin-bottom:0}
  button{flex:1 1 auto;min-width:62px;padding:6px 8px;border-radius:6px;
    border:1px solid #263041;background:#151b25;color:var(--fg);
    font:inherit;font-size:11.5px;cursor:pointer;transition:.12s}
  button:hover{background:#1d2532;border-color:#35435b}
  button.on{background:#123244;border-color:#2f7ea4;color:#9fe4ff}
  label.chk{display:flex;align-items:center;gap:7px;padding:3px 2px;
    font-size:11.5px;color:#b9c3d2;cursor:pointer;user-select:none}
  label.chk input{accent-color:#59d3ff;cursor:pointer}

  #frusUi{margin:10px 0;padding:9px 0;border-top:1px solid var(--line);
    border-bottom:1px solid var(--line)}
  #frusUi .ttl{display:flex;justify-content:space-between;align-items:center;
    margin-bottom:4px;font-size:11.5px}
  #frusCount{color:#8fd8f5;font:11px Consolas,monospace}
  #camFrus{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));
    gap:3px 6px;margin:6px 0 8px}
  #camFrus label{color:var(--cam);font-family:Consolas,monospace}
  #camFrus input{accent-color:var(--cam)}
  #camFrus label.off{opacity:.5}
  #frusHint{color:var(--dim);font-size:10px;line-height:1.6}

  #legend{left:16px;bottom:16px;padding:10px 12px;width:428px}
  #legend .ttl{color:var(--dim);font-size:10.5px;letter-spacing:.9px;
    text-transform:uppercase;margin-bottom:7px}
  #legend table{width:100%;border-collapse:collapse;font-size:11px;
    font-family:Consolas,monospace}
  #legend td{padding:2px 0}
  #legend td:first-child{width:14px}
  #legend .sw{display:inline-block;width:9px;height:9px;border-radius:2px}
  #legend .nm{color:var(--fg)}
  #legend .num{color:var(--dim);text-align:right}
  #legend .cb{width:40px;text-align:center}
  #legend .cb input{accent-color:#59d3ff;cursor:pointer;vertical-align:-2px}
  #legend .cb.all input{accent-color:#8fd8f5}
  #legend .cb label{cursor:pointer}
  #legend tr.hl .nm{color:#fff}

  #hint{bottom:16px;right:16px;padding:9px 12px;color:var(--dim);font-size:11px;
    line-height:1.75;text-align:right}
  kbd{background:#1b2330;border:1px solid #2b3648;border-bottom-width:2px;
    border-radius:4px;padding:1px 5px;font:11px Consolas,monospace;color:#cfd8e6}
  #vdist{color:#8fd8f5;font-family:Consolas,monospace}
  @media (max-width:860px){#legend,#hint{display:none}#ui{width:172px}}
</style>
</head>
<body>
<div id="stage"><canvas id="cv"></canvas></div>

<div id="head" class="panel">
  <h1>mocap-sim · 机库与 6 机位吊装</h1>
  <div class="sub">Arducam OV9281 · 1280×800 mono · M27280M07S 2.8 mm<br>
    环形半径 __RADIUS__ m，安装高度 __HEIGHT__ m，屋顶吊装（无落地杆）</div>
  <div class="meta">
    机库 <b>__HANGAR__ m</b>　捕捉体积 <b>__VOLUME__ m</b><br>
    基线 <b>__BASELINE__ m</b>　俯角 <b>__TILT__</b>　HFOV <b>__HFOV__</b><br>
    编号：俯视逆时针，自 +X（东）起，间隔 60°
  </div>
</div>

<div id="ui" class="panel">
  <div class="row">
    <button id="bAll">机库全景</button>
    <button id="bRig">机位特写</button>
  </div>
  <div class="row">
    <button id="bTop">俯视</button>
    <button id="bSide">侧视</button>
    <button id="bFrame">贴地</button>
  </div>
  <div class="row">
    <button id="bSpin">自动旋转</button>
    <button id="bReset">复位</button>
  </div>
  <div id="frusUi" role="group" aria-labelledby="frusTitle">
    <div class="ttl"><b id="frusTitle">相机视野（视锥）</b>
      <span id="frusCount" aria-live="polite" title="实际显示的视野数量">6 / 6</span></div>
    <label class="chk"><input type="checkbox" id="cFrus" checked> 显示已选视野</label>
    <div id="camFrus"></div>
    <div class="row">
      <button id="bFrAll">全选</button>
      <button id="bFrNone">清空</button>
    </div>
    <div id="frusHint">只切换视野，不隐藏相机。</div>
  </div>
  <div class="row" style="display:block">
    <label class="chk"><input type="checkbox" id="cIds" checked> 标准编号 / 方位角</label>
    <label class="chk"><input type="checkbox" id="cTraj" checked> 三机飞行轨迹</label>
    <label class="chk"><input type="checkbox" id="cHang" checked> 机库外壳（墙 / 屋顶 / 门）</label>
    <label class="chk"><input type="checkbox" id="cMount" checked> 吊装结构</label>
    <label class="chk"><input type="checkbox" id="cVol" checked> 捕捉体积</label>
    <label class="chk"><input type="checkbox" id="cLabel" checked> 相机标签</label>
  </div>
</div>

<div id="legend" class="panel">
  <div class="ttl">标准编号：俯视逆时针，自 +X（东）起 · 每 60° 一台</div>
  <table id="tbl"></table>
</div>

<div id="hint" class="panel">
  <kbd>拖拽</kbd> 旋转　<kbd>滚轮</kbd> 缩放　<kbd>右键拖拽</kbd> 平移<br>
  <kbd>双击</kbd> 相机 = 聚焦　<kbd>R</kbd> 复位　<kbd>H</kbd> 全景<br>
  视距 <span id="vdist">–</span> m
</div>

<script>
const DATA = __DATA__;

const COLORS = ["#ff7a59","#ffc247","#7ee787","#59d3ff","#a78bfa","#f472b6"];

const cv = document.getElementById("cv");
const ctx = cv.getContext("2d");
let W = 0, H = 0, DPR = 1;
const FOVY = 42 * Math.PI / 180;

const view  = { tgt: [0, 0, 2.8], yaw: 0.72, pitch: 0.30, dist: 22 };
const smooth= { tgt: [0, 0, 2.8], yaw: 0.72, pitch: 0.30, dist: 22 };
const opt = { frus: true, traj: true, hang: true, mount: true, vol: true,
              label: true, ids: true, spin: false, hl: -1 };
// per-camera frustum switches (the master gate is opt.frus)
const frusOn = DATA.cameras.map(()=>true);
const RAD = __RADIUS__, MOUNT_Z = __MOUNTZ__, ROOF_Z = __ROOFZ__, RAIL_Z = __RAILZ__;
const RAIL_R = RAD + 0.55;
const G = DATA.hangar;
const HX = G.x/2, HY = G.y/2, HZ = G.z;

/* ---------- vector helpers ---------- */
const sub = (a,b)=>[a[0]-b[0],a[1]-b[1],a[2]-b[2]];
const add = (a,b)=>[a[0]+b[0],a[1]+b[1],a[2]+b[2]];
const mul = (a,s)=>[a[0]*s,a[1]*s,a[2]*s];
const lerp = (a,b,t)=>[a[0]+(b[0]-a[0])*t,a[1]+(b[1]-a[1])*t,a[2]+(b[2]-a[2])*t];
const dot = (a,b)=>a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
const cross = (a,b)=>[a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]];
const len = a=>Math.hypot(a[0],a[1],a[2]);
const norm = a=>{const l=len(a)||1;return [a[0]/l,a[1]/l,a[2]/l];};

function eyePos(v){
  const cp=Math.cos(v.pitch), sp=Math.sin(v.pitch);
  return [v.tgt[0]+v.dist*cp*Math.cos(v.yaw),
          v.tgt[1]+v.dist*cp*Math.sin(v.yaw),
          v.tgt[2]+v.dist*sp];
}
function basis(v){
  const e=eyePos(v);
  const f=norm(sub(v.tgt,e));
  let r=norm(cross(f,[0,0,1]));
  const u=cross(r,f);
  return {e,f,r,u,fpx:(H/2)/Math.tan(FOVY/2)};
}
let B = null;
// The left info panel is wider than the right control panel, so nudge the
// optical centre right/down a little to keep the scene out from under them.
const OFFX = 48, OFFY = 16;
function proj(p){
  const v=sub(p,B.e), z=dot(v,B.f);
  if (z <= 0.08) return null;
  return [W/2 + OFFX + dot(v,B.r)/z*B.fpx,
          H/2 + OFFY - dot(v,B.u)/z*B.fpx, z];
}
/** distance that makes a box of the given half extents fit the viewport */
function fitDist(half){
  const rad = Math.hypot(half[0],half[1],half[2]);
  const fovx = 2*Math.atan(Math.tan(FOVY/2)*W/Math.max(H,1));
  return rad / Math.sin(Math.min(FOVY,fovx)/2) * 0.96;
}

/* ---------- primitives ---------- */
const items = [];
const labels = [];
const badges = [];
const camLabels = [];
function line(a,b,color,width,dash){
  items.push({kind:"line", pts:[a,b], color, width:width||1, dash:dash||null});
}
function poly(pts, fill, stroke){
  items.push({kind:"poly", pts, fill, stroke});
}
function mark(p, color, size){ items.push({kind:"dot", pts:[p], color, size}); }
function text(p, s, color, size, bold){
  labels.push({p, text:s, color, size:size||11, bold:!!bold});
}

function box(c, h, fill, stroke, edges){
  // c = centre, h = half extents; draws 6 translucent faces or just edges
  const p = i=>[c[0]+h[0]*(i&1?1:-1), c[1]+h[1]*(i&2?1:-1), c[2]+h[2]*(i&4?1:-1)];
  if (fill){
    const f=[[0,2,3,1],[4,5,7,6],[0,1,5,4],[2,6,7,3],[0,4,6,2],[1,3,7,5]];
    for (const q of f) poly(q.map(p), fill, null);
  }
  if (edges){
    const e=[[0,1],[1,3],[3,2],[2,0],[4,5],[5,7],[7,6],[6,4],
             [0,4],[1,5],[2,6],[3,7]];
    for (const [i,j] of e) line(p(i),p(j), edges, 1.2);
  }
}

function hex(c){
  const h=c.replace("#","");
  return `${parseInt(h.slice(0,2),16)},${parseInt(h.slice(2,4),16)},${parseInt(h.slice(4,6),16)}`;
}

/* ---------- scene ---------- */
function buildScene(){
  items.length = 0; labels.length = 0; badges.length = 0; camLabels.length = 0;

  /* ---- outside apron, so the building reads as a building ---- */
  if (opt.hang){
    const A=34, Bq=30;
    poly([[-A,-Bq,-0.10],[A,-Bq,-0.10],[A,Bq,-0.10],[-A,Bq,-0.10]],
         "rgba(26,34,46,0.85)", "rgba(60,76,98,0.35)");
    for (let x=-A; x<=A; x+=6)
      line([x,-Bq,-0.09],[x,Bq,-0.09],"rgba(48,62,82,0.5)",1);
    for (let y=-Bq; y<=Bq; y+=6)
      line([-A,y,-0.09],[A,y,-0.09],"rgba(48,62,82,0.5)",1);
  }

  /* ---- floor slab + inner grid ---- */
  poly([[-HX,-HY,0],[HX,-HY,0],[HX,HY,0],[-HX,HY,0]],
       "rgba(44,56,74,0.55)", "rgba(120,145,180,0.55)");
  for (let x=-HX; x<=HX; x+=1){
    const c = Math.abs(x%5)<0.01 ? "rgba(96,118,150,0.42)" : "rgba(72,88,112,0.22)";
    line([x,-HY,0.005],[x,HY,0.005], c, 1);
  }
  for (let y=-HY; y<=HY; y+=1){
    const c = Math.abs(y%5)<0.01 ? "rgba(96,118,150,0.42)" : "rgba(72,88,112,0.22)";
    line([-HX,y,0.005],[HX,y,0.005], c, 1);
  }

  /* ---- shell: 4 walls + roof, translucent glass box ---- */
  if (opt.hang){
    const WF="rgba(92,118,152,0.10)", WS="rgba(126,156,196,0.42)";
    const walls=[
      [[-HX,-HY,0],[HX,-HY,0],[HX,-HY,HZ],[-HX,-HY,HZ]],
      [[ HX,-HY,0],[HX, HY,0],[HX, HY,HZ],[ HX,-HY,HZ]],
      [[ HX, HY,0],[-HX,HY,0],[-HX,HY,HZ],[ HX,HY,HZ]],
      [[-HX, HY,0],[-HX,-HY,0],[-HX,-HY,HZ],[-HX,HY,HZ]],
    ];
    for (const w of walls) poly(w, WF, WS);

    // corrugated ribs
    for (let x=-HX+2.2; x<HX; x+=2.2){
      line([x,-HY,0],[x,-HY,HZ],"rgba(100,126,162,0.16)",1);
      line([x, HY,0],[x, HY,HZ],"rgba(100,126,162,0.16)",1);
    }
    for (let y=-HY+2.2; y<HY; y+=2.2){
      line([-HX,y,0],[-HX,y,HZ],"rgba(100,126,162,0.16)",1);
      line([ HX,y,0],[ HX,y,HZ],"rgba(100,126,162,0.16)",1);
    }

    // roof plane + trusses + purlins
    poly([[-HX,-HY,HZ],[HX,-HY,HZ],[HX,HY,HZ],[-HX,HY,HZ]],
         "rgba(60,80,106,0.13)", "rgba(126,156,196,0.34)");
    const tc="rgba(112,136,172,0.36)";
    for (let y=-7.2; y<=7.21; y+=2.4){
      line([-HX,y,HZ-0.30],[HX,y,HZ-0.30], tc, 1.2);
      line([-HX,y,HZ-0.70],[HX,y,HZ-0.70], "rgba(112,136,172,0.22)", 1);
      for (let x=-10.4; x<10.4; x+=2.6){
        line([x,y,HZ-0.70],[x+1.3,y,HZ-0.30],"rgba(112,136,172,0.16)",1);
        line([x+1.3,y,HZ-0.30],[x+2.6,y,HZ-0.70],"rgba(112,136,172,0.16)",1);
      }
    }
    for (let x=-10.4; x<=10.41; x+=1.3)
      line([x,-HY,HZ-0.20],[x,HY,HZ-0.20],"rgba(112,136,172,0.19)",1);

    // sliding door on the south wall (x centre +4, 11 m wide, 6 m tall)
    const d=G.door, dx=d.x, dw=d.w/2;
    poly([[dx-dw,-HY+0.02,0],[dx+dw,-HY+0.02,0],
          [dx+dw,-HY+0.02,d.h],[dx-dw,-HY+0.02,d.h]],
         "rgba(120,150,190,0.13)","rgba(150,180,220,0.6)");
    for (let k=1;k<=6;k++)
      line([dx-dw,-HY+0.03,d.h*k/7],[dx+dw,-HY+0.03,d.h*k/7],
           "rgba(150,180,220,0.32)",1);
    line([dx,-HY+0.03,0],[dx,-HY+0.03,d.h],"rgba(150,180,220,0.5)",1.4);

    // ceiling luminaires (glow + hood)
    for (const [lx,ly] of G.lamps){
      const z=HZ-0.85;
      poly([[lx-1.6,ly-0.45,z],[lx+1.6,ly-0.45,z],
            [lx+1.6,ly+0.45,z],[lx-1.6,ly+0.45,z]],
           "rgba(255,244,214,0.55)","rgba(255,248,225,0.9)");
      line([lx-1.8,ly,HZ],[lx-1.8,ly,z],"rgba(140,150,165,0.35)",1);
      line([lx+1.8,ly,HZ],[lx+1.8,ly,z],"rgba(140,150,165,0.35)",1);
    }
  }

  /* ---- ceiling rail ring the cameras bolt to ---- */
  if (opt.mount){
    const ring=[];
    for (let i=0;i<=96;i++){
      const a=i/96*Math.PI*2;
      ring.push([RAIL_R*Math.cos(a), RAIL_R*Math.sin(a), RAIL_Z]);
    }
    for (let i=0;i<ring.length-1;i++)
      line(ring[i],ring[i+1],"rgba(160,182,214,0.62)",1.6);

    for (const c of DATA.cameras){
      const p=c.pos, o=norm([p[0],p[1],0]);
      const r=[p[0]+o[0]*0.55, p[1]+o[1]*0.55, RAIL_Z];
      line(p, r, "rgba(160,182,214,0.55)", 1.4);
      line(r, [r[0],r[1],ROOF_Z], "rgba(160,182,214,0.45)", 1.2);
      const pl=0.16;
      poly([[r[0]-pl,r[1]-pl,ROOF_Z],[r[0]+pl,r[1]-pl,ROOF_Z],
            [r[0]+pl,r[1]+pl,ROOF_Z],[r[0]-pl,r[1]+pl,ROOF_Z]],
           "rgba(160,182,214,0.28)","rgba(180,200,226,0.6)");
    }
  }

  /* ---- capture volume ---- */
  if (opt.vol){
    const v=DATA.volume, h=[v.size[0]/2,v.size[1]/2,v.size[2]/2], c=v.center;
    box(c, h, "rgba(70,190,235,0.055)", null,
        "rgba(90,220,255,0.55)");
    // take-off pad
    poly([[-0.42,-1.15,0.012],[0.42,-1.15,0.012],
          [0.42,-0.25,0.012],[-0.42,-0.25,0.012]],
         "rgba(120,200,255,0.18)","rgba(120,210,255,0.65)");
    text([c[0]-h[0]-0.10, c[1]-h[1]-0.10, c[2]+h[2]+0.35], "capture volume", "#5adcff", 11);
  }

  /* ---- standard numbering: azimuth ring, CCW order, ID badges ---- */
  if (opt.ids){
    // dashed reference circle at the camera radius, on the floor
    const ringPts=[];
    for (let i=0;i<=120;i++){
      const a=i/120*Math.PI*2;
      ringPts.push([RAD*Math.cos(a), RAD*Math.sin(a), 0.03]);
    }
    for (let i=0;i<ringPts.length-1;i+=2)
      line(ringPts[i],ringPts[i+2],"rgba(140,165,200,0.30)",1);

    // per-camera azimuth tick (the azimuth itself is printed on the camera label)
    for (const c of DATA.cameras){
      const a=c.az*Math.PI/180;
      const p=[RAD*Math.cos(a), RAD*Math.sin(a), 0.03];
      const q=[(RAD+0.55)*Math.cos(a), (RAD+0.55)*Math.sin(a), 0.03];
      line(p,q,"rgba(150,175,210,0.55)",1.3);
      line([(RAD+0.30)*Math.cos(a+0.02),(RAD+0.30)*Math.sin(a+0.02),0.03],
           [(RAD+0.30)*Math.cos(a-0.02),(RAD+0.30)*Math.sin(a-0.02),0.03],
           "rgba(150,175,210,0.5)",1.2);
    }

    // CCW order arrow: az 30 -> 90
    const arc=[];
    for (let i=0;i<=10;i++){
      const a=(30 + 58*i/10)*Math.PI/180;
      arc.push([(RAD-0.75)*Math.cos(a),(RAD-0.75)*Math.sin(a),0.05]);
    }
    for (let i=0;i<arc.length-1;i++)
      line(arc[i],arc[i+1],"rgba(120,200,240,0.75)",1.6);
    const ae=arc[arc.length-1], ab=arc[arc.length-2];
    const dir=norm(sub(ae,ab)), nrm=[-dir[1],dir[0],0];
    line(ae, add(ae, add(mul(dir,-0.34), mul(nrm, 0.16))), "rgba(120,200,240,0.75)",1.6);
    line(ae, add(ae, add(mul(dir,-0.34), mul(nrm,-0.16))), "rgba(120,200,240,0.75)",1.6);
    text([(RAD-1.55)*Math.cos(44*Math.PI/180),(RAD-1.55)*Math.sin(44*Math.PI/180),0.05],
         "1 → 2 → 3 … CCW", "#78c8f0", 10);
  }

  /* ---- per-body flight paths ---- */
  if (opt.traj){
    DATA.trajectories.forEach((tr,ti)=>{
      const p=tr.points, col=COLORS[ti%COLORS.length];
      for (let i=0;i<p.length-1;i++)
        line(p[i],p[i+1],col,1.9);
      for (let i=0;i<p.length;i+=Math.max(1,Math.floor(p.length/12)))
        mark(p[i],"rgba(255,255,255,0.55)",2.6);
      if (p.length)
        text(p[0], tr.name, col, 11);
    });
  }

  /* ---- cameras ---- */
  DATA.cameras.forEach((c,ci)=>{
    const col=COLORS[ci%COLORS.length];
    const far=Math.min(11.0, c.dist*1.25);
    const n=0.35, hh=n*c.tanH, hv=n*c.tanV;
    const fh=far*c.tanH, fv=far*c.tanV;
    const q=(a,b,d)=>add(c.pos, add(mul(c.fwd,d), add(mul(c.right,a), mul(c.up,b))));
    const corner=[[-1,-1],[1,-1],[1,1],[-1,1]];
    const nq=corner.map(([a,b])=>q(a*hh,b*hv,n));
    const fq=corner.map(([a,b])=>q(a*fh,b*fv,far));

    const showFr = opt.frus && frusOn[ci];
    if (showFr){
      for (let i=0;i<4;i++){
        const j=(i+1)%4;
        poly([nq[i],nq[j],fq[j],fq[i]], `rgba(${hex(col)},0.05)`, null);
      }
      poly(fq, `rgba(${hex(col)},0.09)`, `rgba(${hex(col)},0.6)`);
      for (let i=0;i<4;i++){
        line(nq[i],fq[i], `rgba(${hex(col)},0.20)`, 1);
        line(fq[i],fq[(i+1)%4], `rgba(${hex(col)},0.55)`, 1.3);
      }
      line(c.pos, add(c.pos, mul(c.fwd, far*0.95)),
           `rgba(${hex(col)},0.55)`, 1.2, [5,4]);
    }

    // camera body (exaggerated so it is visible at hangar scale)
    const s=0.16;
    const bq=[];
    for (const a of [-1,1]) for (const b of [-1,1]) for (const d of [-1,1])
      bq.push(add(c.pos, add(mul(c.right,a*s), add(mul(c.up,b*s), mul(c.fwd,d*s*0.7)))));
    for (let i=0;i<4;i++){
      line(bq[i],bq[(i+1)%4], `rgba(${hex(col)},0.9)`, 1.3);
      line(bq[i+4],bq[((i+1)%4)+4], `rgba(${hex(col)},0.9)`, 1.3);
      line(bq[i],bq[i+4], `rgba(${hex(col)},0.9)`, 1.3);
    }
    mark(c.pos, `rgb(${hex(col)})`, opt.hl===ci?9:6);
    if (opt.ids) badges.push({p:c.pos, color:col, n:c.id, hl:opt.hl===ci});

    // label sits outboard of the badge with a leader, so the two never overlap
    const azd = c.az * Math.PI/180;
    const lp = add(c.pos, [Math.cos(azd)*3.1, Math.sin(azd)*3.1, 0.10]);
    line(c.pos, lp, `rgba(${hex(col)},0.30)`, 1);
    camLabels.push({p:lp, anchor:c.pos, color:col,
                    text: opt.ids ? `${c.name} · ${c.az.toFixed(0)}°` : c.name,
                    size: opt.hl===ci?13:11.5, bold: opt.hl===ci});
  });

  // origin triad
  const O=[0,0,0];
  line(O,[1.5,0,0],"rgba(255,96,96,0.85)",1.8);
  line(O,[0,1.5,0],"rgba(120,255,150,0.85)",1.8);
  line(O,[0,0,1.5],"rgba(130,175,255,0.85)",1.8);
  text([1.62,0,0],"x (E)","#ff6b6b",10);
  text([0,1.62,0],"y (N)","#7ee787",10);
  text([0,0,1.66],"z (U)","#7ea8ff",10);
}

/* ---------- render ---------- */
let lastT = 0;
function frame(t){
  const dt = Math.min(0.05,(t-lastT)/1000||0); lastT=t;

  if (opt.spin) view.yaw += dt*0.25;
  const k = Math.min(1, dt*9);
  smooth.yaw  += (view.yaw  -smooth.yaw )*k;
  smooth.pitch+= (view.pitch-smooth.pitch)*k;
  smooth.dist += (view.dist -smooth.dist)*k;
  for (let i=0;i<3;i++) smooth.tgt[i] += (view.tgt[i]-smooth.tgt[i])*k;
  B = basis(smooth);

  ctx.setTransform(DPR,0,0,DPR,0,0);
  ctx.clearRect(0,0,W,H);
  const g=ctx.createRadialGradient(W/2,H*0.46,Math.min(W,H)*0.12,W/2,H/2,Math.max(W,H)*0.8);
  g.addColorStop(0,"#0b1119"); g.addColorStop(1,"#04060a");
  ctx.fillStyle=g; ctx.fillRect(0,0,W,H);

  buildScene();

  const draw=[];
  for (const it of items){
    const ps=it.pts.map(proj);
    if (ps.some(p=>p===null)) continue;
    let z=0; for (const p of ps) z+=p[2]; z/=ps.length;
    draw.push({it,ps,z});
  }
  draw.sort((a,b)=>b.z-a.z);

  for (const {it,ps} of draw){
    if (it.kind==="line"){
      ctx.beginPath();
      ctx.setLineDash(it.dash||[]);
      ctx.strokeStyle=it.color;
      ctx.lineWidth=it.width*Math.min(2.4,Math.max(0.85,9/it.z));
      ctx.moveTo(ps[0][0],ps[0][1]);
      ctx.lineTo(ps[1][0],ps[1][1]);
      ctx.stroke();
      ctx.setLineDash([]);
    } else if (it.kind==="poly"){
      ctx.beginPath();
      ctx.moveTo(ps[0][0],ps[0][1]);
      for (let i=1;i<ps.length;i++) ctx.lineTo(ps[i][0],ps[i][1]);
      ctx.closePath();
      if (it.fill){ctx.fillStyle=it.fill;ctx.fill();}
      if (it.stroke){ctx.strokeStyle=it.stroke;ctx.lineWidth=1.1;ctx.stroke();}
    } else {
      const p=ps[0];
      const r=Math.max(2.2, Math.min(9, it.size*Math.min(1.35, 14/Math.max(p[2],1))));
      ctx.beginPath(); ctx.arc(p[0],p[1],r,0,6.2832);
      ctx.fillStyle=it.color; ctx.fill();
      ctx.strokeStyle="rgba(0,0,0,0.6)"; ctx.lineWidth=1; ctx.stroke();
    }
  }

  if (opt.label){
    ctx.textAlign="center"; ctx.textBaseline="middle";
    for (const L of labels){
      const p=proj(L.p); if(!p) continue;
      const px = L.size * Math.min(1.15, Math.max(0.85, 26/Math.max(p[2],1)));
      ctx.font=`${L.bold?700:500} ${px}px "Segoe UI",system-ui,sans-serif`;
      ctx.lineWidth=3.2; ctx.strokeStyle="rgba(0,0,0,0.8)";
      ctx.strokeText(L.text,p[0],p[1]);
      ctx.fillStyle=L.color; ctx.fillText(L.text,p[0],p[1]);
    }
  }

  // numbered ID badges, always on top of the geometry
  if (opt.ids){
    for (const b of badges){
      const p=proj(b.p); if (!p) continue;
      const r=b.hl?13.5:11;
      ctx.beginPath(); ctx.arc(p[0],p[1],r,0,6.2832);
      ctx.fillStyle=b.color; ctx.fill();
      ctx.lineWidth=b.hl?3:2; ctx.strokeStyle="rgba(6,10,16,0.92)"; ctx.stroke();
      ctx.font=`700 ${b.hl?13:11.5}px "Segoe UI",system-ui,sans-serif`;
      ctx.textAlign="center"; ctx.textBaseline="middle";
      ctx.fillStyle="#08111c";
      ctx.fillText(String(b.n), p[0], p[1]+0.5);
    }
  }

  // camera name labels: aligned away from the badge so text never covers it
  if (opt.label){
    for (const L of camLabels){
      const p=proj(L.p), a=proj(L.anchor);
      if (!p) continue;
      const left = !a || p[0] >= a[0];
      const px = L.size * Math.min(1.15, Math.max(0.85, 26/Math.max(p[2],1)));
      ctx.font=`${L.bold?700:600} ${px}px "Segoe UI",system-ui,sans-serif`;
      // keep the text clear of the left info panel and the right control panel
      const tw = ctx.measureText(L.text).width;
      let align = left ? "left" : "right";
      if (align === "left" && p[0] + tw > W - 292) align = "right";
      if (align === "right" && p[0] - tw < 348) align = "left";
      let tx = p[0];
      if (align === "left") tx = Math.max(350, Math.min(tx, W - 294 - tw));
      else                  tx = Math.min(W - 292, Math.max(tx, 350 + tw));
      ctx.textAlign = align;
      ctx.textBaseline="middle";
      ctx.lineWidth=3.2; ctx.strokeStyle="rgba(0,0,0,0.82)";
      ctx.strokeText(L.text, p[0], p[1]);
      ctx.fillStyle=L.color; ctx.fillText(L.text, p[0], p[1]);
    }
  }

  // scale bar on the floor
  const sb=proj([-2.5,-HY+0.6,0.02]), se=proj([2.5,-HY+0.6,0.02]);  if (sb&&se){
    ctx.strokeStyle="rgba(205,220,240,0.7)"; ctx.lineWidth=1.6;
    ctx.beginPath();
    ctx.moveTo(sb[0],sb[1]); ctx.lineTo(se[0],se[1]);
    ctx.moveTo(sb[0],sb[1]-4); ctx.lineTo(sb[0],sb[1]+4);
    ctx.moveTo(se[0],se[1]-4); ctx.lineTo(se[0],se[1]+4);
    ctx.stroke();
    ctx.fillStyle="rgba(205,220,240,0.85)"; ctx.font="10px Consolas,monospace";
    ctx.textAlign="center"; ctx.fillText("5 m",(sb[0]+se[0])/2,sb[1]-9);
  }

  document.getElementById("vdist").textContent = smooth.dist.toFixed(0);
  requestAnimationFrame(frame);
}

/* ---------- interaction ---------- */
let drag=null;
function resize(){
  DPR=Math.min(2,window.devicePixelRatio||1);
  W=cv.clientWidth; H=cv.clientHeight;
  cv.width=Math.round(W*DPR); cv.height=Math.round(H*DPR);
}
window.addEventListener("resize",resize);
cv.addEventListener("contextmenu",e=>e.preventDefault());

cv.addEventListener("pointerdown",e=>{
  cv.setPointerCapture(e.pointerId); cv.classList.add("drag");
  drag={x:e.clientX,y:e.clientY,pan:(e.button===2||e.button===1||e.shiftKey)};
});
cv.addEventListener("pointermove",e=>{
  if(!drag) return;
  const dx=e.clientX-drag.x, dy=e.clientY-drag.y;
  drag.x=e.clientX; drag.y=e.clientY;
  if (drag.pan){
    const B2=basis(view), s=view.dist*0.0016;
    view.tgt=add(view.tgt, add(mul(B2.r,-dx*s), mul(B2.u,dy*s)));
  } else {
    view.yaw -= dx*0.0055;
    view.pitch=Math.max(-0.24,Math.min(1.47,view.pitch+dy*0.005));
  }
});
cv.addEventListener("pointerup",()=>{drag=null;cv.classList.remove("drag");});
cv.addEventListener("wheel",e=>{
  e.preventDefault();
  view.dist=Math.max(2.5,Math.min(90,view.dist*Math.exp(e.deltaY*0.0011)));
},{passive:false});

let pinch=null;
cv.addEventListener("touchstart",e=>{
  if(e.touches.length===2) pinch=Math.hypot(
    e.touches[0].clientX-e.touches[1].clientX, e.touches[0].clientY-e.touches[1].clientY);
},{passive:true});
cv.addEventListener("touchmove",e=>{
  if(e.touches.length===2&&pinch){
    const d=Math.hypot(e.touches[0].clientX-e.touches[1].clientX,
                       e.touches[0].clientY-e.touches[1].clientY);
    view.dist=Math.max(2.5,Math.min(90,view.dist*pinch/d)); pinch=d;
  }
},{passive:true});

cv.addEventListener("dblclick",e=>{
  let best=-1,bd=1e9;
  DATA.cameras.forEach((c,i)=>{
    const p=proj(c.pos); if(!p) return;
    const d=Math.hypot(p[0]-e.clientX,p[1]-e.clientY);
    if(d<bd){bd=d;best=i;}
  });
  if(best>=0&&bd<44) focusCam(best);
});

function focusCam(i){
  const c=DATA.cameras[i];
  opt.hl = opt.hl===i ? -1 : i;
  if (opt.hl>=0){
    view.tgt=[c.pos[0]*0.5,c.pos[1]*0.5,3.2];
    view.dist=Math.max(10,c.dist*1.45);
    view.pitch=0.26;
    view.yaw=Math.atan2(c.pos[1],c.pos[0])+0.6;
  }
  paintLegend();
}

/* ---------- presets ---------- */
const $=id=>document.getElementById(id);
function goAll(){
  view.tgt=[0,0,HZ*0.46];
  view.yaw=0.72; view.pitch=0.27;
  view.dist=fitDist([HX,HY,HZ*0.5])*1.02;
  opt.hl=-1; paintLegend();
}
function goRig(){
  view.tgt=[0,0,4.0]; view.yaw=0.72; view.pitch=0.30; view.dist=24;
  opt.hl=-1; paintLegend();
}
$("bAll").onclick=goAll;
$("bRig").onclick=goRig;
$("bTop").onclick=()=>{view.tgt=[0,0,1.2];view.yaw=-Math.PI/2;view.pitch=1.42;
  view.dist=fitDist([HX,HY,1.2]);};
$("bSide").onclick=()=>{view.tgt=[0,0,HZ*0.5];view.yaw=Math.PI;view.pitch=0.04;
  view.dist=fitDist([HX,HY,HZ*0.5])*0.92;};
$("bFrame").onclick=()=>{view.tgt=[0,0,1.15];view.yaw=0.95;view.pitch=0.045;view.dist=19;};
$("bReset").onclick=goAll;
$("bSpin").onclick=e=>{opt.spin=!opt.spin;
  e.currentTarget.classList.toggle("on",opt.spin);};

document.addEventListener("keydown",e=>{
  const k=e.key.toLowerCase();
  if(k==="r"||k==="h") goAll();
  if(k==="t") $("bTop").click();
  if(k==="s") $("bSide").click();
});

const chk={cIds:"ids",cTraj:"traj",cHang:"hang",cMount:"mount",cVol:"vol",cLabel:"label"};
for(const id in chk){
  const el=$(id);
  el.onchange=()=>{opt[chk[id]]=el.checked;};
}

/* ---------- per-camera field-of-view controls ---------- */
function onFrustumChange(e){
  const el=e.currentTarget;
  frusOn[+el.dataset.cb]=el.checked;
  syncFrustumControls();
}
function setAllFrusta(on){
  frusOn.fill(on);
  if (on) opt.frus=true;
  syncFrustumControls();
}
function paintFrustumControls(){
  $("camFrus").innerHTML=DATA.cameras.map((c,i)=>`
    <label class="chk" style="--cam:${COLORS[i%COLORS.length]}">
      <input type="checkbox" id="cFov_${c.name}" data-cb="${i}"
             aria-label="${c.name} 视野" checked> ${c.name}
    </label>`).join("");
  $("camFrus").querySelectorAll("input[data-cb]").forEach(el=>{
    el.onchange=onFrustumChange;
  });
}
function syncFrustumControls(){
  const on=frusOn.filter(Boolean).length;
  $("cFrus").checked=opt.frus;
  document.querySelectorAll("input[data-cb]").forEach(el=>{
    el.checked=frusOn[+el.dataset.cb];
    el.disabled=!opt.frus;
    el.closest("label").classList.toggle("off",!opt.frus||!el.checked);
  });
  const ab=$("cbAll");
  if (ab){
    ab.checked=on===frusOn.length;
    ab.indeterminate=on>0&&on<frusOn.length;
    ab.disabled=!opt.frus;
  }
  $("frusCount").textContent=`${opt.frus?on:0} / ${frusOn.length}`;
  $("frusHint").textContent=opt.frus
    ? "只切换视野，不隐藏相机。"
    : "总开关关闭，单机选择已保留。";
}
$("cFrus").onchange=e=>{opt.frus=e.currentTarget.checked;syncFrustumControls();};
$("bFrAll").onclick=()=>setAllFrusta(true);
$("bFrNone").onclick=()=>setAllFrusta(false);

/* ---------- legend ---------- */
function paintLegend(){
  const tb=$("tbl");
  tb.innerHTML = `
    <tr style="color:#5c6879;font-size:9.5px;letter-spacing:.5px">
      <td></td><td>ID</td>
      <td class="cb all"><label title="全选 / 清空相机视野">
        <input type="checkbox" id="cbAll" aria-label="全选相机视野（图例）">视野</label></td>
      <td class="num">方位</td><td class="num">高度</td>
      <td class="num">俯角</td><td class="num">体心距</td><td class="num">HFOV</td>
    </tr>` +
    DATA.cameras.map((c,i)=>`
    <tr class="${opt.hl===i?'hl':''}" data-i="${i}" style="cursor:pointer">
      <td><span class="sw" style="background:${COLORS[i%6]}"></span></td>
      <td class="nm">${c.name}</td>
      <td class="cb"><label title="${c.name} 视野">
        <input type="checkbox" data-cb="${i}" aria-label="${c.name} 视野（图例）"></label></td>
      <td class="num">${c.az.toFixed(0)}°</td>
      <td class="num">${c.pos[2].toFixed(2)}</td>
      <td class="num">${c.tilt.toFixed(1)}°</td>
      <td class="num">${c.dist.toFixed(2)}</td>
      <td class="num">${c.hfov.toFixed(1)}°</td>
    </tr>`).join("");

  // Both control locations share the same per-camera state.
  $("cbAll").onchange=e=>setAllFrusta(e.currentTarget.checked);
  tb.querySelectorAll("input[data-cb]").forEach(el=>{
    el.onchange=onFrustumChange;
  });

  tb.querySelectorAll("tr").forEach(tr=>{
    const i=tr.dataset.i;
    if (i===undefined) return;
    tr.onclick=e=>{ if(!e.target.closest(".cb")) focusCam(+i); };
    tr.onmouseenter=()=>{opt.hl=+i;};
    tr.onmouseleave=()=>{opt.hl=-1;};
  });
  syncFrustumControls();
}

/* ---------- boot ---------- */
resize();
paintFrustumControls();
goAll();
window.viz = { DATA, opt, view, smooth, COLORS, frusOn, get badges(){ return badges; } };
requestAnimationFrame(frame);
</script>
</body>
</html>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    data = collect()
    cams = data["cameras"]

    bl = 0.0
    for i in range(len(cams)):
        for j in range(i + 1, len(cams)):
            bl = max(bl, math.dist(cams[i]["pos"], cams[j]["pos"]))
    tilt = max(c["tilt"] for c in cams)
    hfov = cams[0]["hfov"]
    radius = sum(c["radius"] for c in cams) / len(cams)
    height = max(c["pos"][2] for c in cams)
    hangar = data["hangar"]
    volume = data["volume"]["size"]

    html = (HTML
            .replace("__DATA__", json.dumps(data, separators=(",", ":")))
            .replace("__BASELINE__", f"{bl:.1f}")
            .replace("__TILT__", f"{tilt:.0f}°")
            .replace("__HFOV__", f"{hfov:.1f}°")
            .replace("__RADIUS__", f"{radius:.1f}")
            .replace("__HEIGHT__", f"{height:.2f}")
            .replace("__MOUNTZ__", f"{data['mount_z']:.2f}")
            .replace("__ROOFZ__", f"{data['roof_z']:.2f}")
            .replace("__RAILZ__", f"{data['rail_z']:.2f}")
            .replace("__HANGAR__", f"{hangar['x']:g} × {hangar['y']:g} × {hangar['z']:g}")
            .replace("__VOLUME__", f"{volume[0]:g} × {volume[1]:g} × {volume[2]:g}"))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(html)

    n_traj = sum(len(t["points"]) for t in data["trajectories"])
    print(f"wrote {args.out}  ({os.path.getsize(args.out)/1024:.0f} KiB, "
          f"{len(cams)} cameras, {len(data['trajectories'])} bodies, "
          f"{n_traj} trajectory points)")
    print(f"baseline {bl:.2f} m, tilt {tilt:.1f} deg, hfov {hfov:.2f} deg")


if __name__ == "__main__":
    main()
